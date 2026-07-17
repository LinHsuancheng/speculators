"""On-policy sampled acceptance-loss batch augmentation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import openai
import torch

from speculators.data_generation.vllm_client import (
    DEFAULT_REQUEST_TIMEOUT,
    score_sampled_tokens,
)
from speculators.models.attention import create_float_mask
from speculators.models.dspark.core import DSparkDraftModel

logger = logging.getLogger("speculators")


def _has_dtensor_params(model: torch.nn.Module) -> bool:
    """Detect whether the model is wrapped with FSDP2 (fully_shard).

    FSDP2 converts parameters to DTensor but does NOT change the model type,
    so isinstance(model, FSDP) fails. Detect by checking for DTensor params.
    """
    for param in model.parameters():
        if isinstance(param, DTensor):
            return True
    return False


@dataclass(frozen=True)
class SampledAcceptanceConfig:
    vllm_endpoint: str
    model: str | None = None
    prompt_logprobs: int = 1
    request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT
    hidden_states_file_timeout: float = 30.0
    temperature: float = 0.0
    max_anchors: int = 1


class SampledAcceptanceAugmentor:
    """Add sampled draft/target logprobs to a DSpark training batch."""

    def __init__(self, config: SampledAcceptanceConfig) -> None:
        self.config = config
        self.skipped_no_anchor = 0
        self.client = openai.OpenAI(
            base_url=config.vllm_endpoint,
            api_key="EMPTY",
            max_retries=0,
        )
        self.model_id = config.model or self.client.models.list().data[0].id

    def __call__(
        self,
        model: torch.nn.Module,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        """Augment batch with sampled acceptance logprobs.

        In distributed training, only rank 0 samples and calls vLLM.
        Results are broadcast to all ranks to ensure consistent gradients.
        """
        import torch.distributed as dist

        is_distributed = dist.is_initialized()
        is_rank0 = not is_distributed or dist.get_rank() == 0

        # Only rank 0 performs sampling
        if is_rank0:
            draft_model = model.module if hasattr(model, "module") else model
            if not isinstance(draft_model, DSparkDraftModel):
                raise TypeError(
                    "SampledAcceptanceAugmentor currently supports DSparkDraftModel only"
                )

            self._ensure_sdpa_mask(draft_model)
            if not self._has_valid_anchor(
                batch["loss_mask"], int(draft_model.block_size)
            ):
                self.skipped_no_anchor += 1
                if self.skipped_no_anchor <= 5:
                    logger.warning(
                        "Skipping batch for sampled acceptance loss: no valid anchor "
                        f"position (skipped {self.skipped_no_anchor} so far)"
                    )
                # Broadcast skip signal to other ranks
                if is_distributed:
                    skip_flag = torch.tensor([1], dtype=torch.int32, device="cuda")
                    dist.broadcast(skip_flag, src=0)
                return batch

            # Signal other ranks that we're proceeding
            if is_distributed:
                skip_flag = torch.tensor([0], dtype=torch.int32, device="cuda")
                dist.broadcast(skip_flag, src=0)

            sample = self._sample_from_draft(draft_model, batch)
            scored = score_sampled_tokens(
                client=self.client,
                model=self.model_id,
                prefix_token_ids=sample["prefix_token_ids"],
                sampled_token_ids=sample["sampled_target_token_ids"],
                prompt_logprobs=self.config.prompt_logprobs,
                timeout=self.config.request_timeout,
                cleanup_hidden_states=True,
                hidden_states_file_timeout=self.config.hidden_states_file_timeout,
            )
            sampled_draft_logprobs = sample["draft_logprobs"].unsqueeze(0)
            sampled_target_logprobs = torch.tensor(
                scored["token_logprobs"],
                device=sample["draft_logprobs"].device,
                dtype=sample["draft_logprobs"].dtype,
            ).unsqueeze(0)
        else:
            # Other ranks: wait for skip signal
            skip_flag = torch.tensor([0], dtype=torch.int32, device="cuda")
            dist.broadcast(skip_flag, src=0)
            if skip_flag.item() == 1:
                return batch

            # Receive broadcasted logprobs from rank 0
            # Create placeholder tensors (shape will be broadcast)
            device = next(model.parameters()).device
            # Assume K=7 (block_size - 1), adjust if needed
            K = 7  # TODO: get from model config
            sampled_draft_logprobs = torch.zeros((1, K), dtype=torch.float32, device=device)
            sampled_target_logprobs = torch.zeros((1, K), dtype=torch.float32, device=device)

        # Broadcast logprobs from rank 0 to all ranks
        if is_distributed:
            dist.broadcast(sampled_draft_logprobs, src=0)
            dist.broadcast(sampled_target_logprobs, src=0)

        batch["sampled_draft_logprobs"] = sampled_draft_logprobs
        batch["sampled_target_logprobs"] = sampled_target_logprobs
        batch["anchor_valid"] = sample["anchor_valid"]
        return batch

    @staticmethod
    def _ensure_sdpa_mask(model: DSparkDraftModel) -> None:
        if getattr(model, "_attn_impl", None) == "sdpa":
            model._create_mask_fn = create_float_mask  # noqa: SLF001

    @staticmethod
    def _target_token_id(model: DSparkDraftModel, draft_token_id: int) -> int:
        if model.d2t is None:
            return draft_token_id
        return int(model.d2t[draft_token_id].item())

    @classmethod
    def _valid_anchor_positions(
        cls,
        loss_mask: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        valid_positions = torch.nonzero(loss_mask[0].bool(), as_tuple=False).flatten()
        return valid_positions[valid_positions + block_size - 1 < loss_mask.shape[1]]

    @classmethod
    def _has_valid_anchor(cls, loss_mask: torch.Tensor, block_size: int) -> bool:
        return cls._valid_anchor_positions(loss_mask, block_size).numel() > 0

    @classmethod
    def _sample_anchor_position(
        cls,
        loss_mask: torch.Tensor,
        block_size: int,
    ) -> int:
        valid_positions = cls._valid_anchor_positions(loss_mask, block_size)
        if valid_positions.numel() == 0:
            raise ValueError("No valid anchor position for sampled acceptance loss")
        return int(valid_positions[0].item())

    def _sample_from_draft(
        self,
        model: DSparkDraftModel,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        device = next(model.parameters()).device
        input_ids = batch["input_ids"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        hidden_states = batch["hidden_states"].to(device)
        verifier_last_hidden_states = batch["verifier_last_hidden_states"].to(device)
        document_ids = batch["document_ids"].to(device)
        position_ids = batch.get("position_ids")
        if position_ids is not None:
            position_ids = position_ids.to(device)

        block_size = int(model.block_size)
        anchor_pos = self._sample_anchor_position(loss_mask, block_size)
        sampled_len = block_size - 1
        anchor_loss_mask = torch.zeros_like(loss_mask, dtype=torch.bool)
        anchor_loss_mask[:, anchor_pos] = True
        anchor_positions = torch.tensor([anchor_pos], dtype=torch.long, device=device)
        anchor_valid = torch.ones(1, dtype=torch.bool, device=device)

        old_max_anchors = model.config.max_anchors
        model.config.max_anchors = self.config.max_anchors

        # Sample from draft model (only called on rank 0, no DTensor issues)
        with torch.no_grad():
            hidden, logits, _, _, _ = model._backbone_forward(
                hidden_states,
                input_ids,
                anchor_loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                anchor_positions=anchor_positions,
                anchor_valid=anchor_valid,
            )

        model.config.max_anchors = old_max_anchors

        hidden_blocks = hidden.view(1, block_size, -1)
        logits_blocks = logits.view(1, block_size, -1)
        base_prev_token_ids = torch.full(
            (1, block_size),
            int(input_ids[0, anchor_pos].item()),
            dtype=torch.long,
            device=device,
        )

        sampled_target_ids: list[int] = []
        sampled_draft_ids: list[int] = []
        draft_logprobs: list[torch.Tensor] = []
        for slot in range(1, sampled_len + 1):
            prev_token_ids = base_prev_token_ids.clone()
            if sampled_target_ids:
                prev_token_ids[0, 1 : 1 + len(sampled_target_ids)] = torch.tensor(
                    sampled_target_ids,
                    dtype=torch.long,
                    device=device,
                )
            biased_logits = logits_blocks
            if model.markov_head is not None:
                prev_emb = model.markov_head.prev_embeddings(prev_token_ids)
                biased_logits = biased_logits + model.markov_head.block_bias(
                    prev_token_ids=prev_token_ids,
                    hidden_states=hidden_blocks,
                    prev_emb=prev_emb,
                )

            slot_logits = biased_logits[0, slot].float()
            if self.config.temperature <= 0:
                log_probs = torch.log_softmax(slot_logits, dim=-1)
                draft_token_id = int(torch.argmax(slot_logits).item())
            else:
                scaled = slot_logits / self.config.temperature
                log_probs = torch.log_softmax(scaled, dim=-1)
                draft_token_id = int(
                    torch.multinomial(torch.softmax(scaled, dim=-1), 1).item()
                )
            target_token_id = self._target_token_id(model, draft_token_id)
            sampled_draft_ids.append(draft_token_id)
            sampled_target_ids.append(target_token_id)
            draft_logprobs.append(log_probs[draft_token_id].to(logits.dtype))

        prefix_token_ids = input_ids[0, : anchor_pos + 1].tolist()
        return {
            "prefix_token_ids": prefix_token_ids,
            "sampled_target_token_ids": sampled_target_ids,
            "sampled_draft_token_ids": sampled_draft_ids,
            "draft_logprobs": torch.stack(draft_logprobs),
            "anchor_positions": anchor_positions,
            "anchor_valid": anchor_valid,
        }
