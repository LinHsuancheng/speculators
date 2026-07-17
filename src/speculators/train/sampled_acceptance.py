"""On-policy sampled acceptance-loss batch augmentation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import openai
import torch
import torch.distributed as dist

from speculators.data_generation.vllm_client import (
    DEFAULT_REQUEST_TIMEOUT,
    score_sampled_tokens,
)
from speculators.models.attention import create_float_mask
from speculators.models.dflash.utils import select_anchors
from speculators.models.dspark.core import DSparkDraftModel

logger = logging.getLogger("speculators")


@dataclass(frozen=True)
class SampledAcceptanceConfig:
    vllm_endpoint: str
    model: str | None = None
    prompt_logprobs: int = 1
    request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT
    hidden_states_file_timeout: float = 30.0
    temperature: float = 1.0
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
        """Augment batch with sampled acceptance data.

        Three-phase approach to avoid FSDP2 graph lifecycle issues:
        1. No-grad sampling: only save detached sampled_ids
        2. Target verify: external vLLM call, detached logprobs
        3. Training forward will recompute sampled q_logp using sampled_ids

        This ensures only one backward graph exists at backward time.
        """
        draft_model = model.module if hasattr(model, "module") else model
        if not isinstance(draft_model, DSparkDraftModel):
            raise TypeError(
                "SampledAcceptanceAugmentor currently supports DSparkDraftModel only"
            )

        self._ensure_sdpa_mask(draft_model)
        block_size = int(draft_model.block_size)
        anchor_positions, anchor_valid = select_anchors(
            batch["loss_mask"],
            int(draft_model.config.max_anchors),
            block_size,
        )
        has_anchor = torch.tensor(
            [int(bool(anchor_valid.any()))],
            device=batch["loss_mask"].device,
            dtype=torch.int,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(has_anchor, op=dist.ReduceOp.MIN)
        if not bool(has_anchor.item()):
            self.skipped_no_anchor += 1
            if self.skipped_no_anchor <= 5:
                logger.warning(
                    "Skipping batch for sampled acceptance loss: no valid anchor "
                    "on at least one rank "
                    f"(skipped {self.skipped_no_anchor} so far)"
                )
            return batch
        batch["anchor_positions"] = anchor_positions
        batch["anchor_valid"] = anchor_valid
        valid_anchor_indices = torch.nonzero(anchor_valid, as_tuple=False).flatten()
        sampled_anchor_index = int(valid_anchor_indices[0].item())
        anchor_pos = int(anchor_positions[sampled_anchor_index].item())

        # Phase 1: No-grad sampling (only discrete token IDs)
        with torch.no_grad():
            sample = self._sample_from_draft(draft_model, batch, anchor_pos=anchor_pos)

        # Extract and detach discrete results
        sampled_target_ids = [int(x) for x in sample["sampled_target_token_ids"]]
        sampled_draft_ids = [int(x) for x in sample["sampled_draft_token_ids"]]
        anchor_pos = int(sample["anchor_positions"][0].item())

        # Phase 2: Target verify (external, detached)
        scored = score_sampled_tokens(
            client=self.client,
            model=self.model_id,
            prefix_token_ids=sample["prefix_token_ids"],
            sampled_token_ids=sampled_target_ids,
            prompt_logprobs=self.config.prompt_logprobs,
            timeout=self.config.request_timeout,
            cleanup_hidden_states=True,
            hidden_states_file_timeout=self.config.hidden_states_file_timeout,
        )

        # Store detached data for phase 3 (training forward will recompute q_logp)
        # Note: sampled_draft_ids are in DRAFT vocab (for q_logp recompute),
        # sampled_target_ids are in TARGET vocab (used for vLLM verify).
        device = batch["input_ids"].device
        batch["sampled_draft_ids"] = torch.tensor(
            [sampled_draft_ids], dtype=torch.long, device=device
        )
        batch["sampled_target_ids"] = torch.tensor(
            [sampled_target_ids], dtype=torch.long, device=device
        )
        batch["sampled_target_logprobs"] = torch.tensor(
            [scored["token_logprobs"]], dtype=torch.float32, device=device
        )
        batch["sampled_anchor_pos"] = torch.tensor(
            [anchor_pos], dtype=torch.long, device=device
        )
        batch["sampled_anchor_index"] = torch.tensor(
            [sampled_anchor_index], dtype=torch.long, device=device
        )
        return batch

    @staticmethod
    def _ensure_sdpa_mask(model: DSparkDraftModel) -> None:
        if getattr(model, "_attn_impl", None) == "sdpa":
            model._create_mask_fn = create_float_mask  # noqa: SLF001

    @staticmethod
    def _target_token_id(model: DSparkDraftModel, draft_token_id: int) -> int:
        if model.d2t is None:
            return draft_token_id
        # d2t stores an offset, matching Eagle/DFlash convention:
        # target_token_id = draft_token_id + d2t[draft_token_id].
        return int(draft_token_id + model.d2t[draft_token_id].item())

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

    @staticmethod
    def _document_prefix_start(document_ids: torch.Tensor, anchor_pos: int) -> int:
        anchor_doc = document_ids[0, anchor_pos]
        same_doc = document_ids[0, : anchor_pos + 1] == anchor_doc
        positions = torch.nonzero(same_doc, as_tuple=False).flatten()
        if positions.numel() == 0:
            return 0
        return int(positions[0].item())

    def _sample_from_draft(
        self,
        model: DSparkDraftModel,
        batch: dict[str, Any],
        anchor_pos: int | None = None,
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
        if anchor_pos is None:
            anchor_pos = self._sample_anchor_position(loss_mask, block_size)
        sampled_len = block_size - 1
        anchor_loss_mask = torch.zeros_like(loss_mask, dtype=torch.bool)
        anchor_loss_mask[:, anchor_pos] = True
        anchor_positions = torch.tensor([anchor_pos], dtype=torch.long, device=device)
        anchor_valid = torch.ones(1, dtype=torch.bool, device=device)

        old_max_anchors = model.config.max_anchors
        model.config.max_anchors = self.config.max_anchors
        # Registered as an FSDP forward method after fully_shard(); direct
        # _backbone_forward calls bypass FSDP unshard/reshard hooks.
        try:
            hidden, logits, _, _, _ = model.get_backbone_outputs(
                hidden_states,
                input_ids,
                anchor_loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                anchor_positions=anchor_positions,
                anchor_valid=anchor_valid,
            )
        finally:
            model.config.max_anchors = old_max_anchors

        hidden_blocks = hidden.view(1, block_size, -1)
        logits_blocks = logits.view(1, block_size, -1)
        anchor_token_id = int(input_ids[0, anchor_pos].item())

        sampled_target_ids: list[int] = []
        sampled_draft_ids: list[int] = []
        draft_logprobs: list[torch.Tensor] = []
        for slot in range(1, sampled_len + 1):
            prev_token_ids = torch.full(
                (1, block_size),
                anchor_token_id,
                dtype=torch.long,
                device=device,
            )
            if sampled_target_ids:
                prev_token_ids[0, 2 : 2 + len(sampled_target_ids)] = torch.tensor(
                    sampled_target_ids[: block_size - 2],
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

        doc_start = self._document_prefix_start(document_ids, anchor_pos)
        prefix_token_ids = input_ids[0, doc_start : anchor_pos + 1].tolist()
        return {
            "prefix_token_ids": prefix_token_ids,
            "sampled_target_token_ids": sampled_target_ids,
            "sampled_draft_token_ids": sampled_draft_ids,
            "draft_logprobs": torch.stack(draft_logprobs),
            "anchor_positions": anchor_positions,
            "anchor_valid": anchor_valid,
        }
