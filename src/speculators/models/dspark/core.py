from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.metrics import compute_metrics
from speculators.models.dspark.model_definitions import ConfidenceHead, MarkovHead
from speculators.models.metrics import LossConfig, kl_div_loss, resolve_loss_config
from speculators.models.utils import conditional_torch_compile

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}

__all__ = [
    "DSparkDraftModel",
]


@SpeculatorModel.register("dspark")
class DSparkDraftModel(DFlashDraftModel):
    """DFlash backbone plus a Markov logit-bias head and a confidence head.

    After the base draft logits are produced, the Markov head biases position
    ``k`` using the previous block token and the confidence head predicts each
    position's acceptance probability. Everything else is inherited from DFlash.
    """

    config_class: ClassVar[type[DSparkSpeculatorConfig]] = DSparkSpeculatorConfig  # type: ignore[misc,assignment]

    def __init__(self, config: DSparkSpeculatorConfig) -> None:
        super().__init__(config=config)

        hidden_size = config.transformer_layer_config.hidden_size

        self.markov_head: MarkovHead | None = None
        if config.markov_rank > 0:
            self.markov_head = MarkovHead(
                verifier_vocab_size=self.verifier_vocab_size,
                draft_vocab_size=self.draft_vocab_size,
                markov_rank=config.markov_rank,
                hidden_size=hidden_size,
                head_type=config.markov_head_type,
            )

        self.confidence_head: ConfidenceHead | None = None
        if config.enable_confidence_head:
            if config.confidence_head_with_markov and self.markov_head is None:
                raise ValueError(
                    "confidence_head_with_markov=True requires markov_rank > 0."
                )
            input_dim = hidden_size + (
                config.markov_rank if config.confidence_head_with_markov else 0
            )
            self.confidence_head = ConfidenceHead(input_dim)

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DSparkDraftModel":
        """Create a DSpark model from training arguments (mirrors DFlash)."""
        config = DSparkSpeculatorConfig(
            **cls._build_base_config_kwargs("dspark", verifier_config, **kwargs),
            markov_rank=kwargs.get("markov_rank", 256),
            markov_head_type=kwargs.get("markov_head_type", "vanilla"),
            enable_confidence_head=kwargs.get("enable_confidence_head", True),
            confidence_head_with_markov=kwargs.get("confidence_head_with_markov", True),
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve DSpark's compound loss from ``--loss-fn``."""
        loss_config = resolve_loss_config(kwargs["loss_fn"])
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        confidence_head_alpha = kwargs.get("confidence_head_alpha", 1.0)
        cat_mode = kwargs.get("cat_mode", "none")
        sampled_acceptance_loss_alpha = kwargs.get(
            "sampled_acceptance_loss_alpha", 1.0
        )
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "confidence_head_alpha": confidence_head_alpha,
            "cat_mode": cat_mode,
            "sampled_acceptance_loss_alpha": sampled_acceptance_loss_alpha,
        }
        return dict(shared), dict(shared)

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # [1, total_seq_len, num_hidden*hidden_size]
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        confidence_head_alpha: float = 1.0,
        cat_mode: str = "none",
        sampled_draft_logprobs: torch.Tensor | None = None,
        sampled_target_logprobs: torch.Tensor | None = None,
        sampled_acceptance_loss_alpha: float = 1.0,
        # Phase 3 sampled acceptance: recompute q_logp from sampled_ids
        sampled_draft_ids: torch.Tensor | None = None,
        sampled_target_ids: torch.Tensor | None = None,
        sampled_anchor_pos: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden, logits, targets, aligned_loss_mask, anchored_block_indices = (
            self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                **kwargs,
            )
        )

        # DSpark: add the Markov logit bias and predict per-position confidence.
        num_blocks = self.config.max_anchors
        block = self.block_size
        mask_tokens_size = num_blocks * block
        # Ground-truth block tokens (verifier vocab); position 0 is the anchor.
        block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block)
        # prev_token_ids[:, k] is the token preceding draft position k within the block.
        prev_token_ids = torch.cat(
            [block_tokens[:, :1], block_tokens[:, :-1]], dim=1
        )  # [num_blocks, block]
        hidden_blocks = hidden.view(num_blocks, block, -1)

        confidence_logits = None
        prev_emb = None
        logits_base = logits  # Backbone logits before Markov bias (for sampled replay)
        if self.markov_head is not None:
            prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
            markov_bias = self.markov_head.block_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
                prev_emb=prev_emb,
            )
            logits = (logits.view(num_blocks, block, -1) + markov_bias).view(
                1, mask_tokens_size, -1
            )

        if self.confidence_head is not None:
            # confidence_head_with_markov requires markov_rank > 0 (enforced in
            # __init__), so prev_emb is always set when the flag is on.
            if self.config.confidence_head_with_markov and prev_emb is not None:
                conf_features = torch.cat(
                    [hidden_blocks, prev_emb.to(hidden_blocks.dtype)], dim=-1
                )
            else:
                conf_features = hidden_blocks
            confidence_logits = self.confidence_head(conf_features).reshape(
                1, mask_tokens_size
            )

        # Phase 3: Recompute sampled q_logp if sampled data present
        # This happens during training forward, after no-grad sampling + target verify
        if sampled_draft_ids is not None and sampled_anchor_pos is not None:
            sampled_draft_logprobs, sampled_target_logprobs = (
                self._recompute_sampled_qlogp(
                    hidden=hidden,
                    logits_base=logits_base,  # base logits before markov bias
                    input_ids=input_ids,
                    sampled_draft_ids=sampled_draft_ids,
                    sampled_target_ids=sampled_target_ids,
                    sampled_anchor_pos=sampled_anchor_pos,
                    sampled_target_logprobs=sampled_target_logprobs,
                    num_blocks=num_blocks,
                    block=block,
                )
            )

        loss, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            aligned_loss_mask,
            self.block_size,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            gamma=gamma,
            confidence_head_alpha=confidence_head_alpha,
            cat_mode=cat_mode,  # type: ignore[arg-type]
            sampled_draft_logprobs=sampled_draft_logprobs,
            sampled_target_logprobs=sampled_target_logprobs,
            sampled_acceptance_loss_alpha=sampled_acceptance_loss_alpha,
        )
        draft_tokens = torch.argmax(logits, dim=-1)
        return draft_tokens, loss, metrics

    def _recompute_sampled_qlogp(
        self,
        hidden: torch.Tensor,
        logits_base: torch.Tensor,
        input_ids: torch.Tensor,
        sampled_draft_ids: torch.Tensor,  # [K] draft vocab (for gather)
        sampled_target_ids: torch.Tensor,  # [K] target vocab (for prev history)
        sampled_anchor_pos: torch.Tensor,  # [1]
        sampled_target_logprobs: torch.Tensor,  # [K]
        num_blocks: int,
        block: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute draft q_logp for sampled trajectory (phase 3).

        Given detached sampled ids from phase-1 no-grad sampling, recompute
        q_logp using current model state. This is the ONLY gradient-enabled
        graph, matching the sampling logic in SampledAcceptanceAugmentor:
        - prev token history uses TARGET ids (as during sampling)
        - q_logp is gathered at the DRAFT id (the token sampled from q)

        Returns:
            sampled_draft_logprobs: [1, K] with gradient (through markov head)
            sampled_target_logprobs: [1, K] detached
        """
        device = hidden.device
        anchor_pos = int(sampled_anchor_pos[0].item())
        K = int(sampled_draft_ids.shape[0])

        # The sampling used max_anchors=1, producing a single block from the
        # anchor. Here the training forward may have many blocks; the anchor's
        # block is the one whose slot 0 sits at anchor_pos.
        block_idx = anchor_pos // block
        block_idx = min(block_idx, num_blocks - 1)

        hidden_block = hidden.view(num_blocks, block, -1)[block_idx : block_idx + 1]
        logits_block = logits_base.view(num_blocks, block, -1)[
            block_idx : block_idx + 1
        ]

        anchor_token = int(input_ids[0, anchor_pos].item())

        sampled_draft_logprobs_list = []
        for k in range(K):
            slot = k + 1
            if slot >= block:
                break
            # Rebuild prev history exactly as during sampling: anchor token at
            # position 0, previously sampled TARGET tokens at positions 1..k.
            prev_token_ids = torch.full(
                (1, block), anchor_token, dtype=torch.long, device=device
            )
            if k > 0:
                prev_token_ids[0, 1 : 1 + k] = sampled_target_ids[:k]

            biased_logits = logits_block
            if self.markov_head is not None:
                prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
                biased_logits = logits_block + self.markov_head.block_bias(
                    prev_token_ids=prev_token_ids,
                    hidden_states=hidden_block,
                    prev_emb=prev_emb,
                )

            slot_logits = biased_logits[0, slot].float()
            log_probs = torch.log_softmax(slot_logits, dim=-1)
            token_logp = log_probs[int(sampled_draft_ids[k].item())]
            sampled_draft_logprobs_list.append(token_logp)

        sampled_draft_logprobs = torch.stack(sampled_draft_logprobs_list).unsqueeze(0)
        sampled_target_logprobs_out = sampled_target_logprobs[
            : len(sampled_draft_logprobs_list)
        ].unsqueeze(0)

        return sampled_draft_logprobs, sampled_target_logprobs_out

