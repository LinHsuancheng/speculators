from typing import ClassVar

import torch
from transformers import PretrainedConfig

from speculators.model import SpeculatorModel
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.metrics import compute_metrics
from speculators.models.dspark.model_definitions import ConfidenceHead, MarkovHead
from speculators.models.dspark.onpolicy import VerifierScorer
from speculators.models.metrics import LossConfig, kl_div_loss, resolve_loss_config
from speculators.models.utils import conditional_torch_compile

# Draft sampling temperature is floored to this value during on-policy training.
# The estimator needs a differentiable, full-support proposal q_psi; a strictly
# greedy (T=0) draft is a point mass with no usable log q_t gradient. To target a
# greedy serving regime, train with a small positive temperature (T->0 limit).
_MIN_SAMPLING_TEMPERATURE = 1e-2

# Floor for the uniform draws feeding the Gumbel-max sampler, so log(-log(u)) is
# always finite.
_EPS_ROLLOUT = 1e-9

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

        # On-policy scorer. Not a submodule / not saved: it wraps the frozen
        # verifier (vLLM) that scores sampled draft sequences to produce
        # ``p_t(Y_t)``. The trainer injects it before training when the
        # exact-acceptance-length objective is enabled; it stays ``None`` for the
        # teacher-forced path so existing behaviour is unchanged.
        self.verifier_scorer: VerifierScorer | None = None

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
        # On-policy exact-acceptance-length objective. Enabled when the loss spec
        # contains the ``accept_length`` term (see resolve_loss_config); the
        # trainer must also inject a ``verifier_scorer`` for it to activate.
        onpolicy_sampling = "accept_length" in loss_config
        sampling_temperature = float(kwargs.get("sampling_temperature", 1.0))
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "confidence_head_alpha": confidence_head_alpha,
            "cat_mode": cat_mode,
            "onpolicy_sampling": onpolicy_sampling,
            "sampling_temperature": sampling_temperature,
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
        onpolicy_sampling: bool = False,
        sampling_temperature: float = 1.0,
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

        num_blocks = self.config.max_anchors
        block = self.block_size
        mask_tokens_size = num_blocks * block
        # Ground-truth block tokens (verifier vocab); position 0 is the anchor.
        block_tokens = input_ids[0, anchored_block_indices].view(num_blocks, block)
        hidden_blocks = hidden.view(num_blocks, block, -1)
        # base_logits are the DFlash draft logits *before* the Markov bias. The
        # teacher-forced path adds a gold-conditioned bias; the on-policy rollout
        # instead adds a bias conditioned on its own sampled tokens.
        base_logits = logits.view(num_blocks, block, -1)

        if onpolicy_sampling:
            return self._onpolicy_forward(
                base_logits=base_logits,
                hidden_blocks=hidden_blocks,
                block_tokens=block_tokens,
                targets=targets,
                aligned_loss_mask=aligned_loss_mask,
                anchor_positions=anchored_block_indices[:: self.block_size],
                input_ids=input_ids,
                document_ids=document_ids,
                loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
                confidence_head_alpha=confidence_head_alpha,
                temperature=sampling_temperature,
            )

        # --- Teacher-forced path (unchanged) ---------------------------------
        # prev_token_ids[:, k] is the token preceding draft position k within the block.
        prev_token_ids = torch.cat(
            [block_tokens[:, :1], block_tokens[:, :-1]], dim=1
        )  # [num_blocks, block]

        confidence_logits = None
        prev_emb = None
        logits_bias = base_logits
        if self.markov_head is not None:
            prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
            markov_bias = self.markov_head.block_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
                prev_emb=prev_emb,
            )
            logits_bias = base_logits + markov_bias
        logits = logits_bias.view(1, mask_tokens_size, -1)

        if self.confidence_head is not None:
            confidence_logits = self._confidence_logits(
                hidden_blocks, prev_emb, mask_tokens_size
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
        )
        draft_tokens = torch.argmax(logits, dim=-1)
        return draft_tokens, loss, metrics

    def _confidence_logits(
        self,
        hidden_blocks: torch.Tensor,  # [num_blocks, block, hidden]
        prev_emb: torch.Tensor | None,  # [num_blocks, block, r] or None
        mask_tokens_size: int,
    ) -> torch.Tensor:
        """Run the confidence head over the block features -> [1, mask_tokens_size]."""
        # confidence_head_with_markov requires markov_rank > 0 (enforced in
        # __init__), so prev_emb is always set when the flag is on.
        if self.config.confidence_head_with_markov and prev_emb is not None:
            conf_features = torch.cat(
                [hidden_blocks, prev_emb.to(hidden_blocks.dtype)], dim=-1
            )
        else:
            conf_features = hidden_blocks
        return self.confidence_head(conf_features).reshape(1, mask_tokens_size)  # type: ignore[misc]

    def _sample_block_rollout(
        self,
        base_logits: torch.Tensor,  # [num_blocks, block, draft_vocab]
        hidden_blocks: torch.Tensor,  # [num_blocks, block, hidden]
        anchor_token_ids: torch.Tensor,  # [num_blocks] verifier-vocab anchor token
        temperature: float,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Autoregressively sample the ``K = block_size - 1`` draft tokens per block.

        Slot 0 is the anchor (always emitted, not sampled). For slot ``k`` the
        Markov bias is conditioned on the token *sampled* at slot ``k-1`` (in
        verifier vocabulary), so the proposal is the true joint
        ``prod_t q_t(Y_t | Y_{<t})`` used at serving time — not the gold-conditioned
        teacher-forced factorization.

        Sampling uses the Gumbel-max trick (``argmax(logits/T + Gumbel)``) rather
        than ``torch.multinomial``: it is numerically simple, has a clean greedy
        ``T -> 0`` limit, and avoids the categorical-sampler dtype constraints that
        bite on Ascend NPU. Only ``log q_t(Y_t)`` carries gradient.

        Args:
            base_logits: DFlash draft logits before the Markov bias.
            hidden_blocks: Backbone hidden state per slot (for the gated head).
            anchor_token_ids: Slot-0 gold token per block (verifier vocab); it is
                the ``prev`` token for slot 1.
            temperature: Sampling temperature ``T`` (floored to
                ``_MIN_SAMPLING_TEMPERATURE``).
            generator: Optional RNG for reproducible sampling (tests).

        Returns:
            ``(sampled_draft_ids, q_logp, sampled_verifier_ids)`` each shaped
            ``[num_blocks, K]``. ``sampled_draft_ids`` are draft-vocab tokens,
            ``q_logp`` is ``log q_t(Y_t)`` (grad flows), ``sampled_verifier_ids``
            are the same tokens mapped to verifier vocab (for the scorer).
        """
        num_blocks, block, _ = base_logits.shape
        temp = max(float(temperature), _MIN_SAMPLING_TEMPERATURE)

        sampled_draft: list[torch.Tensor] = []
        sampled_verifier: list[torch.Tensor] = []
        q_logp: list[torch.Tensor] = []

        prev_verifier = anchor_token_ids.long()  # [num_blocks], verifier vocab
        for k in range(1, block):
            step_logits = base_logits[:, k, :]  # [num_blocks, draft_vocab]
            if self.markov_head is not None:
                bias = self.markov_head.step_bias(
                    prev_token_ids=prev_verifier,
                    hidden_states=hidden_blocks[:, k, :],
                )
                step_logits = step_logits + bias

            scaled = step_logits / temp
            log_probs = torch.log_softmax(scaled, dim=-1)  # [num_blocks, draft_vocab]

            # Gumbel-max sampling from q_t = softmax(scaled). argmax is invariant
            # to the log-softmax constant, so we add Gumbel noise to `scaled`.
            noise = torch.rand(
                scaled.shape,
                dtype=scaled.dtype,
                device=scaled.device,
                generator=generator,
            ).clamp_(_EPS_ROLLOUT, 1.0)
            gumbel = -torch.log(-torch.log(noise))
            y_draft = torch.argmax(scaled.detach() + gumbel, dim=-1)  # [num_blocks]

            # log q_t(Y_t): grad flows through base_logits and the Markov bias.
            logp_t = log_probs.gather(-1, y_draft.unsqueeze(-1)).squeeze(-1)

            # Map draft-vocab sample -> verifier vocab for the next step's bias and
            # for the scorer. When draft vocab == verifier vocab the ids coincide.
            if self.use_draft_vocab and self.d2t is not None:
                y_verifier = self.d2t.to(y_draft.device)[y_draft]
            else:
                y_verifier = y_draft

            sampled_draft.append(y_draft)
            sampled_verifier.append(y_verifier)
            q_logp.append(logp_t)
            prev_verifier = y_verifier.long()

        return (
            torch.stack(sampled_draft, dim=1),
            torch.stack(q_logp, dim=1),
            torch.stack(sampled_verifier, dim=1),
        )

    def _onpolicy_forward(
        self,
        *,
        base_logits: torch.Tensor,  # [num_blocks, block, draft_vocab]
        hidden_blocks: torch.Tensor,  # [num_blocks, block, hidden]
        block_tokens: torch.Tensor,  # [num_blocks, block] verifier vocab
        targets: torch.Tensor,  # [1, mask_tokens_size, draft_vocab]
        aligned_loss_mask: torch.Tensor,  # [1, mask_tokens_size]
        anchor_positions: torch.Tensor,  # [num_blocks] index into the packed seq
        input_ids: torch.Tensor,  # [1, total_seq_len] verifier vocab
        document_ids: torch.Tensor,  # [1, total_seq_len]
        loss_config: LossConfig,
        confidence_head_alpha: float,
        temperature: float,
    ):
        """On-policy branch: sample a block, score it with the frozen verifier,
        and compute the exact expected-acceptance-length objective."""
        if self.verifier_scorer is None:
            raise RuntimeError(
                "onpolicy_sampling=True requires a verifier scorer. Inject one via "
                "`model.verifier_scorer = ...` before training (see "
                "speculators.models.dspark.onpolicy)."
            )

        # 1. Sample the block on-policy (draft side, cheap: 1 backbone forward
        #    already done, plus K Markov-head steps here).
        sampled_draft_ids, q_logp, sampled_verifier_ids = self._sample_block_rollout(
            base_logits=base_logits,
            hidden_blocks=hidden_blocks,
            anchor_token_ids=block_tokens[:, 0],
            temperature=temperature,
        )  # each [num_blocks, K], K = block_size - 1

        # 2. Score the sampled sequence with the frozen verifier (the only online,
        #    non-cacheable step). Returns verifier hidden states aligned so that
        #    row t predicts the token that fills draft slot t+1.
        with torch.no_grad():
            verifier_hidden = self.verifier_scorer.score(
                gold_input_ids=input_ids,
                document_ids=document_ids,
                anchor_positions=anchor_positions,
                sampled_verifier_ids=sampled_verifier_ids,
            )  # [num_blocks, K, hidden]

            # 3. p_t(Y_t): apply the frozen verifier head locally. verifier_lm_head
            #    is already sliced to the draft-vocab subset, so p and q share the
            #    same support (same convention as tv_loss / nla_loss).
            verifier_hidden = verifier_hidden.to(self.verifier_lm_head.weight.dtype)
            p_logits = self.verifier_lm_head(self.verifier_norm(verifier_hidden))
            temp = max(float(temperature), _MIN_SAMPLING_TEMPERATURE)
            p_log_probs = torch.log_softmax(p_logits / temp, dim=-1)
            p_logp = p_log_probs.gather(
                -1, sampled_draft_ids.unsqueeze(-1)
            ).squeeze(-1)  # [num_blocks, K]

        # 4. Gold-conditioned Markov-biased logits for the CE regulariser and the
        #    confidence head. The CE term keeps the same teacher-forced semantics
        #    as the non-on-policy path (gold prev tokens); only the position weight
        #    changes (C_t instead of the DFlash decay, applied inside
        #    compute_metrics). This is a second cheap Markov-head pass (no extra
        #    backbone forward).
        num_blocks, block, _ = base_logits.shape
        prev_token_ids = torch.cat(
            [block_tokens[:, :1], block_tokens[:, :-1]], dim=1
        )  # [num_blocks, block]
        prev_emb = None
        gold_logits = base_logits
        if self.markov_head is not None:
            prev_emb = self.markov_head.prev_embeddings(prev_token_ids)
            gold_bias = self.markov_head.block_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=hidden_blocks,
                prev_emb=prev_emb,
            )
            gold_logits = base_logits + gold_bias

        confidence_logits = None
        if self.confidence_head is not None:
            confidence_logits = self._confidence_logits(
                hidden_blocks, prev_emb, aligned_loss_mask.shape[-1]
            )

        # Draft-slot validity mask [num_blocks, K] (drop anchor slot 0).
        draft_mask = aligned_loss_mask.view(num_blocks, block)[:, 1:]

        loss, metrics = compute_metrics(
            gold_logits.view(1, num_blocks * block, -1),
            targets,
            confidence_logits,
            aligned_loss_mask,
            self.block_size,
            loss_config=loss_config,
            confidence_head_alpha=confidence_head_alpha,
            q_logp=q_logp,
            p_logp=p_logp,
            sampled_draft_ids=sampled_draft_ids,
            sampled_mask=draft_mask,
        )
        # Report the sampled tokens (draft vocab) laid back on the block grid.
        return sampled_draft_ids, loss, metrics
