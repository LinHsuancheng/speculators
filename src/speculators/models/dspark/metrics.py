"""Loss and metrics for the DSpark draft model.

loss = compound_loss(logits, targets) + conf_alpha * BCE(confidence, accept_rate)

The confidence target ``accept_rate = sum_v min(q_v, p_v) = 1 - d_TV`` is the
analytical acceptance rate (the overlap ``tv_loss`` already computes).

Optional CAT (Confidence-Adaptive Token) reweights each draft position by the
prefix product of a stop-gradient confidence proxy, following PARD-2:

* ``target``: target-model GT-token confidence
* ``draft``: analytical draft/target acceptance overlap
"""

from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import torch
from torch.nn.functional import binary_cross_entropy_with_logits, softmax

from speculators.models.metrics import (
    LossConfig,
    compound_loss,
    compute_accuracy_multi_step,
    dflash_loss_decay,
    draft_cat_weights,
    target_cat_weights,
)

__all__ = [
    "acceptance_length_credit",
    "compute_metrics",
    "exact_acceptance_length_loss",
]

_EPS = 1e-8

CatMode = Literal["none", "target", "draft"]


def acceptance_length_credit(
    q_logp: torch.Tensor,  # [num_blocks, K] log q_t(Y_t), sampled tokens
    p_logp: torch.Tensor,  # [num_blocks, K] log p_t(Y_t), same tokens, frozen
    mask: torch.Tensor | None = None,  # [num_blocks, K] 1 for valid draft slots
) -> torch.Tensor:
    """Exact per-position credit ``C_t`` for the expected-acceptance-length loss.

    Implements ``C_t = 1[q_t(Y_t) < p_t(Y_t)] * sum_{k>=t} S_k`` with
    ``S_k = prod_{i<=k} alpha_i`` and ``alpha_i = min(1, p_i / q_i)`` (trials.md
    §7-§9). Everything here is a function of *sampled* tokens under the frozen
    verifier, so the returned tensor is detached (stop-gradient): it is meant to
    be used as a per-position weight on ``log q_t(Y_t)``.

    Survival / continuation are computed per block (each block is one
    independent speculative rollout of ``K = block_size - 1`` draft positions;
    slot 0, the anchor, is excluded upstream). Masked (padded/invalid) slots are
    treated as ``alpha = 1`` so they neither shrink nor extend the survival of
    real slots; their own credit is zeroed at the end.

    Args:
        q_logp: Draft log-probability of the sampled token at each draft slot.
        p_logp: Frozen verifier log-probability of the *same* token at the
            *same* sampled prefix. Detached internally regardless.
        mask: Optional validity mask over draft slots. Masked slots contribute
            ``alpha = 1`` to the prefix product and receive credit ``0``.

    Returns:
        Detached credit ``C_t`` with shape ``[num_blocks, K]``.
    """
    if q_logp.shape != p_logp.shape:
        raise ValueError(
            f"Shape mismatch: q_logp={tuple(q_logp.shape)}, "
            f"p_logp={tuple(p_logp.shape)}"
        )

    q_detached = q_logp.detach()
    p_detached = p_logp.detach()

    # log(alpha_t) = min(0, log p_t - log q_t)
    log_alpha = torch.minimum(torch.zeros_like(q_detached), p_detached - q_detached)

    if mask is not None:
        # Masked slots act as alpha = 1 (log_alpha = 0) so they do not shrink the
        # prefix product of the real slots that follow within the block.
        log_alpha = log_alpha * mask.to(log_alpha.dtype)

    # S_k = prod_{i<=k} alpha_i   (per block, over draft slots)
    log_survival = torch.cumsum(log_alpha, dim=-1)
    survival = torch.exp(log_survival)

    # continuation_t = sum_{k>=t} S_k   (reverse cumsum along the block)
    continuation = torch.flip(
        torch.cumsum(torch.flip(survival, dims=[-1]), dim=-1),
        dims=[-1],
    )

    # I_t = 1[q_t(Y_t) < p_t(Y_t)]  (the under-covered / accept-with-prob-1 slots)
    undercovered = (q_detached < p_detached).to(q_logp.dtype)

    credit = undercovered * continuation
    if mask is not None:
        credit = credit * mask.to(credit.dtype)

    return credit.detach()


def exact_acceptance_length_loss(
    q_logp: torch.Tensor,  # [num_blocks, K] log q_t(Y_t), grad flows here
    p_logp: torch.Tensor,  # [num_blocks, K] log p_t(Y_t), frozen
    mask: torch.Tensor | None = None,  # [num_blocks, K]
) -> torch.Tensor:
    """On-policy Monte-Carlo loss for negative expected acceptance length.

    ``L_hat = -(1/K) * sum_t sg(C_t) * log q_t(Y_t)`` (trials.md §9, §13). Its
    gradient is an unbiased estimator of ``-grad R_K`` (negative expected
    acceptance length) under standard speculative sampling, provided ``Y`` is
    sampled on-policy from ``q_psi`` and ``p`` comes from the frozen verifier at
    the *sampled* prefix.

    Only ``q_logp`` carries gradient; the credit ``C_t`` is stop-gradient.
    Normalisation is by a fixed ``K`` (the number of draft slots per block), not
    by the random ``sum_t C_t`` — see trials.md §18.4.

    Args:
        q_logp: Draft log-probability of each sampled token (requires grad).
        p_logp: Frozen verifier log-probability of the same sampled tokens.
        mask: Optional validity mask over draft slots; masked slots are excluded
            from both the credit and the loss, and blocks are averaged over
            their valid-slot count.

    Returns:
        Scalar loss (mean over blocks).
    """
    credit = acceptance_length_credit(q_logp, p_logp, mask=mask)  # [num_blocks, K]

    weighted = credit * q_logp  # grad only through q_logp
    if mask is not None:
        weighted = weighted * mask.to(weighted.dtype)

    block_size_k = q_logp.shape[-1]
    # Divide by a fixed K (block draft length), never by the random credit sum.
    loss_per_block = -weighted.sum(dim=-1) / block_size_k  # [num_blocks]

    if mask is not None:
        # Average only over blocks that have at least one valid draft slot.
        block_valid = (mask.sum(dim=-1) > 0).to(loss_per_block.dtype)
        denom = block_valid.sum().clamp_min(1.0)
        return (loss_per_block * block_valid).sum() / denom

    return loss_per_block.mean()


def _masked_decayed_mean(
    elementwise: torch.Tensor,  # [1, T]
    loss_mask: torch.Tensor,  # [1, T]
    pos_idx: torch.Tensor,  # [1, T]
    decay_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    position_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked, optionally position-decayed mean of a precomputed per-position term."""
    loss_mask = loss_mask.to(elementwise.dtype)
    weighted = elementwise * loss_mask
    if decay_fn is not None:
        weighted = weighted * decay_fn(pos_idx.to(weighted.dtype))
    if position_weights is not None:
        weighted = weighted * position_weights.to(weighted.dtype)
    denominator = loss_mask.sum(dim=1) + _EPS
    return (weighted.sum(dim=1) / denominator).mean()


def _resolve_cat_weights(
    cat_mode: CatMode,
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor | None:
    if cat_mode == "none":
        return None
    if cat_mode == "target":
        return target_cat_weights(targets, block_size, loss_mask=loss_mask)
    if cat_mode == "draft":
        return draft_cat_weights(logits, targets, block_size, loss_mask=loss_mask)
    raise ValueError(
        f"Unknown cat_mode '{cat_mode}'. Choose from: none, target, draft."
    )


def compute_metrics(
    logits: torch.Tensor,  # [1, T, draft_vocab_size] (Markov-corrected)
    targets: torch.Tensor,  # [1, T, draft_vocab_size]
    confidence_logits: torch.Tensor | None,  # [1, T] or None
    loss_mask: torch.Tensor,  # [1, T]
    block_size: int,
    loss_config: LossConfig,
    gamma: float = 4.0,
    confidence_head_alpha: float = 1.0,
    cat_mode: CatMode = "none",
    q_logp: torch.Tensor | None = None,  # [num_blocks, K] on-policy log q_t(Y_t)
    p_logp: torch.Tensor | None = None,  # [num_blocks, K] frozen log p_t(Y_t)
    sampled_draft_ids: torch.Tensor | None = None,  # [num_blocks, K] draft vocab
    sampled_mask: torch.Tensor | None = None,  # [num_blocks, K] valid draft slots
) -> tuple[torch.Tensor, dict]:
    """Compute the DSpark loss and a metrics dict (``*_sum``/``*_total`` pairs).

    Two modes:

    * **Teacher-forced** (default, ``q_logp is None``): the compound loss over
      gold positions with DFlash exponential position decay, unchanged.
    * **On-policy** (``q_logp`` provided): the exact expected-acceptance-length
      objective. The per-position credit ``C_t`` (trials.md §7-§9) *replaces* the
      DFlash decay as the position weight shared by the CE term and the confidence
      BCE, and the ``accept_length`` term ``-(1/K) sum_t C_t log q_t(Y_t)`` is
      added with the weight given for ``"accept_length"`` in ``loss_config``.
    """
    onpolicy = q_logp is not None
    if onpolicy and (p_logp is None or sampled_mask is None):
        raise ValueError(
            "On-policy compute_metrics requires q_logp, p_logp and sampled_mask."
        )

    device = logits.device
    seq_len = logits.shape[1]
    num_blocks = seq_len // block_size
    pos_idx = (torch.arange(seq_len, device=device) % block_size).unsqueeze(0)

    # Position weighting: decay (teacher-forced) vs. credit C_t (on-policy).
    if onpolicy:
        credit = acceptance_length_credit(q_logp, p_logp, mask=sampled_mask)
        # Scatter C_t [num_blocks, K] onto the gold grid [1, T]; anchor slot 0 -> 0.
        credit_grid = torch.zeros(
            num_blocks, block_size, device=device, dtype=logits.dtype
        )
        credit_grid[:, 1:] = credit.to(logits.dtype)
        position_weights = credit_grid.view(1, seq_len)
        decay_fn = None
        cat_weights = None
    else:
        decay_fn = partial(dflash_loss_decay, gamma=gamma)
        cat_weights = _resolve_cat_weights(
            cat_mode, logits, targets, loss_mask, block_size
        )
        position_weights = cat_weights

    # The ``accept_length`` term is not a (logits, targets) loss, so pull it out of
    # the config before compound_loss and add it explicitly below.
    accept_length_weight: float | None = None
    compound_config = loss_config
    if "accept_length" in loss_config:
        if not onpolicy:
            raise ValueError(
                "'accept_length' loss requires on-policy sampling; call the model "
                "with onpolicy_sampling=True (q_logp/p_logp)."
            )
        accept_length_weight = loss_config["accept_length"][1]
        compound_config = {
            k: v for k, v in loss_config.items() if k != "accept_length"
        }

    loss, term_losses = compound_loss(
        logits,
        targets,
        loss_mask,
        pos_idx,
        loss_config=compound_config,
        decay_fn=decay_fn,
        position_weights=position_weights,
    )

    if accept_length_weight is not None:
        accept_loss = exact_acceptance_length_loss(
            q_logp, p_logp, mask=sampled_mask
        )
        loss = loss + accept_length_weight * accept_loss
        term_losses["accept_length_loss"] = accept_loss.detach()

    # Analytical per-position acceptance rate = distributional overlap.
    with torch.no_grad():
        draft_p = softmax(logits.float(), dim=-1)
        target_p = softmax(targets.float(), dim=-1)
        accept_rate = torch.minimum(draft_p, target_p).sum(dim=-1)  # [1, T]
        # Per-block cumulative acceptance product over the draft slots (slot 0
        # is the anchor), shared by the accept-length and calibration metrics.
        num_blocks = seq_len // block_size
        accept_blocks = accept_rate.view(num_blocks, block_size)
        draft_mask = loss_mask.to(accept_rate.dtype).view(num_blocks, block_size)[:, 1:]
        accept_prefix = (accept_blocks[:, 1:] * draft_mask).cumprod(dim=-1)

    metrics: dict[str, Any] = {}
    if confidence_logits is not None:
        c_star = accept_rate.detach().to(confidence_logits.dtype)
        bce = binary_cross_entropy_with_logits(
            confidence_logits, c_star, reduction="none"
        )  # [1, T]
        # Confidence BCE uses the same position weighting as the other terms:
        # C_t on-policy (decay_fn is None there), cat_weights + decay teacher-forced.
        conf_loss = _masked_decayed_mean(
            bce, loss_mask, pos_idx, decay_fn, position_weights=position_weights
        )
        loss = loss + confidence_head_alpha * conf_loss

        with torch.no_grad():
            mask_f = loss_mask.to(accept_rate.dtype)
            mask_total = mask_f.sum().clamp_min(1.0)
            conf_prob = confidence_logits.float().sigmoid()
            metrics["confidence_loss_sum"] = conf_loss.detach().clone()
            metrics["confidence_loss_total"] = torch.ones((), device=device)
            metrics["confidence_abs_error_sum"] = (
                (conf_prob - accept_rate).abs() * mask_f
            ).sum()
            metrics["confidence_abs_error_total"] = mask_total
            # Mean predicted vs. observed acceptance — a calibration sanity check.
            metrics["confidence_pred_mean_sum"] = (conf_prob * mask_f).sum()
            metrics["confidence_pred_mean_total"] = mask_total
            # Calibration of the cumulative acceptance product, which is what
            # dynamic draft-length thresholding consumes (signed pred - target).
            conf_prefix = (
                conf_prob.view(num_blocks, block_size)[:, 1:] * draft_mask
            ).cumprod(dim=-1)
            metrics["confidence_cumprod_bias_sum"] = (
                (conf_prefix - accept_prefix) * draft_mask
            ).sum()
            metrics["confidence_cumprod_bias_total"] = draft_mask.sum().clamp_min(1.0)

    ones = torch.ones((), device=device)
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = ones
    for term_name, term_val in term_losses.items():
        metrics[f"{term_name}_sum"] = term_val
        metrics[f"{term_name}_total"] = ones

    if cat_weights is not None:
        with torch.no_grad():
            mask_f = loss_mask.to(cat_weights.dtype)
            metrics["cat_weight_mean_sum"] = (cat_weights * mask_f).sum()
            metrics["cat_weight_mean_total"] = mask_f.sum().clamp_min(1.0)

    # Mean acceptance rate of the (Markov-corrected) drafter.
    with torch.no_grad():
        mask_f = loss_mask.to(accept_rate.dtype)
        metrics["accept_rate_sum"] = (accept_rate * mask_f).sum()
        metrics["accept_rate_total"] = mask_f.sum().clamp_min(1.0)

    # Expected accepted draft length per block (DSpark's tau): the cumulative
    # acceptance product summed over draft slots, plus the always-emitted anchor.
    with torch.no_grad():
        per_block_len = accept_prefix.sum(dim=-1) + 1.0
        block_valid = (draft_mask.sum(dim=-1) > 0).to(accept_rate.dtype)
        metrics["accept_len_sum"] = (per_block_len * block_valid).sum()
        metrics["accept_len_total"] = block_valid.sum().clamp_min(1.0)

    # Per-position greedy accuracy (position 0 is the anchor — excluded).
    pred_ids = torch.argmax(logits, dim=-1)
    target_ids = torch.argmax(targets, dim=-1)
    correct_per_pos, total_per_pos = compute_accuracy_multi_step(
        pred_ids, target_ids, loss_mask, pos_idx, block_size
    )
    metrics["full_acc_sum"] = correct_per_pos[1:].sum()
    metrics["full_acc_total"] = total_per_pos[1:].sum()
    for pos in range(1, block_size):
        metrics[f"position_{pos}_acc_sum"] = correct_per_pos[pos]
        metrics[f"position_{pos}_acc_total"] = total_per_pos[pos]

    return loss, metrics
