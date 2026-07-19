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
    tf_eal_loss,
)

__all__ = [
    "compute_metrics",
]

_EPS = 1e-8

CatMode = Literal["none", "target", "draft"]


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
    tf_eal_alpha: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Compute the DSpark loss and a metrics dict (``*_sum``/``*_total`` pairs)."""

    device = logits.device
    seq_len = logits.shape[1]
    pos_idx = (torch.arange(seq_len, device=device) % block_size).unsqueeze(0)
    decay_fn = partial(dflash_loss_decay, gamma=gamma)
    cat_weights = _resolve_cat_weights(
        cat_mode, logits, targets, loss_mask, block_size
    )

    loss, term_losses = compound_loss(
        logits,
        targets,
        loss_mask,
        pos_idx,
        loss_config=loss_config,
        decay_fn=decay_fn,
        position_weights=cat_weights,
    )

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

    # Teacher-forced sequence-level expected-acceptance-length (TF-EAL) loss.
    # L = 1 - R_TF/K where R_TF = Σ_k Π_{i≤k} a_i, normalized to [0, 1] so it
    # sits at the same magnitude as tv_loss. No position decay: the survival
    # product already gates later positions (see tf_eal_loss docstring).
    if tf_eal_alpha > 0.0:
        tf_loss, tf_aux = tf_eal_loss(logits, targets, loss_mask, block_size)
        loss = loss + tf_eal_alpha * tf_loss

        with torch.no_grad():
            metrics["tf_eal_loss_sum"] = tf_loss.detach().clone()
            metrics["tf_eal_loss_total"] = torch.ones((), device=device)
            # Teacher-forced expected accept length tau_TF = R_TF + 1 (anchor).
            block_valid = tf_aux["block_valid"]
            valid_total = block_valid.sum().clamp_min(1.0)
            metrics["tf_eal_tau_sum"] = (tf_aux["tau"] * block_valid).sum()
            metrics["tf_eal_tau_total"] = valid_total
            # Per draft-position survival S_k and continuation credit C_t=Σ_{k≥t}S_k,
            # so each step's contribution is visible in the logs (pos 1..K).
            surv = tf_aux["survival"]  # [num_blocks, K]
            cont = tf_aux["continuation"]  # [num_blocks, K]
            dmask = tf_aux["draft_mask"]  # [num_blocks, K]
            per_pos_total = dmask.sum(dim=0).clamp_min(1.0)  # [K]
            surv_pos = (surv * dmask).sum(dim=0)  # [K]
            cont_pos = (cont * dmask).sum(dim=0)  # [K]
            for k in range(surv.shape[1]):
                pos = k + 1  # draft slot 0 is the anchor
                metrics[f"tf_eal_survival_pos_{pos}_sum"] = surv_pos[k]
                metrics[f"tf_eal_survival_pos_{pos}_total"] = per_pos_total[k]
                metrics[f"tf_eal_credit_pos_{pos}_sum"] = cont_pos[k]
                metrics[f"tf_eal_credit_pos_{pos}_total"] = per_pos_total[k]

    if confidence_logits is not None:
        c_star = accept_rate.detach().to(confidence_logits.dtype)
        bce = binary_cross_entropy_with_logits(
            confidence_logits, c_star, reduction="none"
        )  # [1, T]
        conf_loss = _masked_decayed_mean(
            bce, loss_mask, pos_idx, decay_fn, position_weights=cat_weights
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
