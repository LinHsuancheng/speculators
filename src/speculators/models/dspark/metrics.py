"""Loss and metrics for the DSpark draft model.

loss = compound_loss(logits, targets) + conf_alpha * BCE(confidence, accept_rate)

The confidence target ``accept_rate = sum_v min(q_v, p_v) = 1 - d_TV`` is the
analytical acceptance rate (the overlap ``tv_loss`` already computes).

Optional CAT (Confidence-Adaptive Token) reweights each draft position by the
prefix product of a stop-gradient confidence proxy, following PARD-2:

* ``target``: target-model GT-token confidence
* ``draft``: analytical draft/target acceptance overlap

When on-policy sampled target logprobs are provided, DSpark can replace the
position/gamma decay with exact speculative-sampling credit weights and add the
Monte Carlo acceptance-length loss from the sampled path.
"""

from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import torch
from torch.nn.functional import binary_cross_entropy_with_logits, softmax

from speculators.models.metrics import (
    ce_loss,
    LossConfig,
    compound_loss,
    compute_accuracy_multi_step,
    dflash_loss_decay,
    draft_cat_weights,
    loss_function,
    target_cat_weights,
)

__all__ = [
    "compute_metrics",
    "exact_acceptance_length_loss",
    "sampled_acceptance_credit",
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


def sampled_acceptance_credit(
    draft_logp: torch.Tensor,
    target_logp: torch.Tensor,
) -> torch.Tensor:
    """Return stop-gradient exact speculative credit for an on-policy path.

    ``draft_logp`` is ``log q_t(Y_t)`` and must keep gradient. ``target_logp`` is
    ``log p_t(Y_t)`` from the frozen verifier.  The returned credit is

    ``1[q_t(Y_t) < p_t(Y_t)] * sum_{k=t..K} prod_{i<=k} alpha_i``

    with ``alpha_i = min(1, p_i(Y_i) / q_i(Y_i))``.
    """
    if draft_logp.shape != target_logp.shape:
        raise ValueError(
            "sampled logprob shape mismatch: "
            f"draft_logp={draft_logp.shape}, target_logp={target_logp.shape}"
        )
    if draft_logp.ndim < 1:
        raise ValueError("sampled logprobs must have at least one dimension")

    q_detached = draft_logp.detach()
    p_detached = target_logp.detach().to(device=q_detached.device)
    log_alpha = torch.minimum(
        torch.zeros_like(q_detached),
        p_detached - q_detached,
    )
    survival = torch.exp(torch.cumsum(log_alpha, dim=-1))
    continuation = torch.flip(
        torch.cumsum(torch.flip(survival, dims=[-1]), dim=-1),
        dims=[-1],
    )
    undercovered = (q_detached < p_detached).to(dtype=draft_logp.dtype)
    return (undercovered * continuation.to(dtype=draft_logp.dtype)).detach()


def exact_acceptance_length_loss(
    draft_logp: torch.Tensor,
    target_logp: torch.Tensor,
    credit: torch.Tensor | None = None,
) -> torch.Tensor:
    """Monte Carlo loss whose gradient optimizes expected accepted length."""
    if credit is None:
        credit = sampled_acceptance_credit(draft_logp, target_logp)
    if credit.shape != draft_logp.shape:
        raise ValueError(
            f"credit shape mismatch: credit={credit.shape}, draft_logp={draft_logp.shape}"
        )
    block_tokens = draft_logp.shape[-1]
    return -((credit * draft_logp).sum(dim=-1) / block_tokens).mean()


def _sampled_credit_position_weights(
    credit: torch.Tensor,
    loss_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Map sampled draft-slot credit to ``[1, T]`` position weights.

    The first position in every block is the anchor and remains zero.  Credit is
    placed on slots ``1..K``.
    """
    total_seq_len = loss_mask.shape[1]
    if total_seq_len % block_size != 0:
        raise ValueError(
            f"loss_mask length {total_seq_len} is not divisible by block_size={block_size}"
        )
    num_blocks = total_seq_len // block_size
    if credit.ndim == 3:
        if credit.shape[0] != 1:
            raise ValueError(
                "sampled credit currently expects batch size 1, "
                f"got credit shape {credit.shape}"
            )
        credit = credit.squeeze(0)
    if credit.ndim == 1:
        credit = credit.unsqueeze(0)
    if credit.ndim != 2:
        raise ValueError(
            "sampled credit must have shape [K], [num_blocks, K], or "
            f"[1, num_blocks, K], got {credit.shape}"
        )
    if credit.shape[0] != num_blocks:
        raise ValueError(
            f"sampled credit has {credit.shape[0]} blocks but loss has {num_blocks}"
        )
    sampled_slots = credit.shape[1]
    if sampled_slots > block_size - 1:
        raise ValueError(
            f"sampled credit has {sampled_slots} slots, but block_size={block_size} "
            f"allows at most {block_size - 1}"
        )

    weights = torch.zeros(
        num_blocks,
        block_size,
        device=loss_mask.device,
        dtype=credit.dtype,
    )
    weights[:, 1 : 1 + sampled_slots] = credit.to(loss_mask.device)
    return weights.reshape(1, total_seq_len) * loss_mask.to(dtype=credit.dtype)


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
    sampled_draft_logprobs: torch.Tensor | None = None,
    sampled_target_logprobs: torch.Tensor | None = None,
    sampled_acceptance_loss_alpha: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Compute the DSpark loss and a metrics dict (``*_sum``/``*_total`` pairs)."""

    device = logits.device
    seq_len = logits.shape[1]
    pos_idx = (torch.arange(seq_len, device=device) % block_size).unsqueeze(0)
    sampled_credit = None
    sampled_exact_loss = None
    if sampled_draft_logprobs is not None or sampled_target_logprobs is not None:
        if sampled_draft_logprobs is None or sampled_target_logprobs is None:
            raise ValueError(
                "sampled_draft_logprobs and sampled_target_logprobs must be provided "
                "together"
            )
        sampled_draft_logprobs = sampled_draft_logprobs.to(device=device)
        sampled_target_logprobs = sampled_target_logprobs.to(device=device)
        sampled_credit = sampled_acceptance_credit(
            sampled_draft_logprobs, sampled_target_logprobs
        )
        sampled_exact_loss = exact_acceptance_length_loss(
            sampled_draft_logprobs,
            sampled_target_logprobs,
            credit=sampled_credit,
        )
        position_weights = _sampled_credit_position_weights(
            sampled_credit, loss_mask, block_size
        )
        decay_fn = None
        cat_weights = None
        loss = loss_function(
            logits,
            targets,
            loss_mask,
            pos_idx,
            loss_fn=ce_loss,
            decay_fn=None,
            position_weights=position_weights,
        )
        term_losses = {"ce_loss": loss.detach()}
    else:
        decay_fn = partial(dflash_loss_decay, gamma=gamma)
        cat_weights = _resolve_cat_weights(
            cat_mode, logits, targets, loss_mask, block_size
        )
        position_weights = cat_weights
        loss, term_losses = compound_loss(
            logits,
            targets,
            loss_mask,
            pos_idx,
            loss_config=loss_config,
            decay_fn=decay_fn,
            position_weights=position_weights,
        )
    if sampled_exact_loss is not None:
        loss = loss + sampled_acceptance_loss_alpha * sampled_exact_loss

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

    if sampled_exact_loss is not None and sampled_credit is not None:
        metrics["sampled_acceptance_loss_sum"] = sampled_exact_loss.detach().clone()
        metrics["sampled_acceptance_loss_total"] = ones
        with torch.no_grad():
            metrics["sampled_credit_mean_sum"] = sampled_credit.detach().sum()
            metrics["sampled_credit_mean_total"] = torch.tensor(
                sampled_credit.numel(), device=device, dtype=sampled_credit.dtype
            )
            log_alpha = torch.minimum(
                torch.zeros_like(sampled_draft_logprobs.detach()),
                sampled_target_logprobs.detach() - sampled_draft_logprobs.detach(),
            )
            sampled_alpha = torch.exp(log_alpha)
            metrics["sampled_alpha_mean_sum"] = sampled_alpha.sum()
            metrics["sampled_alpha_mean_total"] = torch.tensor(
                sampled_alpha.numel(), device=device, dtype=sampled_alpha.dtype
            )

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
