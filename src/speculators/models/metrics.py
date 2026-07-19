import json
from collections.abc import Callable

import torch

_EPS = 1e-5

LossConfig = dict[
    str, tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], float]
]


def compute_accuracy_single_step(
    pred_ids: torch.Tensor,  # shape: [1, seq_len]
    target_ids: torch.Tensor,  # shape: [1, seq_len]
    loss_mask: torch.Tensor | None,  # shape: [1, seq_len]
    prev_correct: torch.Tensor | None,  # shape: [1, seq_len]
):
    """Compute full and conditional accuracy counts for a single speculative step.

    Args:
        pred_ids: Predicted token IDs.
        target_ids: Ground-truth token IDs.
        loss_mask: If provided, restricts accuracy to masked positions.
        prev_correct: Boolean mask of positions correct so far. Updated in place
            via logical AND with the current step's correctness.

    Returns:
        Tuple of (full_correct, full_total, cond_correct, cond_total) as raw
        counts suitable for distributed reduction before computing ratios.
    """
    correct = pred_ids == target_ids
    cond_total = torch.tensor(correct.numel(), dtype=torch.float, device=correct.device)
    if prev_correct is not None:
        cond_total = prev_correct.sum().float()
        correct = torch.logical_and(prev_correct, correct, out=prev_correct)
    if loss_mask is not None:
        correct = torch.masked_select(correct, loss_mask.to(torch.bool))

    correct_sum = correct.float().sum()
    full_total = torch.tensor(correct.numel(), dtype=torch.float, device=correct.device)

    return correct_sum, full_total, correct_sum, cond_total


@torch.no_grad()
def compute_accuracy_multi_step(
    pred_ids: torch.Tensor,  # shape: [1, seq_len]
    target_ids: torch.Tensor,  # shape: [1, seq_len]
    loss_mask: torch.Tensor,  # shape: [1, seq_len]
    pos_idx: torch.Tensor,  # shape: [1, seq_len]
    num_pos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-position correct/total counts across multiple speculative steps.

    Args:
        pred_ids: Predicted token IDs.
        target_ids: Ground-truth token IDs.
        loss_mask: Boolean mask selecting positions to evaluate.
        pos_idx: Position index within each speculative block (e.g. 0,1,2,3,0,1,2,3).
        num_pos: Number of distinct positions (i.e. block size).

    Returns:
        Tuple of (correct_per_pos, total_per_pos) both with shape [num_pos].
        Overall counts can be derived by summing these.
    """
    correct = pred_ids == target_ids
    correct = torch.masked_select(correct, loss_mask.to(torch.bool))
    pos_idx = torch.masked_select(pos_idx, loss_mask.to(torch.bool))

    correct_per_pos = torch.zeros(num_pos, dtype=torch.float, device=correct.device)
    total_per_pos = torch.zeros(num_pos, dtype=torch.float, device=correct.device)
    correct_per_pos.scatter_add_(0, pos_idx, correct.float())
    total_per_pos.scatter_add_(0, pos_idx, torch.ones_like(correct, dtype=torch.float))

    return correct_per_pos, total_per_pos  # shape: [num_pos], [num_pos]


def kl_div_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Compute per-position KL divergence from draft logits to target logits.

    Args:
        logits: Draft model logits (log-softmax applied internally).
        targets: Target model logits (softmax applied internally).

    Returns:
        Per-position KL divergence with shape [1, seq_len].
    """
    logits = torch.nn.functional.log_softmax(logits, dim=-1)
    target_p = torch.nn.functional.softmax(targets, dim=-1)
    elementwise_loss = torch.nn.functional.kl_div(
        logits, target_p, reduction="none", log_target=False
    ).sum(dim=-1)  # shape: [1, seq_len]

    return elementwise_loss  # noqa: RET504


def ce_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Compute per-position cross-entropy loss using argmax of target logits as labels.

    Args:
        logits: Draft model logits.
        targets: Target model logits (argmax taken to produce hard labels).

    Returns:
        Per-position cross-entropy loss with shape [1, seq_len].
    """
    batch_size, seq_len, draft_vocab_size = logits.shape
    target_ids = torch.argmax(targets, dim=-1)  # shape: [1, seq_len]

    elementwise_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, draft_vocab_size),
        target_ids.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(batch_size, seq_len)

    return elementwise_loss  # noqa: RET504


def tv_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Compute per-position total variation (TV) distance from draft to target.

    The rejection-sampling acceptance rate of speculative decoding equals the
    distributional overlap between target and draft,
    ``alpha = sum_v min(p_v, q_v) = 1 - d_TV(p, q)``. Minimizing this TV distance
    therefore directly optimizes the acceptance rate, whereas cross-entropy and
    KL only optimize it indirectly (KL is a loose upper bound on TV via Pinsker).

    Args:
        logits: Draft model logits (softmax applied internally to form q).
        targets: Target model logits (softmax applied internally to form p).

    Returns:
        Per-position TV distance with shape [1, seq_len].
    """
    draft_p = torch.nn.functional.softmax(logits, dim=-1)
    target_p = torch.nn.functional.softmax(targets, dim=-1)
    overlap = torch.minimum(draft_p, target_p).sum(dim=-1)  # shape: [1, seq_len]
    elementwise_loss = 1.0 - overlap

    return elementwise_loss  # noqa: RET504


def tf_eal_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    loss_mask: torch.Tensor,  # shape: [1, seq_len]
    block_size: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Teacher-forced sequence-level expected-acceptance-length (TF-EAL) loss.

    On the teacher-forced (target/data) path, the per-position acceptance
    probability is the distributional overlap ``a_i = Σ_v min(q_i(v), p_i(v))
    = 1 - d_TV(p_i, q_i)`` (the same quantity as ``tv_loss``). Chaining these
    over a speculative block gives the cumulative survival ``S_k = Π_{i=1}^k a_i``
    and the expected accepted length surrogate ``R_TF = Σ_{k=1}^K S_k``. We
    maximize ``R_TF`` directly. To keep the loss on the same ``[0, 1]`` scale as
    ``tv_loss`` (so it composes with TV/CE without dwarfing them), we normalize
    by the per-block draft-slot count and minimize ``L = 1 - R_TF / K``. Both
    endpoints mirror TV: ``R_TF = K`` (perfect acceptance) → ``L = 0``, and
    ``R_TF = 0`` → ``L = 1``.

    Unlike the on-policy sampled exact loss, this needs no draft-sampled tokens
    and no per-position sampled-token indicator: it is computed directly on the
    existing target logits. It is a teacher-forced *approximation* of the true
    on-policy expected acceptance length, tight when ``p ≈ q`` on the relevant
    prefixes (error ``O(K^2 ε)`` for per-step ``TV ≤ ε``).

    The survival product ``S_k`` supplies a natural position weighting (early
    positions gate all later ones), so this loss deliberately carries **no**
    exponential position decay; the ``∂R_TF/∂a_t = S_{t-1} Σ_{k≥t} S_k / a_t``
    continuation credit is intrinsic.

    Anchor slot 0 (always emitted) is excluded from the draft slots. Masked
    slots act as acceptance ``1`` inside the prefix product so padding does not
    shrink later survivals, and are zeroed out of the ``R_TF`` sum.

    Args:
        logits: Draft model logits (softmax applied internally to form q).
        targets: Target model logits (softmax applied internally to form p).
        loss_mask: Boolean mask selecting valid positions.
        block_size: Speculative block size (including the anchor slot).

    Returns:
        Tuple of ``(loss, aux)``:
        * ``loss``: scalar ``1 - R_TF / K`` averaged over valid blocks, in
          ``[0, 1]`` (same scale as ``tv_loss``).
        * ``aux``: dict of detached logging tensors — ``survival`` and
          ``continuation`` (per-position credit ``Σ_{k≥t} S_k``), both shaped
          ``[num_blocks, block_size - 1]``, plus per-block ``tau`` (``R_TF + 1``)
          and ``block_valid`` masks.
    """
    seq_len = logits.shape[1]
    if seq_len % block_size != 0:
        raise ValueError(
            f"Sequence length {seq_len} is not divisible by block_size {block_size}."
        )
    num_blocks = seq_len // block_size

    draft_p = torch.nn.functional.softmax(logits.float(), dim=-1)
    target_p = torch.nn.functional.softmax(targets.float(), dim=-1)
    accept = torch.minimum(draft_p, target_p).sum(dim=-1)  # a_i, [1, seq_len]

    accept_blocks = accept.view(num_blocks, block_size)
    mask_blocks = loss_mask.to(accept.dtype).view(num_blocks, block_size)
    # Draft slots only (slot 0 is the always-emitted anchor).
    draft_a = accept_blocks[:, 1:]  # [num_blocks, K], K = block_size - 1
    draft_mask = mask_blocks[:, 1:]

    # Masked slots act as acceptance 1 so they do not shrink later survivals.
    a_for_prod = draft_a * draft_mask + (1.0 - draft_mask)
    survival = torch.cumprod(a_for_prod, dim=-1)  # S_k, [num_blocks, K]
    # Only valid draft slots contribute to R_TF = Σ_k S_k.
    survival_masked = survival * draft_mask
    r_tf = survival_masked.sum(dim=-1)  # [num_blocks]

    # Normalize R_TF by the per-block draft-slot count into [0, 1] and map to a
    # minimization loss ``1 - R_TF/K_b`` — the same [0, 1] scale and "1 - accept"
    # form as tv_loss, so it composes with TV/CE at a comparable magnitude. The
    # 1/K_b factor is a fixed constant (not a random per-sample denominator), so
    # this only rescales the gradient without introducing bias (trials.md §18.4).
    slots_per_block = draft_mask.sum(dim=-1)  # K_b, [num_blocks]
    block_valid = (slots_per_block > 0).to(accept.dtype)  # [num_blocks]
    r_tf_norm = r_tf / slots_per_block.clamp_min(1.0)  # R_TF / K_b in [0, 1]
    per_block_loss = 1.0 - r_tf_norm  # [num_blocks], in [0, 1]
    denom = block_valid.sum().clamp_min(1.0)
    loss = (per_block_loss * block_valid).sum() / denom

    with torch.no_grad():
        # Continuation credit C_t = Σ_{k≥t} S_k (reverse cumsum along draft slots).
        continuation = torch.flip(
            torch.cumsum(torch.flip(survival_masked, dims=[-1]), dim=-1), dims=[-1]
        )
        aux = {
            "survival": survival_masked.detach(),  # [num_blocks, K]
            "continuation": continuation.detach(),  # [num_blocks, K]
            "draft_mask": draft_mask.detach(),  # [num_blocks, K]
            "tau": (r_tf + 1.0).detach(),  # [num_blocks]
            "block_valid": block_valid.detach(),  # [num_blocks]
        }
    return loss, aux


def neg_log_acceptance_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Compute per-position negative log-acceptance (LK) loss.

    The speculative-decoding acceptance rate equals the draft/target distribution
    overlap, ``alpha = sum_v min(p_v, q_v)`` (the same quantity computed in
    ``tv_loss``). This loss is ``-log(alpha)``. Its gradient is
    ``(1 / alpha) * grad(TV)``: the ``1 / alpha`` factor amplifies the otherwise
    vanishing TV gradient when overlap is low (early training), giving TV's
    acceptance-optimal target a usable gradient from a cold start. When the target
    is a point mass, this loss reduces to cross-entropy.

    Args:
        logits: Draft model logits (softmax applied internally to form q).
        targets: Target model logits (softmax applied internally to form p).

    Returns:
        Per-position negative log-acceptance with shape [1, seq_len].
    """
    draft_p = torch.nn.functional.softmax(logits, dim=-1)
    target_p = torch.nn.functional.softmax(targets, dim=-1)
    overlap = torch.minimum(draft_p, target_p).sum(dim=-1)  # alpha, shape: [1, seq_len]
    elementwise_loss = -torch.log(overlap.clamp_min(_EPS))

    return elementwise_loss  # noqa: RET504


def lk_hybrid_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    eta: float = 3.0,
):
    """Compute per-position hybrid LK loss (adaptive KL/TV blend).

    Blends KL divergence and total variation per position:
    ``L = lambda * KL(p||q) + (1 - lambda) * TV(p, q)`` with adaptive weight
    ``lambda = exp(-eta * sg[alpha])``, where ``alpha = sum_v min(p_v, q_v)`` is the
    acceptance rate (overlap) and ``sg`` is stop-gradient. When overlap is low
    (early training, misaligned draft) ``lambda -> 1`` and the loss leans on KL's
    strong gradient; as overlap grows ``lambda -> 0`` and it shifts to TV, which
    optimizes acceptance directly. This gives TV's acceptance-optimal target a
    usable gradient from a cold start.

    ``alpha`` in the weight is detached: it controls the blend but is not
    differentiated through; gradients flow only through the KL and TV terms.

    Source: Samarin et al., "LK Losses: Direct Acceptance Rate Optimization for
    Speculative Decoding" (arXiv 2602.23881), hybrid objective.

    Args:
        logits: Draft model logits (softmax applied internally to form q).
        targets: Target model logits (softmax applied internally to form p).
        eta: Blend temperature; larger shifts toward TV sooner. Default 3.0
            (the paper's best hybrid setting).

    Returns:
        Per-position hybrid loss with shape [1, seq_len].
    """
    draft_p = torch.nn.functional.softmax(logits, dim=-1)
    target_p = torch.nn.functional.softmax(targets, dim=-1)
    overlap = torch.minimum(draft_p, target_p).sum(dim=-1)  # alpha, shape: [1, seq_len]
    tv = 1.0 - overlap
    kl = kl_div_loss(logits, targets)  # reuse existing KL, shape: [1, seq_len]
    weight = torch.exp(-eta * overlap.detach())  # lambda = exp(-eta * sg[alpha])
    elementwise_loss = weight * kl + (1.0 - weight) * tv

    return elementwise_loss  # noqa: RET504


def dflash_loss_decay(pos_idx: torch.Tensor, gamma: float):
    """Compute DFlash-style exponential decay weights per position.

    Position 0 gets weight 0, position 1 gets weight 1, and subsequent positions
    decay as exp(-(pos - 1) / gamma).

    Args:
        pos_idx: Position indices within each speculative block.
        gamma: Decay rate (higher = slower decay).

    Returns:
        Decay multiplier tensor with same shape as pos_idx.
    """
    # pos_idx = 0 1 2 3 0 1 2 3, block_size = 4
    decay_mult = torch.exp(-((pos_idx - 1).clamp(min=0)) / gamma)
    # decay_mult = e^-(0 0 1 2 0 0 1 2) / gamma
    decay_mult = decay_mult * (pos_idx != 0).to(decay_mult.dtype)
    # w = 0 1 e^-1/gamma e^-2/gamma 0 1 e^-1/gamma e^-2/gamma
    return decay_mult  # noqa: RET504


def exp_loss_decay(pos_idx: torch.Tensor, gamma: float):
    """Compute simple exponential decay weights as gamma^pos_idx.

    Args:
        pos_idx: Position indices within each speculative block.
        gamma: Base of the exponent (typically in (0, 1]).

    Returns:
        Decay multiplier tensor with same shape as pos_idx.
    """
    return gamma**pos_idx


def cat_prefix_weights(
    per_pos_conf: torch.Tensor,  # [1, T]
    block_size: int,
    loss_mask: torch.Tensor | None = None,  # [1, T]
) -> torch.Tensor:
    """Build PARD-2-style CAT prefix weights from per-position confidences.

    For each speculative block, draft position ``k`` (1-indexed within the block)
    receives weight ``ŝ_k = ∏_{j<k} c_j`` with ``ŝ_1 = 1``. Anchor slot 0 is 0.
    Masked positions contribute ``1`` to the product so padding does not shrink
    later weights. The returned tensor is detached (stop-gradient).

    Args:
        per_pos_conf: Per-position confidence / acceptance proxy in ``[0, 1]``.
        block_size: Speculative block size (including the anchor slot).
        loss_mask: Optional mask; masked slots act as confidence ``1`` in the
            prefix product.

    Returns:
        CAT weights with the same shape as ``per_pos_conf``.
    """
    conf = per_pos_conf.detach().float()
    seq_len = conf.shape[-1]
    if seq_len % block_size != 0:
        raise ValueError(
            f"Sequence length {seq_len} is not divisible by block_size {block_size}."
        )
    num_blocks = seq_len // block_size
    conf_blocks = conf.view(num_blocks, block_size)
    if loss_mask is not None:
        mask = loss_mask.to(conf.dtype).view(num_blocks, block_size)
        # Masked / invalid slots should not shrink the prefix product.
        conf_blocks = conf_blocks * mask + (1.0 - mask)

    draft_conf = conf_blocks[:, 1:].clamp_min(_EPS)
    ones = torch.ones(num_blocks, 1, device=conf.device, dtype=conf.dtype)
    # weight for draft slot i uses product of confidences at draft slots < i.
    prev_prod = torch.cumprod(torch.cat([ones, draft_conf[:, :-1]], dim=-1), dim=-1)
    weights = torch.zeros_like(conf_blocks)
    weights[:, 1:] = prev_prod
    return weights.view_as(per_pos_conf).to(per_pos_conf.dtype)


def target_cat_weights(
    targets: torch.Tensor,  # [1, T, V]
    block_size: int,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """PARD-2 CAT weights from target-model confidence of the ground-truth token.

    ``ĉ_k = softmax(targets)_k[argmax(targets)_k]``, then prefix-cumprod within
    each block as in :func:`cat_prefix_weights`.
    """
    target_p = torch.nn.functional.softmax(targets.float(), dim=-1)
    gt_ids = torch.argmax(targets, dim=-1, keepdim=True)
    conf = target_p.gather(-1, gt_ids).squeeze(-1)  # [1, T]
    return cat_prefix_weights(conf, block_size, loss_mask=loss_mask)


def draft_cat_weights(
    logits: torch.Tensor,  # [1, T, V]
    targets: torch.Tensor,  # [1, T, V]
    block_size: int,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Draft-side CAT weights from analytical acceptance overlap.

    Uses ``a_k = Σ_v min(q_v, p_v)`` (same quantity as ``tv_loss`` / DSpark
    ``accept_rate``) as the per-position confidence proxy, then prefix-cumprod.
    """
    draft_p = torch.nn.functional.softmax(logits.float(), dim=-1)
    target_p = torch.nn.functional.softmax(targets.float(), dim=-1)
    accept_rate = torch.minimum(draft_p, target_p).sum(dim=-1)  # [1, T]
    return cat_prefix_weights(accept_rate, block_size, loss_mask=loss_mask)


_LOSS_FN_MAP: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "kl_div": kl_div_loss,
    "ce": ce_loss,
    "tv": tv_loss,
    "nla": neg_log_acceptance_loss,
    "lk_hybrid": lk_hybrid_loss,
}


def resolve_loss_fn(
    name: str,
) -> "Callable[[torch.Tensor, torch.Tensor], torch.Tensor]":
    """Resolves a loss function given its abbreviated name.

    Args:
        name: ``"kl_div"`` for KL-divergence, ``"ce"`` for cross-entropy,
            ``"tv"`` for total variation, ``"nla"`` for negative
            log-acceptance, or ``"lk_hybrid"`` for the adaptive KL/TV blend.

    Returns:
        The corresponding loss function.

    Raises:
        ValueError: If *name* is not a recognised loss function.
    """
    if name not in _LOSS_FN_MAP:
        raise ValueError(
            f"Unknown loss function '{name}'. "
            f"Choose from: {sorted(_LOSS_FN_MAP.keys())}"
        )
    return _LOSS_FN_MAP[name]


def resolve_loss_config(spec: str) -> LossConfig:
    """Parse a loss spec into ``{name: (loss_fn, weight)}``.

    Accepts either a plain loss name (``"kl_div"``) or a JSON dict mapping
    loss names to weights (``'{"ce": 0.1, "tv": 0.9}'``).
    """
    if spec in _LOSS_FN_MAP:
        return {spec: (_LOSS_FN_MAP[spec], 1.0)}

    try:
        parsed = json.loads(spec)
    except json.JSONDecodeError:
        raise ValueError(
            f"Unknown loss function '{spec}'. Pass a known name "
            f"({sorted(_LOSS_FN_MAP.keys())}) or a JSON dict, "
            f'e.g. \'{{"ce": 0.1, "tv": 0.9}}\'.'
        ) from None

    if not isinstance(parsed, dict) or not parsed:
        raise ValueError(
            "Loss config must be a non-empty JSON dict mapping loss names to weights, "
            f'e.g. \'{{"ce": 0.1, "tv": 0.9}}\'. Got: {spec}'
        )

    config: LossConfig = {}
    for name, weight in parsed.items():
        if name not in _LOSS_FN_MAP:
            raise ValueError(
                f"Unknown loss function '{name}' in loss config. "
                f"Choose from: {sorted(_LOSS_FN_MAP.keys())}"
            )
        if not isinstance(weight, (int, float)):
            raise ValueError(
                f"Loss weight for '{name}' must be a number, "
                f"got {type(weight).__name__}"
            )
        config[name] = (_LOSS_FN_MAP[name], float(weight))

    return config


def compound_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_idx: torch.Tensor,
    loss_config: LossConfig,
    decay_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    position_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute a weighted sum of loss terms.

    Each entry in *loss_config* maps a name to ``(loss_fn, weight)``; the
    result is ``sum(weight * loss_function(logits, targets, ..., loss_fn))``
    over all entries.

    Returns the total loss and a dict of per-term (unweighted) scalar losses
    keyed as ``"{name}_loss"``.  When the config contains a single term the
    dict is empty (the overall loss already captures it).
    """
    total = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    term_losses: dict[str, torch.Tensor] = {}
    multi = len(loss_config) > 1
    for name, (fn, weight) in loss_config.items():
        term = loss_function(
            logits,
            targets,
            loss_mask,
            pos_idx,
            loss_fn=fn,
            decay_fn=decay_fn,
            position_weights=position_weights,
        )
        if multi:
            term_losses[f"{name}_loss"] = term.detach()
        total = total + weight * term
    return total, term_losses


def loss_function(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    loss_mask: torch.Tensor,  # shape: [1, seq_len]
    pos_idx: torch.Tensor,  # shape: [1, seq_len]
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = kl_div_loss,
    decay_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    position_weights: torch.Tensor | None = None,
):
    """Compute masked, optionally position-decayed training loss.

    Args:
        logits: Draft model logits.
        targets: Target model logits.
        loss_mask: Boolean mask selecting positions to include in the loss.
        pos_idx: Position indices within each speculative block.
        loss_fn: Per-position loss function (default: kl_div_loss).
        decay_fn: Optional position-dependent decay weighting function.
        position_weights: Optional per-position importance weights (e.g. CAT).
            Detached stop-gradient weights are expected; denominator stays
            ``loss_mask.sum()`` so this reweights importance without changing
            the effective sample count.

    Returns:
        Scalar mean loss across the batch.
    """
    elementwise_loss = loss_fn(logits, targets)  # shape: [1, seq_len]

    loss_mask = loss_mask.to(elementwise_loss.dtype)
    elementwise_loss = elementwise_loss * loss_mask

    if decay_fn is not None:
        decay_mult = decay_fn(pos_idx.to(elementwise_loss.dtype))
        elementwise_loss = elementwise_loss * decay_mult

    if position_weights is not None:
        elementwise_loss = elementwise_loss * position_weights.to(elementwise_loss.dtype)

    denominator = loss_mask.sum(dim=1) + _EPS

    batch_loss = torch.sum(elementwise_loss, dim=1) / denominator  # shape: [1]
    return batch_loss.mean()  # shape: []
