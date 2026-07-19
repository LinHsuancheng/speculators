# Teacher-Forced Sequence-Level Expected-Acceptance-Length (TF-EAL) Loss Implementation

## Summary

This branch (`aux_loss`) adds a **teacher-forced sequence-level expected-acceptance-length (TF-EAL) loss** to DSpark, implementing the auxiliary training objective described in `trials.md`.

### Core Formula

**TF-EAL Loss**: `L = -R_TF`, where:

```
R_TF = Σ_{k=1}^K S_k
S_k = Π_{i=1}^k a_i
a_i = Σ_v min(p_i(v), q_i(v)) = 1 - TV(p_i, q_i)
```

- `a_i`: per-position acceptance probability (distributional overlap, already computed by DSpark's `tv_loss`)
- `S_k`: cumulative survival probability (first `k` tokens all accepted)
- `R_TF`: expected accepted draft length on the teacher-forced path
- `τ_TF = R_TF + 1`: total tokens per block (including the always-emitted anchor)

### Key Properties

1. **No position decay by design**: The survival product `S_k = Π a_i` naturally gates later positions—early positions with low overlap automatically shrink all downstream survivals. The per-position gradient weight (continuation credit) `C_t = Σ_{k≥t} S_k` already encodes the continuation value, so manual exponential decay is not only unnecessary but theoretically incorrect here.

2. **Teacher-forced approximation**: Unlike the on-policy sampled exact loss (which requires non-teacher-forcing draft sampling), this loss is computed directly on the existing teacher-forced target logits. It approximates the true on-policy expected acceptance length with error `O(K² ε)` when per-step `TV(p, q) ≤ ε`, making it valid once the draft has reasonable acceptance rate (typically after warm-up with CE+TV).

3. **Differentiable through draft logits**: Gradients flow through the `min(p, q)` overlap directly to the draft model parameters (via the softmax), avoiding REINFORCE-style high variance.

4. **Rich per-position logging**: Reports `survival_pos_k` and `credit_pos_k` for every draft slot, so you can see exactly which positions contribute to the sequence-level objective.

## Files Changed

### Core Implementation

1. **`src/speculators/models/metrics.py`**: Added `tf_eal_loss(logits, targets, loss_mask, block_size)` helper (lines 154–246).
   - Computes differentiable `R_TF = Σ_k Π_{i≤k} a_i` on teacher-forced path.
   - Returns `(loss, aux)` where `aux` contains per-block survival/continuation/tau for logging.
   - Handles masked slots (act as `a_i=1` inside cumulative product, excluded from sum).

2. **`src/speculators/models/dspark/metrics.py`**:
   - Imported `tf_eal_loss`.
   - Added `tf_eal_alpha` parameter to `compute_metrics` (default `0.0`, disabled).
   - When `tf_eal_alpha > 0`, computes TF-EAL loss and adds it to the total loss.
   - Logs:
     - `tf_eal_loss`: the term `-R_TF`
     - `tf_eal_tau`: expected tokens per block `τ = R_TF + 1`
     - `tf_eal_survival_pos_{k}`: `S_k` for draft slot `k`
     - `tf_eal_credit_pos_{k}`: continuation credit `C_k = Σ_{j≥k} S_j` for slot `k`

3. **`src/speculators/models/dspark/core.py`**:
   - Wired `tf_eal_alpha` through `get_trainer_kwargs` and `forward`.

4. **`scripts/train.py`**: Added `--tf-eal-alpha` CLI argument (default `0.0`).

### Tests

5. **`tests/unit/models/test_dspark_metrics.py`**: Added `TestTfEalLoss` class with 4 tests:
   - `test_disabled_by_default`: confirms `tf_eal_alpha=0` produces no TF-EAL metrics.
   - `test_perfect_overlap_tau_equals_block`: perfect `p=q` → `τ=K+1`, `loss=-K`.
   - `test_credit_decreases_with_position`: confirms natural position weighting `C_1 ≥ C_2 ≥ ... ≥ C_K`.
   - `test_changes_total_loss`: enabling `tf_eal_alpha > 0` changes the training loss.

## Usage

### Training Command

Add `--tf-eal-alpha <weight>` to your existing DSpark training command:

```bash
python scripts/train.py \
  --verifier-name-or-path meta-llama/Llama-3.1-8B-Instruct \
  --speculator-type dspark \
  --block-size 8 \
  --loss-fn '{"ce": 0.1, "tv": 0.9}' \
  --dflash-decay-gamma 4.0 \
  --confidence-head-alpha 1.0 \
  --tf-eal-alpha 0.5 \
  ...
```

### Recommended Curriculum

**Stage 1 (warm-up)**: Use `--tf-eal-alpha 0.0` (disabled) for the first ~10-20% of training. Let CE+TV+confidence bring the draft into a reasonable acceptance region.

**Stage 2 (sequence-level)**: Enable `--tf-eal-alpha 0.5` (or gradually ramp it up) once:
- `position_1_acc ≳ 0.6`
- `accept_rate ≳ 0.6`
- `accept_len (τ_old) ≳ 1.5`

Monitor `tf_eal_tau` in the logs—it should start near your current `accept_len` and gradually increase as the draft learns to optimize the full sequence-level objective.

**Stage 3 (optional fine-tune)**: If you later implement non-teacher-forcing draft sampling, the on-policy sampled exact loss can refine away the teacher-forced prefix bias.

### What Gets Logged

With `--tf-eal-alpha > 0`, every training step logs:

```
train/tf_eal_loss          # the TF-EAL term -R_TF (negative, bigger magnitude = longer acceptance)
train/tf_eal_tau           # expected tokens per block (R_TF + 1), should increase over training
train/tf_eal_survival_pos_1  # S_1 (first draft token acceptance)
train/tf_eal_survival_pos_2  # S_2 (first two accepted)
...
train/tf_eal_credit_pos_1    # C_1 = Σ_{k≥1} S_k (credit for position 1, largest)
train/tf_eal_credit_pos_2    # C_2 (smaller, gated by earlier positions)
...
```

These per-position metrics let you see exactly where the draft is failing or succeeding across the block.

## Relation to Other Objectives

| Method | Path | Credit | Decay | Target |
|--------|------|--------|-------|--------|
| **AUF** | gold continuation | hard 0/1 mask at first argmax error | implicit (via truncation) | curriculum proxy |
| **LK Losses** | n/a (per-position) | fixed position weight `γ^t` | manual `γ` | single-step acceptance `β_t` |
| **DSpark CE+TV** | teacher-forced | exponential `e^{-(t-1)/γ}` | manual `γ=4` | per-position overlap |
| **TF-EAL (this)** | teacher-forced | continuation `Σ_{k≥t} S_k` | **intrinsic** (no `γ`) | sequence-level expected length |

**TF-EAL** is the only one that directly optimizes the sequence-level acceptance length, carries its own natural position weighting, and requires no manual decay hyperparameter.

## Implementation Notes

### Why No Exponential Decay?

The original DSpark/DFlash losses use `decay_fn(t) = e^{-(t-1)/γ}` because they optimize per-position objectives (KL/TV/CE) and need manual position weighting to prefer early tokens. The TF-EAL loss already has **intrinsic position importance** via the survival product:

```
∂R_TF/∂a_t = S_{t-1} · Σ_{k≥t} S_k / a_t
```

This gradient naturally amplifies early positions (they gate all later ones) and shrinks late positions (only affect themselves). Adding exponential decay on top would be a **double penalty** and would break the theoretical correspondence to the exact objective.

### Masked Slot Handling

Masked slots (padding, invalid positions) are handled by:
1. Setting their `a_i → 1` inside the cumulative product `cumprod`, so they don't shrink later survivals.
2. Zeroing them out in the final sum `Σ_k S_k`, so they don't contribute to `R_TF`.
3. Excluding masked blocks from the denominator when averaging the loss.

This matches DSpark's existing `accept_prefix` logic (lines 117-119 of `dspark/metrics.py`).

## Testing

Run the new tests:

```bash
pytest tests/unit/models/test_dspark_metrics.py::TestTfEalLoss -v
```

Expected:
- `test_disabled_by_default`: PASS
- `test_perfect_overlap_tau_equals_block`: PASS (τ ≈ 4.0, loss ≈ -3.0)
- `test_credit_decreases_with_position`: PASS (C₁ ≥ C₂ ≥ C₃)
- `test_changes_total_loss`: PASS

All existing DSpark tests should also pass unchanged (the new loss is disabled by default).

## Future Work

1. **On-policy sampled exact loss**: The current TF-EAL is a teacher-forced approximation. For full accuracy, implement non-teacher-forcing draft sampling and use the exact credit `C_t = 𝟙[q_t < p_t] Σ_{k≥t} S_k` from `trials.md` §9-10.

2. **Dynamic curriculum**: Auto-enable TF-EAL when `accept_rate` crosses a threshold (e.g., 0.6), avoiding manual two-stage training.

3. **Block-size ablation**: Test whether longer blocks (e.g., 16 vs 8) benefit more from sequence-level vs per-position objectives.

## References

- **trials.md**: Full mathematical derivation of the exact on-policy gradient and the teacher-forced approximation.
- **DSpark paper**: Original sequential-head draft model with confidence and CAT weighting.
- **LK Losses (arXiv 2602.23881)**: Per-position acceptance optimization (this work extends to sequence-level).
