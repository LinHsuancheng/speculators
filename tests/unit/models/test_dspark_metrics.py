"""Unit tests for the DSpark loss and metrics."""

import torch

from speculators.models.dspark.metrics import (
    compute_metrics,
    exact_acceptance_length_loss,
    sampled_acceptance_credit,
)
from speculators.models.metrics import ce_loss, loss_function, resolve_loss_config

_DEFAULT_LOSS = resolve_loss_config('{"ce": 0.1, "tv": 0.9}')


def _ids_to_logits(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    logits = torch.zeros(*ids.shape, vocab_size)
    logits.scatter_(-1, ids.unsqueeze(-1), 100.0)
    return logits


class TestComputeMetrics:
    def test_perfect_draft_low_loss_high_accept(self):
        # block_size=2; position 0 is the anchor (masked), position 1 supervised.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss, metrics = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            2,
            gamma=4.0,
            loss_config=_DEFAULT_LOSS,
        )
        assert torch.isfinite(loss)
        # Matching distributions -> CE/TV ~ 0 and acceptance ~ 1.
        assert float(loss) < 1e-2
        accept = metrics["accept_rate_sum"] / metrics["accept_rate_total"]
        assert float(accept) > 0.99
        # One draft slot per block accepted w.p. ~1, plus the anchor token -> ~2.
        accept_len = metrics["accept_len_sum"] / metrics["accept_len_total"]
        assert abs(float(accept_len) - 2.0) < 1e-2

    def test_confidence_target_is_overlap(self):
        # When draft == target, accept rate == 1, so a confidence logit that is
        # very positive (sigmoid -> 1) yields ~zero abs error.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        confidence_logits = torch.full((1, 4), 20.0)  # sigmoid ~ 1.0
        _, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            gamma=4.0,
            loss_config=_DEFAULT_LOSS,
        )
        abs_err = (
            metrics["confidence_abs_error_sum"] / metrics["confidence_abs_error_total"]
        )
        assert float(abs_err) < 1e-2
        assert "confidence_loss_sum" in metrics

    def test_confidence_term_changes_loss(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss_no_conf, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
        )
        # A badly-calibrated confidence head (predicts accept~1 when accept~0)
        # must add positive BCE on top of the base loss.
        confidence_logits = torch.full((1, 4), 20.0)
        loss_conf, _ = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            confidence_head_alpha=1.0,
        )
        assert float(loss_conf) > float(loss_no_conf)

    def test_confidence_cumprod_bias_sign(self):
        # Draft != target so accept rate is ~0; an over-confident head (predicts
        # accept ~1) must show a positive cumulative-product calibration bias.
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        confidence_logits = torch.full((1, 4), 20.0)  # sigmoid ~ 1.0
        _, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
        )
        bias = (
            metrics["confidence_cumprod_bias_sum"]
            / metrics["confidence_cumprod_bias_total"]
        )
        assert float(bias) > 0.5

    def test_alpha_weighting(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss_small, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=resolve_loss_config('{"tv": 0.1}'),
        )
        loss_large, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=resolve_loss_config('{"tv": 1.0}'),
        )
        assert float(loss_large) > float(loss_small)

    def test_metric_keys_present(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = logits.clone()
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        _, metrics = compute_metrics(
            logits,
            targets,
            torch.zeros(1, 4),
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
        )
        for key in (
            "loss_sum",
            "loss_total",
            "ce_loss_sum",
            "tv_loss_sum",
            "full_acc_sum",
            "full_acc_total",
            "position_1_acc_sum",
            "accept_len_sum",
            "accept_len_total",
            "confidence_cumprod_bias_sum",
        ):
            assert key in metrics
        # all metric values must be tensors (so dist.reduce works in the trainer)
        assert all(torch.is_tensor(v) for v in metrics.values())

    def test_target_cat_changes_loss_and_logs_weight(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss_none, metrics_none = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            cat_mode="none",
        )
        loss_cat, metrics_cat = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            cat_mode="target",
        )
        assert "cat_weight_mean_sum" not in metrics_none
        assert "cat_weight_mean_sum" in metrics_cat
        # With mismatched draft/target, CAT still produces a finite loss.
        assert torch.isfinite(loss_cat)
        assert torch.isfinite(loss_none)

    def test_draft_cat_changes_loss(self):
        ids = torch.tensor([[0, 1, 0, 2]])
        logits = _ids_to_logits(ids, 8)
        targets = _ids_to_logits(torch.tensor([[0, 3, 0, 4]]), 8)
        loss_mask = torch.tensor([[0, 1, 0, 1]], dtype=torch.float32)
        loss_none, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            cat_mode="none",
        )
        loss_draft, metrics = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=2,
            loss_config=_DEFAULT_LOSS,
            cat_mode="draft",
        )
        assert torch.isfinite(loss_draft)
        # Mismatched distributions -> low accept_rate -> later CAT weights < 1,
        # so draft-CAT loss should be <= unweighted loss for the same terms.
        assert float(loss_draft) <= float(loss_none) + 1e-5
        assert "cat_weight_mean_sum" in metrics


class TestSampledAcceptanceLoss:
    def test_credit_matches_formula(self):
        draft_logp = torch.log(torch.tensor([[0.2, 0.8, 0.5]]))
        target_logp = torch.log(torch.tensor([[0.4, 0.4, 0.75]]))

        credit = sampled_acceptance_credit(draft_logp, target_logp)

        alpha = torch.tensor([[1.0, 0.5, 1.0]])
        survival = torch.cumprod(alpha, dim=-1)
        continuation = torch.flip(
            torch.cumsum(torch.flip(survival, dims=[-1]), dim=-1),
            dims=[-1],
        )
        undercovered = torch.tensor([[1.0, 0.0, 1.0]])
        expected = undercovered * continuation
        assert torch.allclose(credit, expected)

    def test_exact_loss_only_backprops_through_draft_logp(self):
        draft_logp = torch.log(torch.tensor([[0.2, 0.8, 0.5]])).requires_grad_()
        target_logp = torch.log(torch.tensor([[0.4, 0.4, 0.75]])).requires_grad_()

        loss = exact_acceptance_length_loss(draft_logp, target_logp)
        loss.backward()

        credit = sampled_acceptance_credit(draft_logp, target_logp)
        expected_grad = -credit / draft_logp.shape[-1]
        assert torch.allclose(draft_logp.grad, expected_grad)
        assert target_logp.grad is None

    def test_sampled_metrics_use_credit_instead_of_gamma_decay(self):
        logits = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0],
                    [0.0, 0.0, 4.0],
                ]
            ],
            requires_grad=True,
        )
        targets = _ids_to_logits(torch.tensor([[0, 0, 1, 2]]), 3)
        loss_mask = torch.tensor([[0, 1, 1, 1]], dtype=torch.float32)
        draft_logp = torch.log(torch.tensor([[0.2, 0.8, 0.5]])).requires_grad_()
        target_logp = torch.log(torch.tensor([[0.4, 0.4, 0.75]]))

        loss, metrics = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=4,
            loss_config=resolve_loss_config("ce"),
            gamma=0.01,
            sampled_draft_logprobs=draft_logp,
            sampled_target_logprobs=target_logp,
            sampled_acceptance_loss_alpha=1.0,
        )

        credit = sampled_acceptance_credit(draft_logp, target_logp)
        position_weights = torch.tensor([[0.0, *credit[0].tolist()]])
        expected_ce = loss_function(
            logits,
            targets,
            loss_mask,
            torch.tensor([[0, 1, 2, 3]]),
            loss_fn=ce_loss,
            decay_fn=None,
            position_weights=position_weights,
        )
        expected_exact = exact_acceptance_length_loss(
            draft_logp, target_logp, credit=credit
        )

        assert torch.allclose(loss, expected_ce + expected_exact)
        assert "sampled_acceptance_loss_sum" in metrics
        assert "sampled_credit_mean_sum" in metrics
        assert "cat_weight_mean_sum" not in metrics

    def test_sampled_confidence_loss_uses_credit_weights(self):
        logits = _ids_to_logits(torch.tensor([[0, 1, 2, 0]]), 4)
        targets = _ids_to_logits(torch.tensor([[0, 1, 2, 0]]), 4)
        loss_mask = torch.tensor([[0, 1, 1, 1]], dtype=torch.float32)
        confidence_logits = torch.zeros(1, 4)
        draft_logp = torch.log(torch.tensor([[0.2, 0.8, 0.5]]))
        target_logp = torch.log(torch.tensor([[0.4, 0.4, 0.75]]))

        loss_no_conf, _ = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=4,
            loss_config=resolve_loss_config("ce"),
            sampled_draft_logprobs=draft_logp,
            sampled_target_logprobs=target_logp,
        )
        loss_conf, metrics = compute_metrics(
            logits,
            targets,
            confidence_logits,
            loss_mask,
            block_size=4,
            loss_config=resolve_loss_config("ce"),
            confidence_head_alpha=1.0,
            sampled_draft_logprobs=draft_logp,
            sampled_target_logprobs=target_logp,
        )

        assert float(loss_conf) > float(loss_no_conf)
        assert "confidence_loss_sum" in metrics
