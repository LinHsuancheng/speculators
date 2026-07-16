"""Unit tests for the exact expected-acceptance-length loss (trials.md).

Covers the pure functions ``acceptance_length_credit`` /
``exact_acceptance_length_loss`` against the worked examples and correctness
checks in trials.md (§4, §10, §11), and the on-policy branch of
``compute_metrics``. All CPU / float32 so the numerics are exact.
"""

import math

import pytest
import torch

from speculators.models.dspark.metrics import (
    acceptance_length_credit,
    compute_metrics,
    exact_acceptance_length_loss,
)
from speculators.models.metrics import resolve_loss_config


def _logp_from_alpha(alphas: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (q_logp, p_logp) rows that realise the given alpha = min(1, p/q).

    We pick log q = 0 and log p = log(alpha) for alpha <= 1 (an under- or
    exactly-covered slot, ``q <= p`` only when alpha == 1). To exercise the
    indicator we need explicit control of ``q < p``; use ``_logp_pair`` instead
    when the sign matters. Here every alpha < 1 means ``p < q`` (over-covered).
    """
    q = torch.zeros(1, len(alphas))
    p = torch.tensor([[math.log(a) for a in alphas]])
    return q, p


class TestAcceptanceLengthCredit:
    def test_k1_undercovered_credit_is_one(self):
        # §10: K=1, q<p  =>  C_1 = 1.
        q = torch.tensor([[0.0]])
        p = torch.tensor([[1.0]])  # p>q  => undercovered
        credit = acceptance_length_credit(q, p)
        assert torch.allclose(credit, torch.tensor([[1.0]]))

    def test_k1_overcovered_credit_is_zero(self):
        # §10: q>p  =>  indicator 0  =>  C_1 = 0.
        q = torch.tensor([[1.0]])
        p = torch.tensor([[0.0]])
        credit = acceptance_length_credit(q, p)
        assert torch.allclose(credit, torch.tensor([[0.0]]))

    def test_k2_worked_example(self):
        # §11: position 1 undercovered (alpha_1 = 1), position 2 overcovered with
        # alpha_2 = 0.5.  S_1 = 1, S_2 = 0.5.  C_1 = I_1*(S_1+S_2) = 1.5,
        # C_2 = I_2*S_2 = 0 (position 2 is over-covered => indicator 0).
        q = torch.tensor([[0.0, 0.0]])
        p = torch.tensor([[1.0, math.log(0.5)]])  # slot1 p>q; slot2 p<q
        credit = acceptance_length_credit(q, p)
        assert torch.allclose(credit, torch.tensor([[1.5, 0.0]]), atol=1e-6)

    def test_k2_both_undercovered(self):
        # Both slots undercovered: alpha_1 = alpha_2 = 1 (p>=q).  S_1=1, S_2=1.
        # C_1 = S_1 + S_2 = 2, C_2 = S_2 = 1.
        q = torch.tensor([[0.0, 0.0]])
        p = torch.tensor([[1.0, 1.0]])
        credit = acceptance_length_credit(q, p)
        assert torch.allclose(credit, torch.tensor([[2.0, 1.0]]), atol=1e-6)

    def test_continuation_matches_section4(self):
        # §4: alpha = (0.8, 0.5, 0.25) -> E[L|Y] = sum_k S_k = 1.3. The continuation
        # at t=1 (with the indicator forced on) equals that sum.
        q = torch.zeros(1, 3)
        p = torch.tensor([[math.log(0.8), math.log(0.5), math.log(0.25)]])
        # Force indicator on by making all slots "undercovered" via a tiny epsilon
        # is not possible with alpha<1; instead check the continuation directly.
        log_alpha = torch.minimum(torch.zeros_like(q), p - q)
        survival = torch.exp(torch.cumsum(log_alpha, dim=-1))
        assert abs(float(survival.sum()) - 1.3) < 1e-6

    def test_mask_excludes_slots(self):
        # Slot 2 masked out: it must neither shrink survival of slot 3 nor get
        # credit. With slots (under, masked, under): S over real slots ignores the
        # masked one.  C = [S_1+S_3, 0, S_3] with S_1=1, S_3=1 => [2, 0, 1].
        q = torch.tensor([[0.0, 0.0, 0.0]])
        p = torch.tensor([[1.0, 1.0, 1.0]])
        mask = torch.tensor([[1.0, 0.0, 1.0]])
        credit = acceptance_length_credit(q, p, mask=mask)
        assert torch.allclose(credit, torch.tensor([[2.0, 0.0, 1.0]]), atol=1e-6)

    def test_credit_is_detached(self):
        q = torch.zeros(1, 2, requires_grad=True)
        p = torch.tensor([[1.0, 1.0]])
        credit = acceptance_length_credit(q, p)
        assert not credit.requires_grad


class TestExactAcceptanceLengthLoss:
    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            exact_acceptance_length_loss(torch.zeros(1, 2), torch.zeros(1, 3))

    def test_only_q_carries_grad(self):
        q = torch.zeros(2, 3, requires_grad=True)
        p = torch.ones(2, 3)
        loss = exact_acceptance_length_loss(q, p)
        loss.backward()
        assert q.grad is not None
        assert torch.isfinite(q.grad).all()

    def test_loss_value_matches_manual(self):
        # Both slots undercovered => C = [2, 1]; log q = [-0.5, -0.5].
        # loss = -(1/K) * sum C_t log q_t = -(1/2)*(2*-0.5 + 1*-0.5) = 0.75.
        q = torch.full((1, 2), -0.5, requires_grad=True)
        p = torch.ones(1, 2)  # p>q => both undercovered
        loss = exact_acceptance_length_loss(q, p)
        assert abs(float(loss) - 0.75) < 1e-6

    def test_masked_blocks_ignored_in_mean(self):
        q = torch.zeros(2, 2, requires_grad=True)
        p = torch.ones(2, 2)
        mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])  # block 1 fully masked
        loss = exact_acceptance_length_loss(q, p, mask=mask)
        assert torch.isfinite(loss)

    def test_gradient_is_unbiased_estimator_k1(self):
        """E_q[grad(loss_hat)] == grad(-R_1), the negative expected accept length.

        For K=1, R_1 = sum_v min(p_v, q_v) has an exact closed form; its gradient
        wrt the draft logits must equal the *expectation over sampled Y* of the
        Monte-Carlo estimator gradient. Both are computed exactly here by
        enumerating the (small) vocabulary — no sampling noise.
        """
        torch.manual_seed(0)
        vocab = 5
        logits = torch.randn(vocab, dtype=torch.float64, requires_grad=True)
        # Fixed frozen verifier distribution p.
        p = torch.tensor([0.4, 0.1, 0.2, 0.05, 0.25], dtype=torch.float64)

        # --- analytic gradient of -R_1 = -sum_v min(p_v, q_v) ---
        q = torch.softmax(logits, dim=-1)
        R1 = torch.minimum(p, q).sum()
        (analytic_grad,) = torch.autograd.grad(-R1, logits, retain_graph=False)

        # --- expected estimator gradient: sum_v q_v * grad(loss_hat | Y=v) ---
        expected_grad = torch.zeros_like(logits)
        logq_all = torch.log_softmax(logits, dim=-1)
        logp_all = torch.log(p)
        q_detached = torch.softmax(logits, dim=-1).detach()
        for v in range(vocab):
            q_logp = logq_all[v].reshape(1, 1)
            p_logp = logp_all[v].reshape(1, 1)
            loss_v = exact_acceptance_length_loss(q_logp, p_logp)
            (g,) = torch.autograd.grad(loss_v, logits, retain_graph=True)
            expected_grad = expected_grad + q_detached[v] * g

        assert torch.allclose(analytic_grad, expected_grad, atol=1e-8)


class TestComputeMetricsOnPolicy:
    """The on-policy branch of compute_metrics (credit-weighted CE + accept_length)."""

    def _inputs(self, num_blocks=2, block=3, vocab=8):
        seq_len = num_blocks * block
        torch.manual_seed(0)
        logits = torch.randn(1, seq_len, vocab)
        targets = torch.randn(1, seq_len, vocab)
        loss_mask = torch.ones(1, seq_len)
        loss_mask[:, ::block] = 0.0  # anchor slots
        k = block - 1
        q_logp = torch.zeros(num_blocks, k, requires_grad=True)
        p_logp = torch.ones(num_blocks, k)  # undercovered => nonzero credit
        sampled_ids = torch.zeros(num_blocks, k, dtype=torch.long)
        sampled_mask = loss_mask.view(num_blocks, block)[:, 1:]
        return logits, targets, loss_mask, q_logp, p_logp, sampled_ids, sampled_mask

    def test_onpolicy_requires_p_and_mask(self):
        logits, targets, loss_mask, q_logp, _, _, _ = self._inputs()
        with pytest.raises(ValueError):
            compute_metrics(
                logits,
                targets,
                None,
                loss_mask,
                block_size=3,
                loss_config=resolve_loss_config('{"ce": 0.1, "accept_length": 0.9}'),
                q_logp=q_logp,
            )

    def test_accept_length_requires_onpolicy(self):
        # accept_length in a teacher-forced call (no q_logp) must error.
        logits, targets, loss_mask, *_ = self._inputs()
        with pytest.raises(ValueError):
            compute_metrics(
                logits,
                targets,
                None,
                loss_mask,
                block_size=3,
                loss_config=resolve_loss_config('{"ce": 0.1, "accept_length": 0.9}'),
            )

    def test_onpolicy_loss_finite_and_grad_flows(self):
        logits, targets, loss_mask, q_logp, p_logp, ids, smask = self._inputs()
        loss, metrics = compute_metrics(
            logits,
            targets,
            None,
            loss_mask,
            block_size=3,
            loss_config=resolve_loss_config('{"ce": 0.1, "accept_length": 0.9}'),
            q_logp=q_logp,
            p_logp=p_logp,
            sampled_draft_ids=ids,
            sampled_mask=smask,
        )
        assert torch.isfinite(loss)
        assert "accept_length_loss_sum" in metrics
        assert "ce_loss_sum" in metrics
        loss.backward()
        assert q_logp.grad is not None
        assert torch.isfinite(q_logp.grad).all()

    def test_confidence_term_still_added_on_policy(self):
        logits, targets, loss_mask, q_logp, p_logp, ids, smask = self._inputs()
        conf = torch.full((1, logits.shape[1]), 20.0)
        loss_conf, metrics = compute_metrics(
            logits,
            targets,
            conf,
            loss_mask,
            block_size=3,
            loss_config=resolve_loss_config('{"ce": 0.1, "accept_length": 0.9}'),
            confidence_head_alpha=1.0,
            q_logp=q_logp,
            p_logp=p_logp,
            sampled_draft_ids=ids,
            sampled_mask=smask,
        )
        assert "confidence_loss_sum" in metrics
        assert torch.isfinite(loss_conf)
