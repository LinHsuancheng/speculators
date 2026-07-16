"""Unit tests for the DSpark on-policy path: rollout, scorer, and wiring.

All tests are CPU-only (float32) so they run on Ascend/CI without a GPU/NPU or a
live vLLM server. The rollout is exercised by calling ``_sample_block_rollout``
directly with synthetic base logits/hidden states (no backbone forward, no
verifier weights needed); the vLLM scoring is exercised via ``MockVerifierScorer``.
"""

import copy

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from speculators import SpeculatorsConfig, VerifierConfig
from speculators.models.dspark.config import DSparkSpeculatorConfig
from speculators.models.dspark.core import DSparkDraftModel
from speculators.models.dspark.metrics import compute_metrics
from speculators.models.dspark.onpolicy import (
    MockVerifierScorer,
    build_scored_sequences,
)
from speculators.models.metrics import resolve_loss_config
from speculators.proposals.greedy import GreedyTokenProposalConfig

_VERIFIER_VOCAB = 32
_DRAFT_VOCAB = 16
_HIDDEN = 32
_BLOCK = 4  # K = block - 1 = 3 draft slots
_MAX_ANCHORS = 5


def _tiny_dspark(head_type: str = "vanilla") -> DSparkDraftModel:
    """Build a tiny CPU DSpark model with deterministic (non-NaN) weights."""
    transformer_config = Qwen3Config(
        vocab_size=_VERIFIER_VOCAB,
        hidden_size=_HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    transformer_config._attn_implementation = "eager"  # noqa: SLF001
    config = DSparkSpeculatorConfig(
        transformer_layer_config=transformer_config,
        draft_vocab_size=_DRAFT_VOCAB,
        block_size=_BLOCK,
        max_anchors=_MAX_ANCHORS,
        aux_hidden_state_layer_ids=[0, 1, 2],
        mask_token_id=0,
        markov_rank=8,
        markov_head_type=head_type,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        speculators_config=SpeculatorsConfig(
            algorithm="dspark",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=_BLOCK - 1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=None, architectures=["Qwen3ForCausalLM"]
            ),
        ),
    )
    model = DSparkDraftModel(config)
    with torch.no_grad():
        for param in model.parameters():
            if param.isnan().any():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
        for buf in model.buffers():
            if buf.is_floating_point() and buf.isnan().any():
                buf.zero_()
    # Draft-vocab mappings: draft id i -> verifier id 2*i (an arbitrary injection).
    t2d = torch.zeros(_VERIFIER_VOCAB, dtype=torch.bool)
    d2t = torch.zeros(_DRAFT_VOCAB, dtype=torch.long)
    for i in range(_DRAFT_VOCAB):
        t2d[2 * i] = True
        d2t[i] = 2 * i
    model.load_vocab_mappings(t2d, d2t)
    return model.to(torch.float32)


class TestSampleBlockRollout:
    def _inputs(self, num_blocks=_MAX_ANCHORS):
        torch.manual_seed(0)
        base_logits = torch.randn(num_blocks, _BLOCK, _DRAFT_VOCAB, requires_grad=True)
        hidden = torch.randn(num_blocks, _BLOCK, _HIDDEN)
        anchor_ids = torch.randint(0, _VERIFIER_VOCAB, (num_blocks,))
        return base_logits, hidden, anchor_ids

    def test_shapes_and_grad(self):
        model = _tiny_dspark()
        base_logits, hidden, anchor_ids = self._inputs()
        sampled, q_logp, sampled_v = model._sample_block_rollout(  # noqa: SLF001
            base_logits=base_logits,
            hidden_blocks=hidden,
            anchor_token_ids=anchor_ids,
            temperature=1.0,
        )
        k = _BLOCK - 1
        assert sampled.shape == (_MAX_ANCHORS, k)
        assert q_logp.shape == (_MAX_ANCHORS, k)
        assert sampled_v.shape == (_MAX_ANCHORS, k)
        # q_logp are log-probs of a categorical -> strictly negative, finite.
        assert torch.isfinite(q_logp).all()
        assert (q_logp <= 0).all()
        # Gradient flows into base_logits and the Markov head through log q_t.
        q_logp.sum().backward()
        assert base_logits.grad is not None
        assert torch.isfinite(base_logits.grad).all()
        assert base_logits.grad.abs().sum() > 0
        assert model.markov_head.markov_w2.weight.grad is not None

    def test_sampled_ids_in_draft_vocab_range(self):
        model = _tiny_dspark()
        base_logits, hidden, anchor_ids = self._inputs()
        sampled, _, sampled_v = model._sample_block_rollout(  # noqa: SLF001
            base_logits=base_logits,
            hidden_blocks=hidden,
            anchor_token_ids=anchor_ids,
            temperature=1.0,
        )
        assert int(sampled.min()) >= 0
        assert int(sampled.max()) < _DRAFT_VOCAB
        # Verifier ids are the d2t image (= 2 * draft id here).
        assert torch.equal(sampled_v, sampled * 2)

    def test_generator_determinism(self):
        model = _tiny_dspark()
        base_logits, hidden, anchor_ids = self._inputs()
        g1 = torch.Generator().manual_seed(1234)
        g2 = torch.Generator().manual_seed(1234)
        s1, _, _ = model._sample_block_rollout(  # noqa: SLF001
            base_logits=base_logits, hidden_blocks=hidden,
            anchor_token_ids=anchor_ids, temperature=1.0, generator=g1,
        )
        s2, _, _ = model._sample_block_rollout(  # noqa: SLF001
            base_logits=base_logits, hidden_blocks=hidden,
            anchor_token_ids=anchor_ids, temperature=1.0, generator=g2,
        )
        assert torch.equal(s1, s2)

    def test_low_temperature_is_greedy(self):
        # As T -> 0, Gumbel-max collapses to argmax of the (biased) logits.
        model = _tiny_dspark()
        base_logits, hidden, anchor_ids = self._inputs()
        g = torch.Generator().manual_seed(7)
        sampled, _, _ = model._sample_block_rollout(  # noqa: SLF001
            base_logits=base_logits, hidden_blocks=hidden,
            anchor_token_ids=anchor_ids, temperature=1e-4, generator=g,
        )
        # Recompute the greedy path explicitly, feeding sampled prev tokens.
        expected = torch.zeros_like(sampled)
        prev = anchor_ids.long()
        for k in range(1, _BLOCK):
            logits_k = base_logits[:, k, :].detach()
            bias = model.markov_head.step_bias(
                prev_token_ids=prev, hidden_states=hidden[:, k, :]
            )
            greedy = torch.argmax(logits_k + bias, dim=-1)
            expected[:, k - 1] = greedy
            prev = (greedy * 2).long()  # d2t image
        assert torch.equal(sampled, expected)

    def test_gated_head_rollout_runs(self):
        model = _tiny_dspark(head_type="gated")
        base_logits, hidden, anchor_ids = self._inputs()
        sampled, q_logp, _ = model._sample_block_rollout(  # noqa: SLF001
            base_logits=base_logits, hidden_blocks=hidden,
            anchor_token_ids=anchor_ids, temperature=1.0,
        )
        assert sampled.shape == (_MAX_ANCHORS, _BLOCK - 1)
        assert torch.isfinite(q_logp).all()


class TestBuildScoredSequences:
    def test_single_document_prefix(self):
        # One document, anchors at positions 2 and 4, K=2 sampled tokens.
        gold = torch.tensor([[10, 11, 12, 13, 14, 15]])
        docs = torch.zeros(1, 6, dtype=torch.long)
        anchors = torch.tensor([2, 4])
        sampled_v = torch.tensor([[91, 92], [93, 94]])
        seqs = build_scored_sequences(gold, docs, anchors, sampled_v)
        # block 0: gold[0:3] + [91,92]; block 1: gold[0:5] + [93,94]
        assert seqs[0] == [10, 11, 12, 91, 92]
        assert seqs[1] == [10, 11, 12, 13, 14, 93, 94]

    def test_multi_document_boundary(self):
        # Two packed documents: doc 0 = positions 0-2, doc 1 = positions 3-5.
        gold = torch.tensor([[10, 11, 12, 20, 21, 22]])
        docs = torch.tensor([[0, 0, 0, 1, 1, 1]])
        anchors = torch.tensor([4])  # anchor in doc 1
        sampled_v = torch.tensor([[99]])
        seqs = build_scored_sequences(gold, docs, anchors, sampled_v)
        # Prefix must start at doc-1 boundary (position 3), not the packed start.
        assert seqs[0] == [20, 21, 99]


class TestMockVerifierScorer:
    def test_fixed_hidden_shape(self):
        scorer = MockVerifierScorer(
            hidden_size=_HIDDEN, fixed_hidden=torch.zeros(1, 1, _HIDDEN)
        )
        gold = torch.randint(0, _VERIFIER_VOCAB, (1, 20))
        docs = torch.zeros(1, 20, dtype=torch.long)
        anchors = torch.tensor([3, 7, 11])
        k = _BLOCK - 1
        sampled_v = torch.randint(0, _VERIFIER_VOCAB, (3, k))
        h = scorer.score(
            gold_input_ids=gold,
            document_ids=docs,
            anchor_positions=anchors,
            sampled_verifier_ids=sampled_v,
        )
        assert h.shape == (3, k, _HIDDEN)
        assert torch.isfinite(h).all()


class TestOnPolicyForward:
    def _batch(self, total_seq_len=48):
        torch.manual_seed(0)
        hidden_states = torch.randn(1, total_seq_len, 3 * _HIDDEN)
        input_ids = torch.randint(0, _VERIFIER_VOCAB, (1, total_seq_len))
        loss_mask = torch.ones(1, total_seq_len)
        verifier_last = torch.randn(1, total_seq_len, _HIDDEN)
        document_ids = torch.zeros(1, total_seq_len, dtype=torch.long)
        return {
            "hidden_states": hidden_states,
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "verifier_last_hidden_states": verifier_last,
            "document_ids": document_ids,
        }

    def test_forward_onpolicy_end_to_end(self):
        model = _tiny_dspark()

        # Token-dependent fake verifier so p_t actually varies with the sampled
        # tokens (more meaningful than a constant hidden state).
        def _fake_verifier(seq: list[int]) -> torch.Tensor:
            gen = torch.Generator().manual_seed(1000 + sum(seq))
            return torch.randn(len(seq), _HIDDEN, generator=gen)

        model.verifier_scorer = MockVerifierScorer(
            hidden_size=_HIDDEN, verifier=_fake_verifier
        )
        batch = self._batch()
        loss_config = resolve_loss_config('{"ce": 0.1, "accept_length": 0.9}')
        draft_tokens, loss, metrics = model(
            **batch,
            loss_config=loss_config,
            confidence_head_alpha=1.0,
            onpolicy_sampling=True,
            sampling_temperature=1.0,
        )
        assert torch.isfinite(loss)
        loss.backward()
        # Gradient must reach the draft params via log q_t (accept_length term).
        grads = [
            p.grad for p in model.markov_head.parameters() if p.grad is not None
        ]
        assert grads, "no gradient reached the Markov head"
        assert any(g.abs().sum() > 0 for g in grads)
        assert "accept_length_loss_sum" in metrics
        assert "ce_loss_sum" in metrics

    def test_missing_scorer_raises(self):
        model = _tiny_dspark()  # no scorer injected
        batch = self._batch()
        loss_config = resolve_loss_config('{"ce": 0.1, "accept_length": 0.9}')
        with pytest.raises(RuntimeError, match="verifier scorer"):
            model(
                **batch,
                loss_config=loss_config,
                onpolicy_sampling=True,
                sampling_temperature=1.0,
            )

    def test_teacher_forced_path_unaffected(self):
        # Without onpolicy_sampling the model must still run the TF compound loss.
        model = _tiny_dspark()
        batch = self._batch()
        loss_config = resolve_loss_config('{"ce": 0.1, "tv": 0.9}')
        _, loss, metrics = model(
            **batch,
            loss_config=loss_config,
            confidence_head_alpha=1.0,
            onpolicy_sampling=False,
        )
        assert torch.isfinite(loss)
        assert "tv_loss_sum" in metrics
        assert "accept_length_loss_sum" not in metrics
