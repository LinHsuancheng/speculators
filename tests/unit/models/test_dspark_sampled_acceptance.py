"""Regression tests for DSpark sampled-acceptance replay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import speculators.models.dflash.core as dflash_core
import speculators.train.sampled_acceptance as sampled_acceptance
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dspark.core import DSparkDraftModel
from speculators.train.sampled_acceptance import (
    SampledAcceptanceAugmentor,
    SampledAcceptanceConfig,
)


class RecordingMarkovHead:
    def __init__(self, draft_vocab_size: int) -> None:
        self.draft_vocab_size = draft_vocab_size
        self.prev_token_calls: list[torch.Tensor] = []

    def prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return token_ids.unsqueeze(-1).float()

    def block_bias(
        self,
        *,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        prev_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.prev_token_calls.append(prev_token_ids.detach().clone())
        return torch.zeros(
            *prev_token_ids.shape,
            self.draft_vocab_size,
            device=prev_token_ids.device,
            dtype=hidden_states.dtype,
        )


def test_build_attention_mask_uses_provided_anchors(monkeypatch):
    model = DFlashDraftModel.__new__(DFlashDraftModel)
    model.config = SimpleNamespace(max_anchors=3)
    model.block_size = 4
    model.uses_full_attn = False
    model.uses_sliding_window_attn = False

    def fail_select_anchors(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("select_anchors should not be called")

    monkeypatch.setattr(dflash_core, "select_anchors", fail_select_anchors)

    loss_mask = torch.ones(1, 16, dtype=torch.bool)
    document_ids = torch.zeros(1, 16, dtype=torch.long)
    anchors = torch.tensor([9, 2, 0])
    valid = torch.tensor([True, True, False])

    _, _, actual_anchors, actual_valid = model._build_attention_mask(
        loss_mask,
        document_ids,
        torch.device("cpu"),
        anchor_positions=anchors,
        anchor_valid=valid,
    )

    assert torch.equal(actual_anchors, anchors)
    assert torch.equal(actual_valid, valid)


def test_build_attention_mask_rejects_anchor_count_mismatch():
    model = DFlashDraftModel.__new__(DFlashDraftModel)
    model.config = SimpleNamespace(max_anchors=3)
    model.block_size = 4
    model.uses_full_attn = False
    model.uses_sliding_window_attn = False

    with pytest.raises(ValueError, match="anchor_positions must match"):
        model._build_attention_mask(
            torch.ones(1, 16, dtype=torch.bool),
            torch.zeros(1, 16, dtype=torch.long),
            torch.device("cpu"),
            anchor_positions=torch.tensor([1, 2]),
        )


def test_recompute_sampled_qlogp_uses_anchor_index_not_position_division():
    model = DSparkDraftModel.__new__(DSparkDraftModel)
    model.markov_head = None

    num_blocks, block, vocab = 3, 4, 8
    hidden = torch.zeros(1, num_blocks * block, 2)
    logits_base = torch.full((1, num_blocks * block, vocab), -100.0)
    block_idx = 1
    sampled_draft_ids = torch.tensor([[2, 4, 6]])
    sampled_target_logprobs = torch.tensor([[-0.2, -0.4, -0.6]])
    for k, draft_id in enumerate(sampled_draft_ids[0], start=1):
        logits_base[0, block_idx * block + k, int(draft_id)] = float(10 + k)

    input_ids = torch.arange(32).unsqueeze(0)
    anchored_block_indices = torch.tensor(
        [
            3,
            4,
            5,
            6,
            17,
            18,
            19,
            20,
            0,
            1,
            2,
            3,
        ]
    )

    q_logp, p_logp = model._recompute_sampled_qlogp(
        hidden=hidden,
        logits_base=logits_base,
        input_ids=input_ids,
        anchored_block_indices=anchored_block_indices,
        sampled_draft_ids=sampled_draft_ids,
        sampled_target_ids=torch.tensor([[101, 102, 103]]),
        sampled_anchor_pos=torch.tensor([17]),
        sampled_anchor_index=torch.tensor([block_idx]),
        sampled_target_logprobs=sampled_target_logprobs,
        num_blocks=num_blocks,
        block=block,
    )

    assert q_logp.shape == (1, 3)
    assert torch.all(q_logp > -1e-3)
    assert torch.equal(p_logp, sampled_target_logprobs)


def test_recompute_sampled_qlogp_uses_anchor_index_when_anchor_zero_is_padded():
    model = DSparkDraftModel.__new__(DSparkDraftModel)
    model.markov_head = None

    num_blocks, block, vocab = 3, 4, 8
    hidden = torch.zeros(1, num_blocks * block, 2)
    logits_base = torch.full((1, num_blocks * block, vocab), -100.0)
    # Both block 0 and padded block 2 have anchor position 0. The stored ordinal
    # must select block 0; searching by position would be ambiguous.
    logits_base[0, 1, 3] = 10.0
    logits_base[0, 2, 4] = 10.0
    logits_base[0, 3, 5] = 10.0
    logits_base[0, 9, 3] = -10.0
    logits_base[0, 10, 4] = -10.0
    logits_base[0, 11, 5] = -10.0

    q_logp, _ = model._recompute_sampled_qlogp(
        hidden=hidden,
        logits_base=logits_base,
        input_ids=torch.arange(16).unsqueeze(0),
        anchored_block_indices=torch.tensor([0, 1, 2, 3, 6, 7, 8, 9, 0, 1, 2, 3]),
        sampled_draft_ids=torch.tensor([[3, 4, 5]]),
        sampled_target_ids=torch.tensor([[30, 31, 32]]),
        sampled_anchor_pos=torch.tensor([0]),
        sampled_anchor_index=torch.tensor([0]),
        sampled_target_logprobs=torch.zeros(1, 3),
        num_blocks=num_blocks,
        block=block,
    )

    assert torch.all(q_logp > -1e-3)


def test_recompute_sampled_qlogp_fails_fast_on_anchor_mismatch():
    model = DSparkDraftModel.__new__(DSparkDraftModel)
    model.markov_head = None

    with pytest.raises(RuntimeError, match="Sampled anchor position mismatch"):
        model._recompute_sampled_qlogp(
            hidden=torch.zeros(1, 8, 2),
            logits_base=torch.zeros(1, 8, 5),
            input_ids=torch.arange(16).unsqueeze(0),
            anchored_block_indices=torch.tensor([2, 3, 4, 5, 8, 9, 10, 11]),
            sampled_draft_ids=torch.tensor([[1, 2, 3]]),
            sampled_target_ids=torch.tensor([[10, 11, 12]]),
            sampled_anchor_pos=torch.tensor([7]),
            sampled_anchor_index=torch.tensor([1]),
            sampled_target_logprobs=torch.zeros(1, 3),
            num_blocks=2,
            block=4,
        )


def test_recompute_sampled_qlogp_builds_shifted_prev_history_once():
    markov = RecordingMarkovHead(draft_vocab_size=8)
    model = DSparkDraftModel.__new__(DSparkDraftModel)
    model.markov_head = markov

    q_logp, _ = model._recompute_sampled_qlogp(
        hidden=torch.zeros(1, 4, 2),
        logits_base=torch.zeros(1, 4, 8),
        input_ids=torch.tensor([[50, 51, 52, 53, 54, 55]]),
        anchored_block_indices=torch.tensor([2, 3, 4, 5]),
        sampled_draft_ids=torch.tensor([[1, 2, 3]]),
        sampled_target_ids=torch.tensor([[101, 102, 103]]),
        sampled_anchor_pos=torch.tensor([2]),
        sampled_anchor_index=torch.tensor([0]),
        sampled_target_logprobs=torch.zeros(1, 3),
        num_blocks=1,
        block=4,
    )

    assert q_logp.shape == (1, 3)
    assert len(markov.prev_token_calls) == 1
    assert torch.equal(markov.prev_token_calls[0], torch.tensor([[52, 52, 101, 102]]))


class FakeDraftModel:
    def __init__(self) -> None:
        self.block_size = 4
        self.config = SimpleNamespace(max_anchors=512)
        self.markov_head = RecordingMarkovHead(draft_vocab_size=6)
        self.d2t = torch.tensor([20, 20, 20, 20, 20, 20])
        self._param = torch.nn.Parameter(torch.zeros(()))
        self.backbone_anchor_positions: torch.Tensor | None = None
        self.backbone_anchor_valid: torch.Tensor | None = None

    def parameters(self):
        yield self._param

    def get_backbone_outputs(
        self,
        hidden_states,
        input_ids,
        loss_mask,
        verifier_last_hidden_states,
        document_ids,
        position_ids=None,
        *,
        anchor_positions,
        anchor_valid,
    ):
        self.backbone_anchor_positions = anchor_positions.detach().clone()
        self.backbone_anchor_valid = anchor_valid.detach().clone()
        logits = torch.zeros(1, self.block_size, 6)
        logits[0, 1, 1] = 10.0
        logits[0, 2, 2] = 10.0
        logits[0, 3, 3] = 10.0
        hidden = torch.zeros(1, self.block_size, 2)
        return hidden, logits, None, None, None


def test_sample_from_draft_uses_shifted_prev_history_and_explicit_anchor():
    model = FakeDraftModel()
    augmentor = SampledAcceptanceAugmentor.__new__(SampledAcceptanceAugmentor)
    augmentor.config = SampledAcceptanceConfig("http://unused", temperature=0.0)

    batch = {
        "input_ids": torch.tensor([[7, 8, 9, 10, 11, 12]]),
        "loss_mask": torch.tensor([[False, True, True, True, False, False]]),
        "hidden_states": torch.zeros(1, 6, 2),
        "verifier_last_hidden_states": torch.zeros(1, 6, 2),
        "document_ids": torch.zeros(1, 6, dtype=torch.long),
        "position_ids": torch.arange(6).unsqueeze(0),
    }

    sample = augmentor._sample_from_draft(model, batch, anchor_pos=1)

    assert sample["sampled_draft_token_ids"] == [1, 2, 3]
    assert sample["sampled_target_token_ids"] == [21, 22, 23]
    assert torch.equal(model.backbone_anchor_positions, torch.tensor([1]))
    assert torch.equal(model.backbone_anchor_valid, torch.tensor([True]))
    assert len(model.markov_head.prev_token_calls) == 3
    assert torch.equal(
        model.markov_head.prev_token_calls[0],
        torch.tensor([[8, 8, 8, 8]]),
    )
    assert torch.equal(
        model.markov_head.prev_token_calls[1],
        torch.tensor([[8, 8, 21, 8]]),
    )
    assert torch.equal(
        model.markov_head.prev_token_calls[2],
        torch.tensor([[8, 8, 21, 22]]),
    )


def test_sample_from_draft_scores_vllm_with_document_local_prefix():
    model = FakeDraftModel()
    augmentor = SampledAcceptanceAugmentor.__new__(SampledAcceptanceAugmentor)
    augmentor.config = SampledAcceptanceConfig("http://unused", temperature=0.0)

    batch = {
        "input_ids": torch.tensor([[101, 102, 103, 201, 202, 203, 204]]),
        "loss_mask": torch.tensor([[False, False, False, True, True, True, False]]),
        "hidden_states": torch.zeros(1, 7, 2),
        "verifier_last_hidden_states": torch.zeros(1, 7, 2),
        "document_ids": torch.tensor([[0, 0, 0, 1, 1, 1, 1]]),
        "position_ids": torch.tensor([[0, 1, 2, 0, 1, 2, 3]]),
    }

    sample = augmentor._sample_from_draft(model, batch, anchor_pos=4)

    assert sample["prefix_token_ids"] == [201, 202]


def test_sampled_acceptance_default_temperature_is_on_policy():
    assert SampledAcceptanceConfig("http://unused").temperature == 1.0


def test_target_token_id_uses_d2t_offset_convention():
    model = DSparkDraftModel.__new__(DSparkDraftModel)
    model.d2t = torch.zeros(128, dtype=torch.long)
    model.d2t[90] = 1805

    assert SampledAcceptanceAugmentor._target_token_id(model, 11) == 11
    assert SampledAcceptanceAugmentor._target_token_id(model, 90) == 1895


def test_augmentor_globally_skips_when_any_rank_has_no_anchor(monkeypatch):
    class FakeDist:
        class ReduceOp:
            MIN = "min"

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def all_reduce(tensor, op):
            tensor.zero_()

    model = DSparkDraftModel.__new__(DSparkDraftModel)
    model.block_size = 4
    model.config = SimpleNamespace(max_anchors=2)
    model._attn_impl = "eager"

    def fail_sample(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("sampling must be skipped on all ranks")

    augmentor = SampledAcceptanceAugmentor.__new__(SampledAcceptanceAugmentor)
    augmentor.config = SampledAcceptanceConfig("http://unused")
    augmentor.skipped_no_anchor = 0
    augmentor._sample_from_draft = fail_sample

    monkeypatch.setattr(sampled_acceptance, "dist", FakeDist)

    batch = {
        "loss_mask": torch.tensor([[True, True, True, True, False, False]]),
    }

    assert augmentor(model, batch) is batch
    assert "anchor_positions" not in batch
    assert "sampled_draft_ids" not in batch
    assert "sampled_target_logprobs" not in batch
    assert augmentor.skipped_no_anchor == 1
