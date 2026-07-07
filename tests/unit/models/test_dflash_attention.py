"""Unit tests for the DFlash anchor-block attention mask."""

import torch
from torch.nn.attention.flex_attention import create_mask

from speculators.models.dflash.attention import (
    create_anchor_block_mask_mod,
    create_anchor_micro_block_causal_mask_mod,
)


def _lengths_to_document_ids(lengths, total_seq_len):
    document_ids = torch.full((total_seq_len,), -1, dtype=torch.long)
    document_ids[: lengths.sum()] = torch.repeat_interleave(
        torch.arange(lengths.shape[0], dtype=torch.long), lengths
    )
    return document_ids


def _reference_dense_from_mask_mod(
    document_ids,
    total_seq_len,
    anchor_positions,
    block_size,
    sliding_window=None,
    sliding_window_non_causal=False,
):
    """Ground truth: evaluate the flex mask_mod element-wise over the q x kv grid."""
    mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
        document_ids=document_ids,
        total_seq_len=total_seq_len,
        anchor_positions=anchor_positions,
        block_size=block_size,
        sliding_window=sliding_window,
        sliding_window_non_causal=sliding_window_non_causal,
    )
    zero = torch.zeros((), dtype=torch.long)
    ref = torch.zeros(q_len, kv_len, dtype=torch.bool)
    for q in range(q_len):
        for kv in range(kv_len):
            ref[q, kv] = bool(mask_mod(zero, zero, torch.tensor(q), torch.tensor(kv)))
    return ref


def _dense_from_create_mask(
    document_ids,
    total_seq_len,
    anchor_positions,
    block_size,
    sliding_window=None,
    sliding_window_non_causal=False,
):
    mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
        document_ids=document_ids,
        total_seq_len=total_seq_len,
        anchor_positions=anchor_positions,
        block_size=block_size,
        sliding_window=sliding_window,
        sliding_window_non_causal=sliding_window_non_causal,
    )
    return create_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=document_ids.device,
    )


def test_create_mask_matches_mask_mod_full_attention():
    """Dense mask equals the mask_mod for the full-attention case."""
    device = torch.device("cpu")
    total_seq_len, block_size = 16, 4
    lengths = torch.tensor([10, 6])  # two packed documents summing to total_seq_len
    document_ids = _lengths_to_document_ids(lengths, total_seq_len)
    anchor_positions = torch.tensor([3, 8, 12])

    ref = _reference_dense_from_mask_mod(
        document_ids, total_seq_len, anchor_positions, block_size
    )
    dense = _dense_from_create_mask(
        document_ids.to(device), total_seq_len, anchor_positions, block_size
    )

    assert dense.shape == (1, 1, ref.shape[0], ref.shape[1])
    assert torch.equal(dense[0, 0].bool(), ref)


def test_create_mask_matches_mask_mod_sliding_window():
    """Dense mask equals the mask_mod when a sliding window is set."""
    device = torch.device("cpu")
    total_seq_len, block_size = 16, 4
    lengths = torch.tensor([16])  # single document
    document_ids = _lengths_to_document_ids(lengths, total_seq_len)
    anchor_positions = torch.tensor([5, 9, 14])
    sliding_window = 4

    ref = _reference_dense_from_mask_mod(
        document_ids,
        total_seq_len,
        anchor_positions,
        block_size,
        sliding_window=sliding_window,
    )
    dense = _dense_from_create_mask(
        document_ids.to(device),
        total_seq_len,
        anchor_positions,
        block_size,
        sliding_window=sliding_window,
    )

    assert torch.equal(dense[0, 0].bool(), ref)


def test_create_mask_each_query_sees_its_own_block():
    """Every query must attend to at least its own synthetic block."""
    device = torch.device("cpu")
    total_seq_len, block_size = 12, 4
    lengths = torch.tensor([12])
    document_ids = _lengths_to_document_ids(lengths, total_seq_len)
    anchor_positions = torch.tensor([2, 7, 10])

    dense = _dense_from_create_mask(
        document_ids.to(device), total_seq_len, anchor_positions, block_size
    )

    assert bool(dense[0, 0].any(dim=-1).all())


def test_micro_block_mask_is_bidirectional_within_micro_blocks():
    """Micro-block mask is causal across chunks and bidirectional within chunks."""
    total_seq_len, block_size = 8, 5
    anchor_len, micro_block_size = 1, 2
    document_ids = _lengths_to_document_ids(torch.tensor([8]), total_seq_len)
    anchor_positions = torch.tensor([4])

    mask_mod, q_len, kv_len = create_anchor_micro_block_causal_mask_mod(
        document_ids=document_ids,
        total_seq_len=total_seq_len,
        anchor_positions=anchor_positions,
        block_size=block_size,
        micro_block_size=micro_block_size,
        anchor_len=anchor_len,
    )
    dense = create_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=document_ids.device,
    )[0, 0].bool()

    block_start = total_seq_len

    # Anchor can see only the synthetic anchor position inside its own block.
    assert bool(dense[0, block_start])
    assert not bool(dense[0, block_start + 1])

    # First speculative micro block sees the anchor and both tokens in micro 0.
    assert torch.equal(
        dense[1, block_start : block_start + block_size],
        torch.tensor([True, True, True, False, False]),
    )
    assert torch.equal(
        dense[2, block_start : block_start + block_size],
        torch.tensor([True, True, True, False, False]),
    )

    # Second speculative micro block additionally sees all earlier spec tokens.
    assert torch.equal(
        dense[3, block_start : block_start + block_size],
        torch.tensor([True, True, True, True, True]),
    )
    assert torch.equal(
        dense[4, block_start : block_start + block_size],
        torch.tensor([True, True, True, True, True]),
    )


def test_micro_block_mask_limits_previous_micro_blocks():
    """Layer-growth masks can restrict how many previous micro blocks are visible."""
    total_seq_len, block_size = 8, 7
    anchor_len, micro_block_size = 1, 2
    document_ids = _lengths_to_document_ids(torch.tensor([8]), total_seq_len)
    anchor_positions = torch.tensor([4])

    mask_mod, q_len, kv_len = create_anchor_micro_block_causal_mask_mod(
        document_ids=document_ids,
        total_seq_len=total_seq_len,
        anchor_positions=anchor_positions,
        block_size=block_size,
        micro_block_size=micro_block_size,
        anchor_len=anchor_len,
        max_prev_micro_blocks=1,
    )
    dense = create_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=q_len,
        KV_LEN=kv_len,
        device=document_ids.device,
    )[0, 0].bool()

    block_start = total_seq_len

    # Query in m2 can see anchor, m1, and m2, but not m0 when the window is 1.
    assert torch.equal(
        dense[5, block_start : block_start + block_size],
        torch.tensor([True, False, False, True, True, True, True]),
    )
    assert torch.equal(
        dense[6, block_start : block_start + block_size],
        torch.tensor([True, False, False, True, True, True, True]),
    )
