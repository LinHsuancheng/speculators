import torch
from torch.nn.attention.flex_attention import (
    or_masks,
)


def _validate_anchor_mask_inputs(
    document_ids: torch.Tensor,
    total_seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    if document_ids.dim() != 1:
        raise ValueError(f"document_ids must be 1-D, got shape {document_ids.shape}")
    if document_ids.numel() != total_seq_len:
        raise ValueError(
            "document_ids length must match total_seq_len, got "
            f"{document_ids.numel()} and {total_seq_len}"
        )
    if anchor_positions.dim() != 1:
        raise ValueError(
            f"anchor_positions must be 1-D, got shape {anchor_positions.shape}"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    anchor_positions = anchor_positions.to(
        device=document_ids.device, dtype=torch.long
    ).contiguous()
    if anchor_positions.numel() and (
        bool((anchor_positions < 0).any())
        or bool((anchor_positions >= total_seq_len).any())
    ):
        raise ValueError(
            "anchor_positions must be within [0, total_seq_len), got "
            f"min={int(anchor_positions.min())}, max={int(anchor_positions.max())}, "
            f"total_seq_len={total_seq_len}"
        )
    return anchor_positions


def create_anchor_block_mask_mod(
    document_ids: torch.Tensor,
    total_seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
    sliding_window: int | None = None,
    sliding_window_non_causal: bool = False,
):
    """
    Build a flex-attention mask mod where each query block corresponds to one anchor.

    Q side:
        n_anchors * block_size synthetic query tokens
        block j corresponds to anchor_positions[j]

    KV side:
        [ original packed sequence | synthetic anchor blocks ]

    For queries in block j:
        - may attend to base tokens in the same document with
          position < anchor_positions[j]
        - may attend to all tokens in their own synthetic block j
        - may not attend to other synthetic blocks or later base tokens

    Args:
        document_ids: [total_seq_len] maps each position to its doc index, pad -1
        total_seq_len: padded packed sequence width
        anchor_positions: [n_anchors] absolute positions into the packed base sequence
        block_size: number of query tokens per anchor block
        sliding_window: integer size of sliding window or None for full attn
        sliding_window_non_causal: Use non causal mask for sliding window attn

    Returns:
        mask_mod, q_len, kv_len
    """
    # Always use non_causal for full attn
    non_causal = sliding_window is None or sliding_window_non_causal

    anchor_positions = _validate_anchor_mask_inputs(
        document_ids=document_ids,
        total_seq_len=total_seq_len,
        anchor_positions=anchor_positions,
        block_size=block_size,
    )

    n_anchors = anchor_positions.numel()
    q_len = n_anchors * block_size
    kv_len = total_seq_len + q_len

    # For each query position, which anchor does it belong to?
    # query q in [j*block_size, (j+1)*block_size) belongs to anchor_positions[j]
    query_anchor_positions = torch.repeat_interleave(anchor_positions, block_size)

    def base_prefix_mod(_b, _h, q_idx, kv_idx):
        """
        Queries may see base-sequence tokens in the same document before the anchor.
        """
        # absolute base position
        q_anchor = query_anchor_positions[q_idx]
        # doc id for this query block
        q_doc = document_ids[q_anchor]

        kv_is_base = kv_idx < total_seq_len
        kv_base_pos = torch.remainder(kv_idx, total_seq_len)  # safe indexing
        kv_doc = document_ids[kv_base_pos]

        same_doc = (q_doc == kv_doc) & (q_doc != -1)
        before_anchor = kv_base_pos < q_anchor

        in_window = (
            (kv_base_pos >= q_anchor - sliding_window)
            if sliding_window is not None
            else True
        )

        return kv_is_base & same_doc & before_anchor & in_window

    def same_block_mod(_b, _h, q_idx, kv_idx):
        """
        Queries may attend to tokens in their own synthetic block.
        Non-causal unless non_causal=False,
        in which case only prior positions are attended.
        """
        q_block = q_idx // block_size
        kv_is_block = kv_idx >= total_seq_len
        kv_block = (kv_idx - total_seq_len) // block_size

        same = kv_is_block & (q_block == kv_block)
        if not non_causal:
            same = same & (kv_idx <= q_idx + total_seq_len)
        return same

    return or_masks(base_prefix_mod, same_block_mod), q_len, kv_len


def create_anchor_micro_block_causal_mask_mod(
    document_ids: torch.Tensor,
    total_seq_len: int,
    anchor_positions: torch.Tensor,
    block_size: int,
    micro_block_size: int,
    anchor_len: int = 1,
    sliding_window: int | None = None,
    max_prev_micro_blocks: int | None = None,
    max_prev_micro_tokens: int | None = None,
):
    """
    Build a DFlash flex-attention mask with pseudo-autoregressive micro blocks.

    KV layout:
        [ original packed sequence | synthetic anchor blocks ]

    Each synthetic block layout:
        [ anchor token(s) | speculative token positions ]

    Within a synthetic block, anchor token(s) are visible to speculative tokens.
    Speculative tokens can attend bidirectionally inside their own micro block,
    and can attend to earlier micro blocks, but not later micro blocks.
    """
    anchor_positions = _validate_anchor_mask_inputs(
        document_ids=document_ids,
        total_seq_len=total_seq_len,
        anchor_positions=anchor_positions,
        block_size=block_size,
    )
    if micro_block_size <= 0:
        raise ValueError(
            f"micro_block_size must be positive, got {micro_block_size}"
        )
    if anchor_len < 0 or anchor_len >= block_size:
        raise ValueError(
            f"anchor_len must be in [0, block_size), got "
            f"anchor_len={anchor_len}, block_size={block_size}"
        )

    spec_len = block_size - anchor_len
    if spec_len % micro_block_size != 0:
        raise ValueError(
            "block_size - anchor_len must be divisible by micro_block_size, "
            f"got block_size={block_size}, anchor_len={anchor_len}, "
            f"micro_block_size={micro_block_size}"
        )
    if max_prev_micro_blocks is not None and max_prev_micro_blocks < 0:
        raise ValueError(
            "max_prev_micro_blocks must be non-negative when set, got "
            f"{max_prev_micro_blocks}"
        )
    if max_prev_micro_tokens is not None and max_prev_micro_tokens < 0:
        raise ValueError(
            "max_prev_micro_tokens must be non-negative when set, got "
            f"{max_prev_micro_tokens}"
        )

    n_anchors = anchor_positions.numel()
    q_len = n_anchors * block_size
    kv_len = total_seq_len + q_len

    query_anchor_positions = torch.repeat_interleave(anchor_positions, block_size)
    query_document_ids = document_ids[query_anchor_positions]

    def base_prefix_mod(_b, _h, q_idx, kv_idx):
        """
        Synthetic query tokens may see original-sequence tokens before the anchor.
        """
        kv_is_base = kv_idx < total_seq_len

        q_anchor_pos = query_anchor_positions[q_idx]
        q_doc_id = query_document_ids[q_idx]

        safe_kv_idx = kv_idx.clamp(max=total_seq_len - 1)
        kv_doc_id = document_ids[safe_kv_idx]

        same_doc = (kv_doc_id == q_doc_id) & (q_doc_id != -1)
        before_anchor = kv_idx < q_anchor_pos

        in_window = (
            (kv_idx >= q_anchor_pos - sliding_window)
            if sliding_window is not None
            else True
        )

        return kv_is_base & same_doc & before_anchor & in_window

    def micro_block_causal_mod(_b, _h, q_idx, kv_idx):
        """
        Synthetic-block visibility: block-local pseudo autoregression.
        """
        kv_is_block = kv_idx >= total_seq_len

        q_block = q_idx // block_size
        kv_block = (kv_idx - total_seq_len) // block_size
        same_block = kv_is_block & (q_block == kv_block)

        q_offset = q_idx % block_size
        kv_offset = (kv_idx - total_seq_len) % block_size

        q_is_anchor = q_offset < anchor_len
        kv_is_anchor = kv_offset < anchor_len

        anchor_query_visible = q_is_anchor & kv_is_anchor
        spec_query_anchor_visible = (~q_is_anchor) & kv_is_anchor

        q_micro = (q_offset - anchor_len) // micro_block_size
        kv_micro = (kv_offset - anchor_len) // micro_block_size
        same_micro_visible = kv_micro == q_micro
        if max_prev_micro_tokens is not None:
            q_micro_offset = (q_offset - anchor_len) % micro_block_size
            kv_micro_offset = (kv_offset - anchor_len) % micro_block_size
            same_micro_visible = (
                same_micro_visible
                & (kv_micro_offset <= q_micro_offset)
                & (q_micro_offset - kv_micro_offset <= max_prev_micro_tokens)
            )
        prev_micro_visible = kv_micro < q_micro
        if max_prev_micro_blocks is not None:
            prev_micro_visible = prev_micro_visible & (
                q_micro - kv_micro <= max_prev_micro_blocks
            )
        spec_to_spec_visible = (
            (~q_is_anchor)
            & (~kv_is_anchor)
            & (same_micro_visible | prev_micro_visible)
        )

        return same_block & (
            anchor_query_visible
            | spec_query_anchor_visible
            | spec_to_spec_visible
        )

    return or_masks(base_prefix_mod, micro_block_causal_mod), q_len, kv_len
