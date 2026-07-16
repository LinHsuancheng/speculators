"""On-policy verifier scoring for the exact expected-acceptance-length objective.

The exact acceptance-length loss (see :mod:`speculators.models.dspark.metrics`)
is on-policy: the draft samples a block ``Y_{1:K} ~ q_psi`` and the *frozen*
target model must score those sampled tokens **at the sampled prefix** to give
``p_t(Y_t | x, Y_{<t})``. The gold verifier hidden states cached on disk are of
no use here because the sampled continuation diverges from the gold one, so this
scoring is the only genuinely online, non-cacheable step of training.

A :class:`VerifierScorer` returns, per anchored block, the verifier hidden
states aligned so that row ``t`` is the hidden state that *predicts* the token
filling draft slot ``t+1`` (i.e. the hidden state at sequence position
``anchor + t``). The caller applies the (frozen, draft-vocab-sliced)
``verifier_lm_head`` locally to turn those hidden states into ``p_t``.

Two implementations:

* :class:`VLLMVerifierScorer` -- production. Sends ``gold[:anchor+1] + Y`` to the
  existing vLLM hidden-states endpoint (one prefill, ``max_tokens=1``) and reads
  the sampled-position hidden states back. Relies on prefix caching so the shared
  gold prefix KV is computed once (see the ``--enable-prefix-caching`` flag added
  to the launch scripts).
* :class:`MockVerifierScorer` -- tests / dry runs. Scores locally from a supplied
  frozen verifier callable (or returns a fixed tensor), so the whole on-policy
  path can be unit-tested without a live vLLM server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable
import warnings

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "MockVerifierScorer",
    "VLLMVerifierScorer",
    "VerifierScorer",
    "build_scored_sequences",
]


@runtime_checkable
class VerifierScorer(Protocol):
    """Returns frozen-verifier hidden states at the sampled draft positions.

    The returned tensor is ``[num_blocks, K, hidden]`` where ``K = block_size - 1``
    and row ``t`` (0-indexed) is the verifier hidden state at packed-sequence
    position ``anchor + t`` when the block's continuation is the sampled
    ``Y_{1:K}``. Equivalently, row ``t`` predicts the token filling draft slot
    ``t + 1``. Slot 0 (the anchor) is always emitted and is not scored.
    """

    def score(
        self,
        *,
        gold_input_ids: torch.Tensor,  # [1, total_seq_len] verifier vocab
        document_ids: torch.Tensor,  # [1, total_seq_len]
        anchor_positions: torch.Tensor,  # [num_blocks] packed-seq index of slot 0
        sampled_verifier_ids: torch.Tensor,  # [num_blocks, K] verifier vocab
    ) -> torch.Tensor:  # [num_blocks, K, hidden]
        ...


def build_scored_sequences(
    gold_input_ids: torch.Tensor,  # [1, total_seq_len] verifier vocab
    document_ids: torch.Tensor,  # [1, total_seq_len]
    anchor_positions: torch.Tensor,  # [num_blocks]
    sampled_verifier_ids: torch.Tensor,  # [num_blocks, K]
) -> list[list[int]]:
    """Build one verifier-vocab token sequence per block: ``gold_prefix + Y``.

    The prefix for block ``b`` is the gold tokens of ``b``'s document from the
    document start up to and including the anchor at ``anchor_positions[b]``,
    followed by the sampled continuation ``sampled_verifier_ids[b]``. Restricting
    to the document (via ``document_ids``) keeps packed, multi-document sequences
    from leaking cross-document context into the verifier — the same boundary the
    draft attention respects.

    Returns a list of ``num_blocks`` python int lists (vLLM prompt payloads). The
    verifier distribution that predicts draft slot ``t`` (1-indexed) is read from
    the output position ``prefix_len - 1 + (t - 1)``; see
    :meth:`VLLMVerifierScorer.score` for how those rows are gathered.
    """
    ids = gold_input_ids[0].tolist()
    docs = document_ids[0].tolist()
    anchors = anchor_positions.tolist()
    sampled = sampled_verifier_ids.tolist()

    sequences: list[list[int]] = []
    for anchor, cont in zip(anchors, sampled, strict=True):
        doc = docs[anchor]
        # Walk back to the first token of this document.
        start = anchor
        while start > 0 and docs[start - 1] == doc:
            start -= 1
        prefix = ids[start : anchor + 1]
        sequences.append(prefix + list(cont))
    return sequences


class VLLMVerifierScorer:
    """Score sampled sequences via the vLLM hidden-states endpoint.

    One :meth:`score` call issues a batched ``max_tokens=1`` prefill for the
    ``num_blocks`` sequences ``gold_prefix + Y`` and reads back the hidden states
    at the sampled positions. The verifier does **no** decoding — the sampled
    tokens are part of the prompt — so this is pure scoring.

    Args:
        client: An OpenAI-compatible client pointed at the vLLM server.
        model: Model id served by vLLM.
        load_hidden_states: Callable mapping a returned hidden-states path to a
            ``[seq_len, num_target_layers * hidden]`` (or last-layer
            ``[seq_len, hidden]``) tensor. Injected so this module stays free of
            the on-disk safetensors/lock details owned by the data pipeline.
        hidden_size: Verifier hidden size, used to slice the last-layer block from
            a concatenated multi-layer hidden-states row.
        request_timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        *,
        client,  # noqa: ANN001 (openai.Client, kept loose to avoid a hard dep here)
        model: str,
        load_hidden_states: Callable[[str], torch.Tensor],
        hidden_size: int,
        request_timeout: float | None = 120.0,
    ) -> None:
        self.client = client
        self.model = model
        self.load_hidden_states = load_hidden_states
        self.hidden_size = hidden_size
        self.request_timeout = request_timeout

    @torch.no_grad()
    def score(
        self,
        *,
        gold_input_ids: torch.Tensor,
        document_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        sampled_verifier_ids: torch.Tensor,
    ) -> torch.Tensor:
        from speculators.data_generation.vllm_client import (  # noqa: PLC0415
            generate_hidden_states,
        )

        num_blocks, k = sampled_verifier_ids.shape
        device = sampled_verifier_ids.device

        sequences = build_scored_sequences(
            gold_input_ids, document_ids, anchor_positions, sampled_verifier_ids
        )

        out = torch.zeros(num_blocks, k, self.hidden_size, device=device)
        valid_mask = torch.ones(num_blocks, dtype=torch.bool, device=device)

        for b, seq in enumerate(sequences):
            prefix_len = len(seq) - k
            path = generate_hidden_states(
                self.client,
                self.model,
                {"input_ids": seq},
                timeout=self.request_timeout,
            )
            hs = self.load_hidden_states(path).to(device)  # [seq_len, H or L*H]
            # Keep only the last-layer block if a multi-layer row was returned.
            if hs.shape[-1] != self.hidden_size:
                hs = hs[:, -self.hidden_size :]
            # Row predicting draft slot t (1-indexed) sits at prefix_len-1+(t-1).
            first = prefix_len - 1
            actual_len = hs.shape[0]
            if actual_len < first + k:
                warnings.warn(
                    f"Sample {b}: vLLM returned incomplete hidden states ({actual_len} tokens) "
                    f"for sequence with {len(seq)} tokens. Needed positions [{first}:{first+k}]. "
                    f"Skipping this sample (prefix caching interference)."
                )
                valid_mask[b] = False
            else:
                out[b] = hs[first : first + k]
        return out, valid_mask


class MockVerifierScorer:
    """Local scorer for tests / dry runs — no vLLM server required.

    Either wraps a frozen ``verifier`` callable ``ids[list[int]] -> [seq_len, H]``
    hidden states, or (when ``verifier`` is None) returns a fixed hidden-states
    tensor supplied at construction. Alignment matches :class:`VLLMVerifierScorer`.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        verifier: Callable[[list[int]], torch.Tensor] | None = None,
        fixed_hidden: torch.Tensor | None = None,
    ) -> None:
        if verifier is None and fixed_hidden is None:
            raise ValueError("Provide either `verifier` or `fixed_hidden`.")
        self.hidden_size = hidden_size
        self.verifier = verifier
        self.fixed_hidden = fixed_hidden

    @torch.no_grad()
    def score(
        self,
        *,
        gold_input_ids: torch.Tensor,
        document_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        sampled_verifier_ids: torch.Tensor,
    ) -> torch.Tensor:
        num_blocks, k = sampled_verifier_ids.shape
        device = sampled_verifier_ids.device
        if self.fixed_hidden is not None:
            return self.fixed_hidden.to(device).expand(num_blocks, k, self.hidden_size)

        sequences = build_scored_sequences(
            gold_input_ids, document_ids, anchor_positions, sampled_verifier_ids
        )
        out = torch.zeros(num_blocks, k, self.hidden_size, device=device)
        for b, seq in enumerate(sequences):
            prefix_len = len(seq) - k
            hs = self.verifier(seq).to(device)  # type: ignore[misc]  # [seq_len, H]
            first = prefix_len - 1
            out[b] = hs[first : first + k]
        return out
