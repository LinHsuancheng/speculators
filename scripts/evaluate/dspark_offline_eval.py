#!/usr/bin/env python3
"""Offline DSpark evaluation, adapted from DeepSeek/DeepSpec.

DeepSpec evaluates speculative decoding without vLLM by keeping the verifier KV
cache live and verifying each draft proposal incrementally. This script ports
that evaluator shape to the speculators DSpark model:

* target prefill runs once with ``DynamicCache``;
* each loop builds a DSpark draft proposal from the current anchor token;
* verifier checks ``[anchor, draft_tokens...]`` with the target cache;
* acceptance uses standard rejection sampling from target/draft probabilities.

The current speculators DSpark backbone does not expose DeepSpec's cached draft
``_forward_backbone`` API, so the draft proposal path uses a single-anchor
backbone forward. The evaluator structure is intentionally the same, so that can
be swapped for a cached draft path once the model API supports it.
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

logger = logging.getLogger("dspark_offline_eval")
torch = None
DynamicCache = None

PROMPT_FIELDS = (
    "prompt",
    "input",
    "question",
    "instruction",
    "text",
    "problem",
    "problem_statement",
    "question_content",
)
RESULT_COLUMNS = [
    "dataset",
    "num_requests",
    "elapsed_s",
    "requests_per_second",
    "output_tokens_per_second",
    "total_output_tokens",
    "num_proposals",
    "num_proposed_draft_tokens",
    "num_accepted_draft_tokens",
    "draft_length",
    "acceptance_length",
    "accepted_draft_length",
    "position_accept_rates",
    "position_accepted_counts",
    "position_proposed_counts",
]


@dataclass
class DraftProposal:
    draft_token_count: int
    verify_input_ids: Any
    draft_probs: Any | None


@dataclass
class VerificationResult:
    target_output: Any
    target_probs: Any
    accept_prefix_mask: Any | None
    accepted_draft_tokens: int
    next_token: Any
    effective_proposal_length: int
    terminated_by_stop_token: bool = False
    committed_tokens: Any | None = None


@dataclass
class EvalStats:
    elapsed_s: float = 0.0
    total_output_tokens: int = 0
    num_proposals: int = 0
    num_proposed_draft_tokens: int = 0
    num_accepted_draft_tokens: int = 0
    position_proposed_counts: list[int] = field(default_factory=list)
    position_accepted_counts: list[int] = field(default_factory=list)

    @property
    def acceptance_length(self) -> float:
        if self.num_proposals == 0:
            return 1.0
        return 1.0 + self.num_accepted_draft_tokens / self.num_proposals

    @property
    def draft_length(self) -> float:
        if self.num_proposals == 0:
            return 0.0
        return self.num_proposed_draft_tokens / self.num_proposals

    @property
    def accepted_draft_length(self) -> float:
        if self.num_proposals == 0:
            return 0.0
        return self.num_accepted_draft_tokens / self.num_proposals

    @property
    def position_accept_rates(self) -> list[float]:
        return [
            accepted / proposed if proposed else 0.0
            for accepted, proposed in zip(
                self.position_accepted_counts,
                self.position_proposed_counts,
                strict=True,
            )
        ]

    def add_response(self, response: SimpleNamespace) -> None:
        self.total_output_tokens += int(response.num_output_tokens)
        proposal_lengths = getattr(response, "proposal_lengths", [])
        accepted_lengths = getattr(response, "accepted_draft_lengths", [])
        self.num_proposals += len(proposal_lengths)
        self.num_proposed_draft_tokens += sum(int(x) for x in proposal_lengths)
        self.num_accepted_draft_tokens += sum(int(x) for x in accepted_lengths)
        for proposal_len, accepted_len in zip(
            proposal_lengths,
            accepted_lengths,
            strict=True,
        ):
            self.add_proposal_positions(int(proposal_len), int(accepted_len))

    def add_proposal_positions(self, proposal_len: int, accepted_len: int) -> None:
        if proposal_len < 0 or accepted_len < 0:
            raise ValueError("proposal_len and accepted_len must be non-negative")
        if accepted_len > proposal_len:
            raise ValueError(
                f"accepted_len must not exceed proposal_len, got "
                f"{accepted_len}>{proposal_len}"
            )
        missing = proposal_len - len(self.position_proposed_counts)
        if missing > 0:
            self.position_proposed_counts.extend([0] * missing)
            self.position_accepted_counts.extend([0] * missing)
        for pos in range(proposal_len):
            self.position_proposed_counts[pos] += 1
            if pos < accepted_len:
                self.position_accepted_counts[pos] += 1


def _format_position_accept_rates(stats: EvalStats) -> str:
    rates = stats.position_accept_rates
    if not rates:
        return "[]"
    return "[" + ", ".join(
        (
            f"pos{idx}={rate:.4f}"
            f"({stats.position_accepted_counts[idx]}/"
            f"{stats.position_proposed_counts[idx]})"
        )
        for idx, rate in enumerate(rates)
    ) + "]"


def _parse_count_list(value: Any) -> list[int]:
    if isinstance(value, str):
        if not value:
            return []
        value = json.loads(value)
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def logits_to_probs(logits, temperature: float):
    if temperature <= 0:
        return torch.nn.functional.one_hot(
            torch.argmax(logits, dim=-1),
            num_classes=logits.shape[-1],
        ).to(logits.dtype)
    return torch.softmax(logits.float() / temperature, dim=-1)


def sample_from_probs(probs):
    flat = probs.reshape(-1, probs.shape[-1])
    sampled = torch.multinomial(flat, num_samples=1)
    return sampled.reshape(*probs.shape[:-1])


def gather_token_probs(probs, token_ids):
    return torch.gather(probs, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def sample_residual(target_probs, draft_probs):
    residual = (target_probs - draft_probs).clamp_min(0)
    denom = residual.sum(dim=-1, keepdim=True)
    residual = torch.where(denom > 0, residual / denom.clamp_min(1e-8), target_probs)
    return sample_from_probs(residual)


def has_stop_token(token_ids, stop_token_ids: list[int] | None) -> bool:
    if stop_token_ids is None:
        return False
    stop_tensor = torch.tensor(stop_token_ids, device=token_ids.device)
    return bool(torch.isin(token_ids, stop_tensor).any().item())


def trim_output_ids(output_ids, num_input_tokens: int, stop_token_ids: list[int] | None):
    if stop_token_ids is None:
        return output_ids
    stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
    stop_indices = torch.isin(output_ids[0][num_input_tokens:], stop_tensor).nonzero(
        as_tuple=True,
    )[0]
    if stop_indices.numel() == 0:
        return output_ids
    return output_ids[:, : num_input_tokens + int(stop_indices[0].item()) + 1]


def resolve_stop_token_ids(target_model, tokenizer) -> list[int] | None:
    generation_config = getattr(target_model, "generation_config", None)
    eos_token_id = getattr(generation_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return None
    if isinstance(eos_token_id, int):
        return [int(eos_token_id)]
    return list(dict.fromkeys(int(token_id) for token_id in eos_token_id))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(item)
    return records


def _string_turns(value: Any) -> list[str] | None:
    if isinstance(value, str) and value.strip():
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        turns = [item for item in value if item.strip()]
        return turns or None
    return None


def _messages_from_conversations(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list):
        return None

    messages: list[dict[str, str]] = []
    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }
    for item in value:
        if not isinstance(item, dict):
            return None
        raw_role = item.get("from", item.get("role"))
        raw_content = item.get("value", item.get("content"))
        if not isinstance(raw_role, str) or not isinstance(raw_content, str):
            return None
        role = role_map.get(raw_role)
        content = raw_content.strip()
        if role is None or not content:
            return None
        if role == "assistant":
            break
        messages.append({"role": role, "content": content})

    return messages or None


def _prompt_from_record(record: dict[str, Any], tokenizer, *, source: str) -> str:
    turns = _string_turns(record.get("turns"))
    if turns is not None:
        # DeepSpec keeps only the first turn for eval. Keep all turns joined here
        # because existing local data may already be single-response prompts.
        return "\n\n".join(turns)

    messages = record.get("messages")
    if isinstance(messages, list):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    messages = _messages_from_conversations(record.get("conversations"))
    if messages is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    for field in PROMPT_FIELDS:
        turns = _string_turns(record.get(field))
        if turns is not None:
            return "\n\n".join(turns)

    keys = ", ".join(sorted(record.keys()))
    supported = ", ".join(["turns", "messages", "conversations", *PROMPT_FIELDS])
    raise ValueError(
        f"{source}: record has no supported prompt field ({supported}); keys=[{keys}]"
    )


def _discover_datasets(root: Path, names: list[str] | None) -> list[Path]:
    paths = [root] if root.is_file() else sorted(root.rglob("*.jsonl"))
    if names:
        wanted = set(names)
        paths = [
            path
            for path in paths
            if path.stem in wanted or path.name in wanted or str(path) in wanted
        ]
    if not paths:
        raise FileNotFoundError(f"No JSONL datasets found under {root}")
    return paths


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _shard_records(
    records: list[dict[str, Any]],
    *,
    shard_index: int | None,
    num_shards: int,
) -> list[tuple[int, dict[str, Any]]]:
    indexed_records = list(enumerate(records, start=1))
    if shard_index is None or num_shards <= 1:
        return indexed_records
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}), got {shard_index}"
        )
    return [
        item
        for zero_based_index, item in enumerate(indexed_records)
        if zero_based_index % num_shards == shard_index
    ]


def _shard_label(args: argparse.Namespace) -> str:
    shard_index = getattr(args, "worker_shard_index", None)
    if shard_index is None:
        return ""
    return f"/shard{shard_index}"


def _format_device_memory() -> str:
    if torch is None:
        return "unavailable"

    backend = None
    if hasattr(torch, "npu"):
        backend = torch.npu
    elif hasattr(torch, "cuda") and torch.cuda.is_available():
        backend = torch.cuda
    if backend is None:
        return "unavailable"

    try:
        allocated = backend.memory_allocated()
        reserved = backend.memory_reserved()
        max_allocated = backend.max_memory_allocated()
    except Exception as exc:  # pragma: no cover - backend-specific diagnostics
        return f"unavailable:{exc}"

    gib = 1024**3
    return (
        f"allocated={allocated / gib:.2f}GiB "
        f"reserved={reserved / gib:.2f}GiB "
        f"max_allocated={max_allocated / gib:.2f}GiB"
    )


def _aggregate_rows(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    position_proposed_counts: list[int] = []
    position_accepted_counts: list[int] = []
    for row in rows:
        proposed = _parse_count_list(row.get("position_proposed_counts", []))
        accepted = _parse_count_list(row.get("position_accepted_counts", []))
        size = max(len(position_proposed_counts), len(proposed))
        if len(position_proposed_counts) < size:
            position_proposed_counts.extend([0] * (size - len(position_proposed_counts)))
            position_accepted_counts.extend([0] * (size - len(position_accepted_counts)))
        for idx, count in enumerate(proposed):
            position_proposed_counts[idx] += count
        for idx, count in enumerate(accepted):
            position_accepted_counts[idx] += count

    stats = EvalStats(
        elapsed_s=max((float(row["elapsed_s"]) for row in rows), default=0.0),
        total_output_tokens=sum(int(row["total_output_tokens"]) for row in rows),
        num_proposals=sum(int(row["num_proposals"]) for row in rows),
        num_proposed_draft_tokens=sum(
            int(row["num_proposed_draft_tokens"]) for row in rows
        ),
        num_accepted_draft_tokens=sum(
            int(row["num_accepted_draft_tokens"]) for row in rows
        ),
        position_proposed_counts=position_proposed_counts,
        position_accepted_counts=position_accepted_counts,
    )
    num_requests = sum(int(row["num_requests"]) for row in rows)
    return {
        "dataset": dataset,
        "num_requests": num_requests,
        "elapsed_s": stats.elapsed_s,
        "requests_per_second": num_requests / stats.elapsed_s if stats.elapsed_s else 0,
        "output_tokens_per_second": (
            stats.total_output_tokens / stats.elapsed_s if stats.elapsed_s else 0
        ),
        "total_output_tokens": stats.total_output_tokens,
        "num_proposals": stats.num_proposals,
        "num_proposed_draft_tokens": stats.num_proposed_draft_tokens,
        "num_accepted_draft_tokens": stats.num_accepted_draft_tokens,
        "draft_length": stats.draft_length,
        "acceptance_length": stats.acceptance_length,
        "accepted_draft_length": stats.accepted_draft_length,
        "position_accept_rates": json.dumps(stats.position_accept_rates),
        "position_accepted_counts": json.dumps(stats.position_accepted_counts),
        "position_proposed_counts": json.dumps(stats.position_proposed_counts),
    }


def _draft_ids_to_target_ids(draft, draft_ids: list[int]) -> list[int]:
    if draft.use_draft_vocab and draft.d2t is not None:
        d2t = draft.d2t
        return [int(token_id + d2t[token_id].item()) for token_id in draft_ids]
    return draft_ids


def _draft_sample_from_anchor(draft) -> bool:
    return bool(getattr(getattr(draft, "config", None), "sample_from_anchor", False))


def _load_vocab_mapping_tensors(
    *,
    draft_model_path: str,
    d2t_path: Path | None,
    t2d_path: Path | None,
):
    if d2t_path is None and t2d_path is None:
        draft_path = Path(draft_model_path)
        d2t_path = draft_path / "d2t.npy"
        t2d_path = draft_path / "t2d.npy"
        if not d2t_path.exists() and not t2d_path.exists():
            return None, None
    elif d2t_path is None or t2d_path is None:
        raise ValueError("--d2t-path and --t2d-path must be provided together.")

    if d2t_path is None or t2d_path is None:
        return None, None
    if not d2t_path.exists():
        raise FileNotFoundError(f"d2t mapping file not found: {d2t_path}")
    if not t2d_path.exists():
        raise FileNotFoundError(f"t2d mapping file not found: {t2d_path}")

    import numpy as np  # noqa: PLC0415

    logger.info("Loading vocab mappings: d2t=%s t2d=%s", d2t_path, t2d_path)
    return torch.from_numpy(np.load(d2t_path)), torch.from_numpy(np.load(t2d_path))


def _ensure_loaded_vocab_mappings(draft_model, args: argparse.Namespace) -> None:
    if not draft_model.use_draft_vocab:
        return
    if (
        draft_model.t2d is not None
        and int(draft_model.t2d.sum(dtype=torch.long).item())
        == int(draft_model.draft_vocab_size)
    ):
        return

    d2t, t2d = _load_vocab_mapping_tensors(
        draft_model_path=args.draft_model,
        d2t_path=args.d2t_path,
        t2d_path=args.t2d_path,
    )
    if d2t is None or t2d is None:
        raise ValueError(
            "DSpark draft uses a pruned draft vocab, but no real d2t/t2d mapping "
            "was loaded. Pass --d2t-path and --t2d-path, or place d2t.npy and "
            "t2d.npy under --draft-model."
        )
    draft_model.load_vocab_mappings(t2d, d2t)


def verify_draft_tokens(
    *,
    target_model,
    proposal: DraftProposal,
    position_ids,
    start: int,
    past_key_values_target,
    temperature: float,
    max_proposal_tokens: int,
    current_token_ids=None,
    stop_token_ids: list[int] | None = None,
) -> VerificationResult:
    if proposal.draft_token_count > max_proposal_tokens:
        raise ValueError(
            "DraftProposal.draft_token_count must not exceed "
            f"max_proposal_tokens={max_proposal_tokens}, "
            f"got {proposal.draft_token_count}."
        )
    if current_token_ids is not None and not torch.equal(
        proposal.verify_input_ids[:, :1],
        current_token_ids,
    ):
        raise ValueError("DraftProposal.verify_input_ids must start with current token.")

    draft_token_count = int(proposal.draft_token_count)
    verify_length = draft_token_count + 1
    target_output = target_model(
        input_ids=proposal.verify_input_ids,
        position_ids=position_ids[:, start : start + verify_length],
        past_key_values=past_key_values_target,
        use_cache=True,
        output_hidden_states=True,
    )
    target_probs = logits_to_probs(target_output.logits, float(temperature))

    accept_prefix_mask = None
    if draft_token_count > 0:
        assert proposal.draft_probs is not None
        proposed_tokens = proposal.verify_input_ids[:, 1:]
        selected_target_probs = gather_token_probs(target_probs[:, :-1, :], proposed_tokens)
        selected_draft_probs = gather_token_probs(
            proposal.draft_probs,
            proposed_tokens,
        ).clamp_min(1e-8)
        accept_prob = torch.clamp(selected_target_probs / selected_draft_probs, max=1.0)
        accept_mask = (torch.rand_like(accept_prob) < accept_prob).to(torch.int64)
        accept_prefix_mask = accept_mask.cumprod(dim=1)
        accepted_draft_tokens = int(accept_prefix_mask.sum(dim=1)[0].item())
    else:
        accepted_draft_tokens = 0

    effective_proposal_length = draft_token_count
    terminated_by_stop_token = False
    if stop_token_ids and accepted_draft_tokens > 0:
        accepted_slice = proposal.verify_input_ids[0, 1 : accepted_draft_tokens + 1]
        stop_tensor = torch.tensor(
            stop_token_ids,
            device=accepted_slice.device,
            dtype=accepted_slice.dtype,
        )
        eos_hits = torch.isin(accepted_slice, stop_tensor).nonzero(as_tuple=True)[0]
        if eos_hits.numel() > 0:
            accepted_draft_tokens = int(eos_hits[0].item()) + 1
            effective_proposal_length = accepted_draft_tokens
            terminated_by_stop_token = True

    if 0 < draft_token_count and accepted_draft_tokens < draft_token_count:
        assert proposal.draft_probs is not None
        next_token = sample_residual(
            target_probs[:, accepted_draft_tokens, :],
            proposal.draft_probs[:, accepted_draft_tokens, :],
        )
    else:
        next_token = sample_from_probs(target_probs[:, -1:, :]).squeeze(1)

    committed_tokens = torch.cat(
        [
            proposal.verify_input_ids[:, 1 : accepted_draft_tokens + 1],
            next_token.unsqueeze(1),
        ],
        dim=1,
    )
    return VerificationResult(
        target_output=target_output,
        target_probs=target_probs,
        accept_prefix_mask=accept_prefix_mask,
        accepted_draft_tokens=accepted_draft_tokens,
        next_token=next_token,
        effective_proposal_length=effective_proposal_length,
        terminated_by_stop_token=terminated_by_stop_token,
        committed_tokens=committed_tokens,
    )


def generate_decoding_sample(
    *,
    target_model,
    input_ids,
    max_new_tokens: int,
    max_proposal_tokens: int,
    temperature: float,
    stop_token_ids: list[int] | None,
    init_context: Callable[..., Any],
    propose: Callable[..., DraftProposal],
    update: Callable[[Any, VerificationResult], None],
) -> SimpleNamespace:
    assert max_proposal_tokens >= 1
    assert input_ids.size(0) == 1, "only bsz=1 is supported"

    device = input_ids.device
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + int(max_new_tokens)
    output_ids = torch.empty(
        (1, max_length + max_proposal_tokens + 1),
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    past_key_values_target = DynamicCache()

    output = target_model(
        input_ids=input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample_from_probs(
        logits_to_probs(output.logits[:, -1:, :], float(temperature))
    )
    start = num_input_tokens
    acceptance_lengths: list[int] = []
    proposal_lengths: list[int] = []
    accepted_draft_lengths: list[int] = []
    initial_token = output_ids[:, num_input_tokens : num_input_tokens + 1]
    if has_stop_token(initial_token, stop_token_ids):
        output_ids = output_ids[:, : num_input_tokens + 1]
        output_ids = trim_output_ids(output_ids, num_input_tokens, stop_token_ids)
        return SimpleNamespace(
            output_ids=output_ids,
            num_input_tokens=num_input_tokens,
            num_output_tokens=output_ids.shape[1] - num_input_tokens,
            acceptance_lengths=acceptance_lengths,
            proposal_lengths=proposal_lengths,
            accepted_draft_lengths=accepted_draft_lengths,
            verify_count=0,
        )

    context = init_context(
        initial_output=output,
        output_ids=output_ids,
        position_ids=position_ids,
        num_input_tokens=num_input_tokens,
    )
    del output

    while start < max_length:
        proposal = propose(
            context=context,
            output_ids=output_ids,
            position_ids=position_ids,
            start=start,
            stop_token_ids=stop_token_ids,
        )
        verification = verify_draft_tokens(
            target_model=target_model,
            proposal=proposal,
            position_ids=position_ids,
            start=start,
            past_key_values_target=past_key_values_target,
            temperature=temperature,
            max_proposal_tokens=max_proposal_tokens,
            current_token_ids=output_ids[:, start : start + 1],
            stop_token_ids=stop_token_ids,
        )

        proposal_lengths.append(int(verification.effective_proposal_length))
        accepted = int(verification.accepted_draft_tokens)
        accepted_draft_lengths.append(accepted)
        output_ids[:, start : start + accepted + 1] = (
            proposal.verify_input_ids[:, : accepted + 1]
        )

        if verification.terminated_by_stop_token:
            acceptance_lengths.append(accepted)
            start += accepted
            past_key_values_target.crop(start)
            break

        output_ids[:, start + accepted + 1] = verification.next_token
        new_token_ids = output_ids[:, start + 1 : start + accepted + 2]
        acceptance_lengths.append(accepted + 1)
        start += accepted + 1
        past_key_values_target.crop(start)
        update(context, verification)
        del proposal, verification

        if has_stop_token(new_token_ids, stop_token_ids):
            break

    output_ids = output_ids[:, : min(start + 1, max_length)]
    output_ids = trim_output_ids(output_ids, num_input_tokens, stop_token_ids)
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=output_ids.shape[1] - num_input_tokens,
        acceptance_lengths=acceptance_lengths,
        proposal_lengths=proposal_lengths,
        accepted_draft_lengths=accepted_draft_lengths,
        verify_count=len(proposal_lengths),
    )


class DSparkOfflineRunner:
    def __init__(self, target_model, draft_model, tokenizer, args) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.args = args
        self.device = next(target_model.parameters()).device
        self.sample_from_anchor = _draft_sample_from_anchor(draft_model)
        speculative_slots = (
            int(draft_model.block_size)
            if self.sample_from_anchor
            else int(draft_model.block_size) - 1
        )
        self.max_proposal_tokens = max(1, speculative_slots)

    def _extract_context_feature(self, hidden_states):
        return torch.cat(
            [hidden_states[i] for i in self.draft_model.target_layer_ids],
            dim=-1,
        )

    def _init_context(self, *, initial_output, **kwargs) -> SimpleNamespace:
        return SimpleNamespace(
            target_hidden_states=self._extract_context_feature(
                initial_output.hidden_states,
            ),
        )

    def _single_anchor_backbone(self, hidden_states, input_ids, start: int):
        draft = self.draft_model
        block = draft.block_size
        if hidden_states.shape[1] != start:
            raise ValueError(
                "DSpark context hidden states must contain exactly the prefix before "
                f"the current anchor; got {hidden_states.shape[1]} and start={start}."
            )
        # The synthetic block carries the current anchor token. DFlash masks base
        # tokens with position >= anchor, so this dummy anchor hidden state is
        # present only to preserve the training-time base-sequence layout.
        hidden_states = torch.cat(
            [hidden_states, hidden_states.new_zeros(hidden_states[:, :1, :].shape)],
            dim=1,
        )
        total_seq_len = hidden_states.shape[1]
        current_ids = input_ids[:, :total_seq_len]
        anchor_positions = torch.tensor([start], dtype=torch.long, device=self.device)
        document_ids = torch.zeros_like(current_ids)

        full_attn_mask = None
        if draft.uses_full_attn:
            full_attn_mask = draft._create_attention_masks_for_layers(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=self.device,
                sliding_window=None,
            )

        sliding_window_attn_mask = None
        if draft.uses_sliding_window_attn:
            sliding_window_attn_mask = draft._create_attention_masks_for_layers(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=self.device,
                sliding_window=draft.sliding_window,
                sliding_window_non_causal=draft.sliding_window_non_causal,
            )

        mask_token_ids = torch.full(
            (1, block),
            draft.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        mask_token_ids[:, 0] = input_ids[:, start]
        noise_embedding = draft.embed_tokens(mask_token_ids)
        fc_output = draft.hidden_norm(draft.fc(hidden_states))
        base_position_ids = torch.arange(
            total_seq_len,
            dtype=torch.long,
            device=self.device,
        )
        block_offsets = torch.arange(block, dtype=torch.long, device=self.device)
        position_ids = torch.cat(
            [base_position_ids, base_position_ids[start] + block_offsets],
            dim=0,
        ).unsqueeze(0)
        position_embeddings = draft.rotary_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(draft.layers):
            if layer_idx in draft.sliding_window_indices:
                attention_mask = sliding_window_attn_mask[layer_idx]
            else:
                attention_mask = full_attn_mask[layer_idx]
            noise_embedding = layer(
                hidden_states=noise_embedding,
                target_hidden=fc_output,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
            )

        hidden = draft.norm(noise_embedding)
        return hidden, draft.lm_head(hidden)

    def _sample_dspark_tokens(self, base_logits, hidden_states, first_prev_token_id):
        draft = self.draft_model
        max_tokens = self.max_proposal_tokens
        proposed_target_ids: list[int] = []
        draft_probs = []
        prev_token = first_prev_token_id.reshape(1, 1).long()
        first_slot = 0 if self.sample_from_anchor else 1

        for token_idx in range(max_tokens):
            slot = first_slot + token_idx
            logits = base_logits[:, slot : slot + 1, :]
            if draft.markov_head is not None:
                logits = logits + draft.markov_head.block_bias(
                    prev_token_ids=prev_token,
                    hidden_states=hidden_states[:, slot : slot + 1, :],
                )
            probs = logits_to_probs(logits, float(self.args.temperature))
            draft_id = int(sample_from_probs(probs)[0, 0].item())
            target_id = _draft_ids_to_target_ids(draft, [draft_id])[0]
            proposed_target_ids.append(target_id)
            draft_probs.append(probs)
            prev_token = torch.tensor([[target_id]], dtype=torch.long, device=self.device)

        return proposed_target_ids, torch.cat(draft_probs, dim=1)

    def _expand_draft_probs_to_target_vocab(self, draft_probs):
        draft = self.draft_model
        if not draft.use_draft_vocab or draft.d2t is None:
            return draft_probs
        if draft.t2d is not None:
            target_vocab_size = int(draft.t2d.shape[0])
        else:
            target_vocab_size = int(draft.verifier_vocab_size)
        expanded = draft_probs.new_zeros(*draft_probs.shape[:-1], target_vocab_size)
        draft_ids = torch.arange(
            draft_probs.shape[-1],
            device=draft_probs.device,
            dtype=draft.d2t.dtype,
        )
        target_ids = (draft_ids + draft.d2t.to(draft_probs.device)).long()
        expanded.index_copy_(-1, target_ids, draft_probs)
        return expanded

    def _propose(
        self,
        *,
        context: SimpleNamespace,
        output_ids,
        position_ids,
        start: int,
        stop_token_ids: list[int] | None = None,
    ) -> DraftProposal:
        hidden, base_logits = self._single_anchor_backbone(
            context.target_hidden_states,
            output_ids,
            start,
        )
        proposed_target_ids, draft_probs = self._sample_dspark_tokens(
            base_logits,
            hidden,
            output_ids[:, start],
        )
        verify_input_ids = torch.cat(
            [
                output_ids[:, start : start + 1],
                torch.tensor(
                    [proposed_target_ids],
                    dtype=torch.long,
                    device=self.device,
                ),
            ],
            dim=1,
        )
        return DraftProposal(
            draft_token_count=len(proposed_target_ids),
            verify_input_ids=verify_input_ids,
            draft_probs=self._expand_draft_probs_to_target_vocab(draft_probs),
        )

    def _update(self, context: SimpleNamespace, verification: VerificationResult) -> None:
        hidden = self._extract_context_feature(verification.target_output.hidden_states)
        committed_hidden = hidden[:, : verification.accepted_draft_tokens + 1, :]
        context.target_hidden_states = torch.cat(
            [context.target_hidden_states, committed_hidden],
            dim=1,
        )

    def generate_one(self, prompt: str, stop_token_ids: list[int] | None):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.inference_mode():
            return generate_decoding_sample(
                target_model=self.target_model,
                input_ids=input_ids,
                max_new_tokens=int(self.args.max_new_tokens),
                max_proposal_tokens=self.max_proposal_tokens,
                temperature=float(self.args.temperature),
                stop_token_ids=stop_token_ids,
                init_context=self._init_context,
                propose=self._propose,
                update=self._update,
            )


def _evaluate_dataset(
    *,
    path: Path,
    runner: DSparkOfflineRunner,
    args: argparse.Namespace,
    stop_token_ids: list[int] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = _load_jsonl(path)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    indexed_records = _shard_records(
        records,
        shard_index=getattr(args, "worker_shard_index", None),
        num_shards=getattr(args, "worker_num_shards", 1),
    )

    stats = EvalStats()
    artifacts: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    iterator = indexed_records
    if tqdm is not None and not args.no_progress:
        iterator = tqdm(
            iterator,
            total=len(indexed_records),
            desc=path.stem,
            unit="sample",
        )

    for processed, (idx, record) in enumerate(iterator, start=1):
        prompt = _prompt_from_record(
            record,
            runner.tokenizer,
            source=f"{path}:{idx}",
        )
        response = runner.generate_one(prompt, stop_token_ids)
        stats.add_response(response)
        if not args.skip_artifacts:
            artifacts.append(
                {
                    "prompt": prompt,
                    "output_token_ids": response.output_ids[0].tolist(),
                    "num_input_tokens": int(response.num_input_tokens),
                    "source_index": idx,
                }
            )

        if (
            processed == 1
            or processed % args.log_every == 0
            or processed == len(indexed_records)
        ):
            elapsed = time.perf_counter() - start_time
            out_tps = stats.total_output_tokens / elapsed if elapsed else 0.0
            logger.info(
                (
                    "[%s%s] %d/%d samples | out_tok=%d | tok/s=%.2f | "
                    "acc_len=%.3f | draft_len=%.3f | accepted_draft_len=%.3f | "
                    "pos_accept=%s | mem=%s"
                ),
                path.stem,
                _shard_label(args),
                processed,
                len(indexed_records),
                stats.total_output_tokens,
                out_tps,
                stats.acceptance_length,
                stats.draft_length,
                stats.accepted_draft_length,
                _format_position_accept_rates(stats),
                _format_device_memory(),
            )
        del response

    stats.elapsed_s = time.perf_counter() - start_time
    row = {
        "dataset": path.stem,
        "num_requests": len(indexed_records),
        "elapsed_s": stats.elapsed_s,
        "requests_per_second": (
            len(indexed_records) / stats.elapsed_s if stats.elapsed_s else 0
        ),
        "output_tokens_per_second": (
            stats.total_output_tokens / stats.elapsed_s if stats.elapsed_s else 0
        ),
        "total_output_tokens": stats.total_output_tokens,
        "num_proposals": stats.num_proposals,
        "num_proposed_draft_tokens": stats.num_proposed_draft_tokens,
        "num_accepted_draft_tokens": stats.num_accepted_draft_tokens,
        "draft_length": stats.draft_length,
        "acceptance_length": stats.acceptance_length,
        "accepted_draft_length": stats.accepted_draft_length,
        "position_accept_rates": json.dumps(stats.position_accept_rates),
        "position_accepted_counts": json.dumps(stats.position_accepted_counts),
        "position_proposed_counts": json.dumps(stats.position_proposed_counts),
    }
    return row, artifacts


def _write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    artifacts_by_dataset: dict[str, list[dict[str, Any]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    if not artifacts_by_dataset:
        return
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    for dataset, artifacts in artifacts_by_dataset.items():
        with (artifacts_dir / f"{dataset}.jsonl").open("w", encoding="utf-8") as f:
            for artifact in artifacts:
                f.write(json.dumps(artifact) + "\n")


def _read_worker_row(output_dir: Path) -> dict[str, Any]:
    with (output_dir / "summary.json").open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(f"{output_dir}/summary.json must contain one result row")
    return rows[0]


def _read_worker_artifacts(output_dir: Path, dataset: str) -> list[dict[str, Any]]:
    path = output_dir / "artifacts" / f"{dataset}.jsonl"
    if not path.exists():
        return []
    return _load_jsonl(path)


def _worker_command(
    args: argparse.Namespace,
    *,
    dataset_path: Path,
    shard_index: int,
    num_shards: int,
    output_dir: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--verifier-model",
        args.verifier_model,
        "--draft-model",
        args.draft_model,
        "--datasets-root",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--draft-attn-impl",
        args.draft_attn_impl,
        "--log-every",
        str(args.log_every),
        "--worker-shard-index",
        str(shard_index),
        "--worker-num-shards",
        str(num_shards),
        "--no-progress",
    ]
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.d2t_path is not None:
        cmd.extend(["--d2t-path", str(args.d2t_path)])
    if args.t2d_path is not None:
        cmd.extend(["--t2d-path", str(args.t2d_path)])
    if args.skip_artifacts:
        cmd.append("--skip-artifacts")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def run_ascend_data_parallel(args: argparse.Namespace) -> None:
    devices = _split_csv(args.ascend_devices)
    if not devices:
        raise ValueError("--ascend-devices must contain at least one device id")

    dataset_names = _split_csv(args.datasets)
    dataset_paths = _discover_datasets(args.datasets_root, dataset_names or None)
    rows: list[dict[str, Any]] = []
    artifacts_by_dataset: dict[str, list[dict[str, Any]]] = {}

    for dataset_path in dataset_paths:
        dataset_start = time.perf_counter()
        shard_root = args.output_dir / "_shards" / dataset_path.stem
        logger.info(
            "[%s] loading dataset | shards=%d | ASCEND devices=%s",
            dataset_path.stem,
            len(devices),
            ",".join(devices),
        )

        processes = []
        for shard_index, visible_device in enumerate(devices):
            shard_output_dir = shard_root / f"shard_{shard_index}"
            cmd = _worker_command(
                args,
                dataset_path=dataset_path,
                shard_index=shard_index,
                num_shards=len(devices),
                output_dir=shard_output_dir,
            )
            env = os.environ.copy()
            env["ASCEND_RT_VISIBLE_DEVICES"] = visible_device
            logger.info(
                "[%s/shard%d] start | ASCEND_RT_VISIBLE_DEVICES=%s",
                dataset_path.stem,
                shard_index,
                visible_device,
            )
            processes.append((shard_index, shard_output_dir, subprocess.Popen(cmd, env=env)))

        failed = []
        for shard_index, _, process in processes:
            returncode = process.wait()
            if returncode != 0:
                failed.append((shard_index, returncode))
        if failed:
            raise RuntimeError(f"{dataset_path.stem} worker failures: {failed}")

        shard_rows = [
            _read_worker_row(shard_output_dir) for _, shard_output_dir, _ in processes
        ]
        row = _aggregate_rows(dataset_path.stem, shard_rows)
        row["elapsed_s"] = time.perf_counter() - dataset_start
        row["requests_per_second"] = (
            row["num_requests"] / row["elapsed_s"] if row["elapsed_s"] else 0
        )
        row["output_tokens_per_second"] = (
            row["total_output_tokens"] / row["elapsed_s"] if row["elapsed_s"] else 0
        )
        rows.append(row)

        if not args.skip_artifacts:
            artifacts = []
            for _, shard_output_dir, _ in processes:
                artifacts.extend(_read_worker_artifacts(shard_output_dir, dataset_path.stem))
            artifacts.sort(key=lambda item: int(item.get("source_index", 0)))
            artifacts_by_dataset[dataset_path.stem] = artifacts

        _write_outputs(args.output_dir, rows, artifacts_by_dataset)
        logger.info(
            (
                "[%s] result | requests=%d | tok/s=%.2f | acc_len=%.4f | "
                "draft_len=%.4f | accepted_draft_len=%.4f | pos_accept=%s"
            ),
            dataset_path.stem,
            row["num_requests"],
            row["output_tokens_per_second"],
            row["acceptance_length"],
            row["draft_length"],
            row["accepted_draft_length"],
            _format_position_accept_rates(
                EvalStats(
                    position_accepted_counts=_parse_count_list(
                        row.get("position_accepted_counts", []),
                    ),
                    position_proposed_counts=_parse_count_list(
                        row.get("position_proposed_counts", []),
                    ),
                ),
            ),
        )

    logger.info("Wrote merged results to %s", args.output_dir)


def _resolve_draft_attn_impl(device: str, draft_attn_impl: str) -> str | None:
    if draft_attn_impl != "auto":
        return draft_attn_impl
    if str(device).startswith("npu"):
        return "sdpa"
    return None


def run(args: argparse.Namespace) -> None:
    global torch, DynamicCache

    if (
        getattr(args, "ascend_devices", None)
        and getattr(args, "worker_shard_index", None) is None
    ):
        run_ascend_data_parallel(args)
        return

    import torch as torch_module  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
    from transformers import DynamicCache as DynamicCacheClass  # noqa: PLC0415

    from speculators.models.dspark.core import DSparkDraftModel  # noqa: PLC0415

    torch = torch_module
    DynamicCache = DynamicCacheClass
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"

    logger.info("Loading tokenizer: %s", args.verifier_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.verifier_model,
        trust_remote_code=args.trust_remote_code,
    )
    logger.info("Loading verifier: %s", args.verifier_model)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.verifier_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()

    logger.info("Loading DSpark draft: %s", args.draft_model)
    draft_config = DSparkDraftModel.config_class.from_pretrained(args.draft_model)
    if args.sample_from_anchor:
        draft_config.sample_from_anchor = True
    draft_attn_impl = _resolve_draft_attn_impl(args.device, args.draft_attn_impl)
    if draft_attn_impl is not None:
        logger.info("Using draft attention backend: %s", draft_attn_impl)
        draft_config.transformer_layer_config._attn_implementation = draft_attn_impl
    d2t, t2d = _load_vocab_mapping_tensors(
        draft_model_path=args.draft_model,
        d2t_path=args.d2t_path,
        t2d_path=args.t2d_path,
    )
    draft_model = DSparkDraftModel.from_pretrained(
        args.draft_model,
        config=draft_config,
        d2t=d2t,
        t2d=t2d,
    ).to(device).eval()
    _ensure_loaded_vocab_mappings(draft_model, args)
    logger.info(
        (
            "Loaded models | device=%s | ASCEND_RT_VISIBLE_DEVICES=%s | "
            "use_draft_vocab=%s | d2t_loaded=%s | mem=%s"
        ),
        device,
        os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
        draft_model.use_draft_vocab,
        draft_model.t2d is not None and bool(draft_model.t2d.any().item()),
        _format_device_memory(),
    )

    runner = DSparkOfflineRunner(target_model, draft_model, tokenizer, args)
    stop_token_ids = resolve_stop_token_ids(target_model, tokenizer)
    dataset_names = args.datasets.split(",") if args.datasets else None
    dataset_paths = _discover_datasets(args.datasets_root, dataset_names)
    rows: list[dict[str, Any]] = []
    artifacts_by_dataset: dict[str, list[dict[str, Any]]] = {}

    for path in dataset_paths:
        row, artifacts = _evaluate_dataset(
            path=path,
            runner=runner,
            args=args,
            stop_token_ids=stop_token_ids,
        )
        rows.append(row)
        if not args.skip_artifacts:
            artifacts_by_dataset[path.stem] = artifacts
        logger.info(
            (
                "[%s%s] done | requests=%d | tok/s=%.2f | "
                "acc_len=%.4f | draft_len=%.4f | accepted_draft_len=%.4f | "
                "pos_accept=%s | mem=%s"
            ),
            path.stem,
            _shard_label(args),
            row["num_requests"],
            row["output_tokens_per_second"],
            row["acceptance_length"],
            row["draft_length"],
            row["accepted_draft_length"],
            _format_position_accept_rates(
                EvalStats(
                    position_accepted_counts=_parse_count_list(
                        row.get("position_accepted_counts", []),
                    ),
                    position_proposed_counts=_parse_count_list(
                        row.get("position_proposed_counts", []),
                    ),
                ),
            ),
            _format_device_memory(),
        )

    _write_outputs(args.output_dir, rows, artifacts_by_dataset)
    logger.info("Wrote results to %s", args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline DSpark/speculators evaluation on JSONL data.",
    )
    parser.add_argument("--verifier-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("dspark_offline_eval"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--draft-attn-impl",
        choices=["auto", "simple_flex_attention", "sdpa", "eager"],
        default="auto",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--d2t-path", type=Path, default=None)
    parser.add_argument("--t2d-path", type=Path, default=None)
    parser.add_argument(
        "--sample-from-anchor",
        action="store_true",
        help="Enable PR 806 DSpark Markov previous-token alignment.",
    )
    parser.add_argument(
        "--ascend-devices",
        default=None,
        help=(
            "Comma-separated physical Ascend device ids for dataset-sharded "
            "parallel evaluation, for example 8,9,10,11,12,13,14,15."
        ),
    )
    parser.add_argument("--worker-shard-index", type=int, default=None)
    parser.add_argument("--worker-num-shards", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
