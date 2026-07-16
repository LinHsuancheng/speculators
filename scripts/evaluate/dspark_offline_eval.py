#!/usr/bin/env python3
"""Offline DSpark evaluation on DeepSpec-style JSONL datasets.

This follows DeepSpec's offline evaluation shape instead of vLLM serving:
load the verifier and DSpark draft with PyTorch, run speculative decoding over
JSONL prompts, and aggregate throughput plus acceptance-length metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("dspark_offline_eval")
torch = None

PROMPT_FIELDS = ("prompt", "input", "question", "instruction", "text")
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
    "acceptance_length",
    "accepted_draft_length",
]


@dataclass
class EvalStats:
    elapsed_s: float = 0.0
    total_output_tokens: int = 0
    num_proposals: int = 0
    num_proposed_draft_tokens: int = 0
    num_accepted_draft_tokens: int = 0

    @property
    def acceptance_length(self) -> float:
        if self.num_proposals == 0:
            return 1.0
        return 1.0 + self.num_accepted_draft_tokens / self.num_proposals

    @property
    def accepted_draft_length(self) -> float:
        if self.num_proposals == 0:
            return 0.0
        return self.num_accepted_draft_tokens / self.num_proposals


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


def _prompt_from_record(record: dict[str, Any], tokenizer) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    for field in PROMPT_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value

    raise ValueError(
        "record has no supported prompt field "
        f"({', '.join(['messages', *PROMPT_FIELDS])})"
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


def _target_hidden_states(
    target,
    input_ids,
    target_layer_ids: list[int],
) -> tuple[Any, Any]:
    with torch.no_grad():
        out = target(
            input_ids=input_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden = torch.cat([out.hidden_states[i] for i in target_layer_ids], dim=-1)
    return hidden, out.hidden_states[-1]


def _draft_logits(
    draft,
    target,
    input_ids,
) -> Any:
    """Return DSpark logits for one anchored block at the sequence tail."""
    block = draft.block_size
    device = input_ids.device
    hidden_states, verifier_last_hidden_states = _target_hidden_states(
        target,
        input_ids,
        draft.target_layer_ids,
    )
    loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    loss_mask[:, -block:] = 1.0
    document_ids = torch.zeros_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    hidden, logits, _, _, _ = draft._backbone_forward(
        hidden_states,
        input_ids,
        loss_mask,
        verifier_last_hidden_states,
        document_ids,
        position_ids,
    )

    block_tokens = input_ids[:, -block:].view(1, block)
    prev_token_ids = torch.cat([block_tokens[:, :1], block_tokens[:, :-1]], dim=1)
    hidden_blocks = hidden.view(1, block, -1)
    if draft.markov_head is not None:
        markov_bias = draft.markov_head.block_bias(
            prev_token_ids=prev_token_ids,
            hidden_states=hidden_blocks,
        )
        logits = (logits.view(1, block, -1) + markov_bias).view(1, block, -1)
    return logits


def _verify_acceptance(
    target,
    prefix_ids,
    draft_ids: list[int],
) -> int:
    if not draft_ids:
        return 0
    candidate = torch.cat(
        [
            prefix_ids,
            torch.tensor([draft_ids], device=prefix_ids.device, dtype=prefix_ids.dtype),
        ],
        dim=1,
    )
    with torch.no_grad():
        logits = target(candidate, use_cache=False).logits
    # Position `prefix_len - 1 + i` predicts draft token `i`.
    start = prefix_ids.shape[1] - 1
    verifier_tokens = torch.argmax(
        logits[:, start : start + len(draft_ids), :],
        dim=-1,
    )[0]
    accepted = 0
    for expected, proposed in zip(verifier_tokens.tolist(), draft_ids):
        if expected != proposed:
            break
        accepted += 1
    return accepted


def _draft_ids_to_target_ids(draft, draft_ids: list[int]) -> list[int]:
    if draft.use_draft_vocab and draft.d2t is not None:
        d2t = draft.d2t
        return [int(d2t[token_id].item()) for token_id in draft_ids]
    return draft_ids


def _generate_one(
    *,
    target,
    draft,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    eos_token_id: int | None,
) -> tuple[list[int], EvalStats]:
    device = next(target.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated: list[int] = []
    stats = EvalStats()

    while len(generated) < max_new_tokens:
        if input_ids.shape[1] < draft.block_size:
            with torch.no_grad():
                next_token = torch.argmax(target(input_ids).logits[:, -1, :], dim=-1)
            accepted = 0
            proposed: list[int] = []
        else:
            logits = _draft_logits(draft, target, input_ids)
            proposed_draft_ids = torch.argmax(logits[:, 1:, :], dim=-1)[0].tolist()
            proposed = _draft_ids_to_target_ids(draft, proposed_draft_ids)
            remaining = max_new_tokens - len(generated)
            proposed = proposed[:remaining]
            stats.num_proposals += 1
            stats.num_proposed_draft_tokens += len(proposed)
            accepted = _verify_acceptance(target, input_ids, proposed)
            stats.num_accepted_draft_tokens += accepted

            if accepted < len(proposed):
                candidate = torch.cat(
                    [
                        input_ids,
                        torch.tensor(
                            [proposed[: accepted + 1]],
                            device=device,
                            dtype=input_ids.dtype,
                        ),
                    ],
                    dim=1,
                )
                with torch.no_grad():
                    verifier_logits = target(candidate, use_cache=False).logits
                next_token = torch.argmax(
                    verifier_logits[:, input_ids.shape[1] + accepted - 1, :],
                    dim=-1,
                )
            else:
                accepted_ids = proposed[:accepted]
                if len(generated) + len(accepted_ids) >= max_new_tokens:
                    append_ids = accepted_ids[: max_new_tokens - len(generated)]
                    input_ids = torch.cat(
                        [
                            input_ids,
                            torch.tensor(
                                [append_ids], device=device, dtype=input_ids.dtype
                            ),
                        ],
                        dim=1,
                    )
                    generated.extend(append_ids)
                    break
                candidate = torch.cat(
                    [
                        input_ids,
                        torch.tensor(
                            [accepted_ids],
                            device=device,
                            dtype=input_ids.dtype,
                        ),
                    ],
                    dim=1,
                )
                with torch.no_grad():
                    verifier_logits = target(candidate, use_cache=False).logits
                next_token = torch.argmax(
                    verifier_logits[:, candidate.shape[1] - 1, :],
                    dim=-1,
                )

        append_ids = proposed[:accepted] if accepted else []
        append_ids.append(int(next_token.item()))
        append_ids = append_ids[: max_new_tokens - len(generated)]
        input_ids = torch.cat(
            [
                input_ids,
                torch.tensor([append_ids], device=device, dtype=input_ids.dtype),
            ],
            dim=1,
        )
        generated.extend(append_ids)
        if eos_token_id is not None and eos_token_id in append_ids:
            break

    stats.total_output_tokens = len(generated)
    return generated, stats


def _evaluate_dataset(
    *,
    path: Path,
    target,
    draft,
    tokenizer,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = _load_jsonl(path)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    stats = EvalStats()
    artifacts: list[dict[str, Any]] = []
    start = time.perf_counter()
    for record in records:
        prompt = _prompt_from_record(record, tokenizer)
        token_ids, sample_stats = _generate_one(
            target=target,
            draft=draft,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        stats.total_output_tokens += sample_stats.total_output_tokens
        stats.num_proposals += sample_stats.num_proposals
        stats.num_proposed_draft_tokens += sample_stats.num_proposed_draft_tokens
        stats.num_accepted_draft_tokens += sample_stats.num_accepted_draft_tokens
        artifacts.append({"prompt": prompt, "output_token_ids": token_ids})
    stats.elapsed_s = time.perf_counter() - start

    row = {
        "dataset": path.stem,
        "num_requests": len(records),
        "elapsed_s": stats.elapsed_s,
        "requests_per_second": len(records) / stats.elapsed_s if stats.elapsed_s else 0,
        "output_tokens_per_second": (
            stats.total_output_tokens / stats.elapsed_s if stats.elapsed_s else 0
        ),
        "total_output_tokens": stats.total_output_tokens,
        "num_proposals": stats.num_proposals,
        "num_proposed_draft_tokens": stats.num_proposed_draft_tokens,
        "num_accepted_draft_tokens": stats.num_accepted_draft_tokens,
        "acceptance_length": stats.acceptance_length,
        "accepted_draft_length": stats.accepted_draft_length,
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

    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    for dataset, artifacts in artifacts_by_dataset.items():
        with (artifacts_dir / f"{dataset}.jsonl").open("w", encoding="utf-8") as f:
            for artifact in artifacts:
                f.write(json.dumps(artifact) + "\n")


def run(args: argparse.Namespace) -> None:
    global torch

    import torch as torch_module  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    from speculators.models.dspark.core import DSparkDraftModel  # noqa: PLC0415

    torch = torch_module
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    tokenizer = AutoTokenizer.from_pretrained(
        args.verifier_model,
        trust_remote_code=args.trust_remote_code,
    )
    target = AutoModelForCausalLM.from_pretrained(
        args.verifier_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    draft = DSparkDraftModel.from_pretrained(args.draft_model).to(device)
    target.eval()
    draft.eval()

    dataset_names = args.datasets.split(",") if args.datasets else None
    dataset_paths = _discover_datasets(args.datasets_root, dataset_names)
    rows: list[dict[str, Any]] = []
    artifacts_by_dataset: dict[str, list[dict[str, Any]]] = {}

    for path in dataset_paths:
        logger.info("[%s] evaluating", path.stem)
        row, artifacts = _evaluate_dataset(
            path=path,
            target=target,
            draft=draft,
            tokenizer=tokenizer,
            args=args,
        )
        rows.append(row)
        artifacts_by_dataset[path.stem] = artifacts
        logger.info(
            "[%s] output_tps=%.2f acceptance_length=%.4f",
            path.stem,
            row["output_tokens_per_second"],
            row["acceptance_length"],
        )

    _write_outputs(args.output_dir, rows, artifacts_by_dataset)
    logger.info("Wrote results to %s", args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline DSpark/speculators evaluation on DeepSpec JSONL data.",
    )
    parser.add_argument("--verifier-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("dspark_offline_eval"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
