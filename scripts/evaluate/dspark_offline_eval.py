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
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger("dspark_offline_eval")
torch = None

PROMPT_FIELDS = (
    "prompt",
    "input",
    "question",
    "instruction",
    "text",
    # Raw fields used by DeepSpec's converter before it writes {"turns": ...}.
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


def _string_turns(value: Any) -> list[str] | None:
    if isinstance(value, str) and value.strip():
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        turns = [item for item in value if item.strip()]
        return turns or None
    return None


def _prompt_from_record(record: dict[str, Any], tokenizer, *, source: str) -> str:
    turns = _string_turns(record.get("turns"))
    if turns is not None:
        # DeepSpec's eval_datasets/*.jsonl are normalized as {"turns": [...]}. Most
        # datasets have one turn; multi-turn sets are kept deterministic here by
        # joining the turns into one prompt for this single-response throughput run.
        return "\n\n".join(turns)

    messages = record.get("messages")
    if isinstance(messages, list):
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
    raise ValueError(
        f"{source}: record has no supported prompt field "
        f"({', '.join(['turns', 'messages', *PROMPT_FIELDS])}); keys=[{keys}]"
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
    anchor_pos = input_ids.shape[1] - 1
    # DFlash/DSpark training uses anchored blocks inside a fixed sequence and
    # `select_anchors` deliberately excludes the final `block_size` positions.
    # For generation, append dummy tokens after the current prefix so the last
    # real token can be selected as the single valid anchor.
    dummy_ids = torch.full(
        (1, block),
        draft.mask_token_id,
        dtype=input_ids.dtype,
        device=device,
    )
    draft_input_ids = torch.cat([input_ids, dummy_ids], dim=1)

    hidden_states, verifier_last_hidden_states = _target_hidden_states(
        target,
        draft_input_ids,
        draft.target_layer_ids,
    )
    loss_mask = torch.zeros_like(draft_input_ids, dtype=torch.float32)
    loss_mask[:, anchor_pos] = 1.0
    document_ids = torch.zeros_like(draft_input_ids)
    position_ids = torch.arange(draft_input_ids.shape[1], device=device).unsqueeze(0)
    hidden, logits, _, _, _ = draft._backbone_forward(
        hidden_states,
        draft_input_ids,
        loss_mask,
        verifier_last_hidden_states,
        document_ids,
        position_ids,
    )

    logits = logits.view(draft.config.max_anchors, block, -1)[:1]
    hidden = hidden.view(draft.config.max_anchors, block, -1)[:1]
    block_tokens = draft_input_ids[:, anchor_pos : anchor_pos + block].view(1, block)
    prev_token_ids = torch.cat([block_tokens[:, :1], block_tokens[:, :-1]], dim=1)
    if draft.markov_head is not None:
        markov_bias = draft.markov_head.block_bias(
            prev_token_ids=prev_token_ids,
            hidden_states=hidden,
        )
        logits = logits + markov_bias
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
    total_records = len(records)
    if args.worker_rank is not None:
        records = records[args.worker_rank :: args.worker_count]
        logger.info(
            "[%s] worker %d/%d processing %d/%d samples",
            path.stem,
            args.worker_rank,
            args.worker_count,
            len(records),
            total_records,
        )

    stats = EvalStats()
    artifacts: list[dict[str, Any]] = []
    start = time.perf_counter()
    use_tqdm = tqdm is not None and not args.no_progress
    iterator = enumerate(records, start=1)
    if use_tqdm:
        iterator = tqdm(
            iterator,
            total=len(records),
            desc=f"{path.stem}",
            unit="sample",
            dynamic_ncols=True,
        )
    for row_idx, record in iterator:
        prompt = _prompt_from_record(record, tokenizer, source=f"{path}:{row_idx}")
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
        if not args.skip_artifacts:
            artifacts.append({"prompt": prompt, "output_token_ids": token_ids})
        elapsed = time.perf_counter() - start
        if elapsed > 0:
            out_tps = stats.total_output_tokens / elapsed
            if use_tqdm:
                iterator.set_postfix(
                    out_tok=stats.total_output_tokens,
                    out_tps=f"{out_tps:.2f}",
                    acc_len=f"{stats.acceptance_length:.3f}",
                )
            elif row_idx == 1 or row_idx % args.log_every == 0 or row_idx == len(records):
                logger.info(
                    "[%s] %d/%d samples | out_tok=%d | out_tps=%.2f | acc_len=%.3f",
                    path.stem,
                    row_idx,
                    len(records),
                    stats.total_output_tokens,
                    out_tps,
                    stats.acceptance_length,
                )
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


def _visible_devices_for(device: str) -> tuple[str | None, list[str]]:
    if str(device).startswith("npu"):
        env_name = "ASCEND_RT_VISIBLE_DEVICES"
    elif str(device).startswith("cuda"):
        env_name = "CUDA_VISIBLE_DEVICES"
    else:
        return None, []

    raw = os.environ.get(env_name, "")
    devices = [item.strip() for item in raw.split(",") if item.strip()]
    return env_name, devices


def _resolve_num_workers(args: argparse.Namespace) -> int:
    if args.worker_rank is not None:
        return args.worker_count
    if args.num_workers != "auto":
        return max(1, int(args.num_workers))
    _, devices = _visible_devices_for(args.device)
    return max(1, len(devices))


def _child_cmd(args: argparse.Namespace, rank: int, worker_count: int, output_dir: Path):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--verifier-model",
        args.verifier_model,
        "--draft-model",
        args.draft_model,
        "--datasets-root",
        str(args.datasets_root),
        "--output-dir",
        str(output_dir),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--draft-attn-impl",
        args.draft_attn_impl,
        "--num-workers",
        str(worker_count),
        "--worker-rank",
        str(rank),
        "--worker-count",
        str(worker_count),
        "--no-progress",
        "--log-every",
        str(args.log_every),
    ]
    if args.datasets:
        cmd.extend(["--datasets", args.datasets])
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.skip_artifacts:
        cmd.append("--skip-artifacts")
    return cmd


def _merge_worker_outputs(output_dir: Path, worker_dirs: list[Path]) -> None:
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for worker_dir in worker_dirs:
        summary_path = worker_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing worker summary: {summary_path}")
        with summary_path.open(encoding="utf-8") as f:
            for row in json.load(f):
                rows_by_dataset.setdefault(row["dataset"], []).append(row)

    merged_rows: list[dict[str, Any]] = []
    for dataset, rows in sorted(rows_by_dataset.items()):
        elapsed = max(float(row["elapsed_s"]) for row in rows)
        num_requests = sum(int(row["num_requests"]) for row in rows)
        total_output_tokens = sum(int(row["total_output_tokens"]) for row in rows)
        num_proposals = sum(int(row["num_proposals"]) for row in rows)
        num_proposed = sum(int(row["num_proposed_draft_tokens"]) for row in rows)
        num_accepted = sum(int(row["num_accepted_draft_tokens"]) for row in rows)
        merged_rows.append(
            {
                "dataset": dataset,
                "num_requests": num_requests,
                "elapsed_s": elapsed,
                "requests_per_second": num_requests / elapsed if elapsed else 0,
                "output_tokens_per_second": (
                    total_output_tokens / elapsed if elapsed else 0
                ),
                "total_output_tokens": total_output_tokens,
                "num_proposals": num_proposals,
                "num_proposed_draft_tokens": num_proposed,
                "num_accepted_draft_tokens": num_accepted,
                "acceptance_length": (
                    1.0 + num_accepted / num_proposals if num_proposals else 1.0
                ),
                "accepted_draft_length": (
                    num_accepted / num_proposals if num_proposals else 0.0
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(merged_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(merged_rows, f, indent=2)

    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    for worker_dir in worker_dirs:
        worker_artifacts = worker_dir / "artifacts"
        if not worker_artifacts.exists():
            continue
        for artifact_path in worker_artifacts.glob("*.jsonl"):
            with artifact_path.open(encoding="utf-8") as src, (
                artifacts_dir / artifact_path.name
            ).open("a", encoding="utf-8") as dst:
                for line in src:
                    dst.write(line)


def _run_multi_worker(args: argparse.Namespace, worker_count: int) -> None:
    env_name, devices = _visible_devices_for(args.device)
    if devices and len(devices) < worker_count:
        raise ValueError(
            f"Requested {worker_count} workers but {env_name} has only "
            f"{len(devices)} visible devices: {devices}"
        )

    worker_root = args.output_dir / "workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    logger.info("Launching %d worker(s)", worker_count)

    procs = []
    worker_dirs = []
    for rank in range(worker_count):
        worker_dir = worker_root / f"rank{rank}"
        worker_dirs.append(worker_dir)
        env = os.environ.copy()
        if env_name is not None:
            visible = devices[rank] if devices else str(rank)
            env[env_name] = visible
            logger.info("Worker %d uses %s=%s", rank, env_name, visible)
        cmd = _child_cmd(args, rank, worker_count, worker_dir)
        procs.append(subprocess.Popen(cmd, env=env))  # noqa: S603

    failed = []
    for rank, proc in enumerate(procs):
        ret = proc.wait()
        if ret != 0:
            failed.append((rank, ret))
    if failed:
        raise RuntimeError(f"Worker failures: {failed}")

    _merge_worker_outputs(args.output_dir, worker_dirs)
    logger.info("Merged worker results into %s", args.output_dir)


def _resolve_draft_attn_impl(args: argparse.Namespace) -> str | None:
    if args.draft_attn_impl != "auto":
        return args.draft_attn_impl
    if str(args.device).startswith("npu"):
        return "sdpa"
    return None


def run(args: argparse.Namespace) -> None:
    global torch

    worker_count = _resolve_num_workers(args)
    if args.worker_rank is None and worker_count > 1:
        _run_multi_worker(args, worker_count)
        return

    import torch as torch_module  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    from speculators.models.dspark.core import DSparkDraftModel  # noqa: PLC0415

    torch = torch_module
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"
    logger.info("Loading tokenizer: %s", args.verifier_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.verifier_model,
        trust_remote_code=args.trust_remote_code,
    )
    logger.info(
        "Loading verifier model: %s (device=%s, dtype=%s)",
        args.verifier_model,
        args.device,
        args.dtype,
    )
    target = AutoModelForCausalLM.from_pretrained(
        args.verifier_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    logger.info("Loading DSpark draft model: %s", args.draft_model)
    draft_config = DSparkDraftModel.config_class.from_pretrained(args.draft_model)
    draft_attn_impl = _resolve_draft_attn_impl(args)
    if draft_attn_impl is not None:
        logger.info("Using draft attention backend: %s", draft_attn_impl)
        draft_config.transformer_layer_config._attn_implementation = draft_attn_impl
    draft = DSparkDraftModel.from_pretrained(
        args.draft_model,
        config=draft_config,
    ).to(device)
    target.eval()
    draft.eval()

    dataset_names = args.datasets.split(",") if args.datasets else None
    dataset_paths = _discover_datasets(args.datasets_root, dataset_names)
    logger.info(
        "Discovered %d dataset(s): %s",
        len(dataset_paths),
        ", ".join(path.stem for path in dataset_paths),
    )
    rows: list[dict[str, Any]] = []
    artifacts_by_dataset: dict[str, list[dict[str, Any]]] = {}

    for path in dataset_paths:
        logger.info("[%s] evaluating %s", path.stem, path)
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
    parser.add_argument(
        "--num-workers",
        default="auto",
        help=(
            "Number of parallel worker processes. 'auto' uses the number of "
            "visible ASCEND/CUDA devices; 1 disables multi-worker mode."
        ),
    )
    parser.add_argument(
        "--draft-attn-impl",
        choices=["auto", "simple_flex_attention", "sdpa", "eager"],
        default="auto",
        help=(
            "Draft attention backend. auto keeps the checkpoint setting except on "
            "NPU, where it uses sdpa because FlexAttention is unsupported."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm sample progress bars.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="When tqdm is unavailable or disabled, log progress every N samples.",
    )
    parser.add_argument(
        "--skip-artifacts",
        action="store_true",
        help="Do not write per-sample output_token_ids artifacts.",
    )
    parser.add_argument("--worker-rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
