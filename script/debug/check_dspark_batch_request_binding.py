#!/usr/bin/env python3
"""Replay one DSpark batch's vLLM requests and check request/hidden binding.

This script targets the remaining ambiguity after single-request checks pass:
whether the bad hidden states only appear when the normal batch request set is
submitted to vLLM.

For the multipack batch containing ``--dataset-index`` it:

1. Builds single-request references for every document in the batch.
2. Replays the same document prompts either sequentially or concurrently.
3. For every replayed request, compares:
   - connector token_ids vs dataset token_ids
   - replay prompt_logprobs vs single-request reference prompt_logprobs
   - replay connector hidden vs single-request reference connector hidden

Interpretation:

- replay prompt_logprobs wrong and hidden wrong:
  target forward/context/mask/batching is implicated.
- replay prompt_logprobs correct but hidden wrong:
  connector/export/request binding is implicated.
- both correct:
  earlier bad tensor likely came from client file association, read lifecycle, or
  downstream data handling.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay one DSpark batch's vLLM requests with prompt_logprobs and hidden states."
    )
    parser.add_argument("--model-path", default="/models/Qwen3-4B")
    parser.add_argument("--data-path", default="/data/open_perfectblend_qwen3_4b_100k")
    parser.add_argument("--dataset-index", type=int, default=45760)
    parser.add_argument("--batch-indices", type=int, nargs="*", default=None)
    parser.add_argument("--vllm-endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--device", default="npu:15")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--total-seq-len", type=int, default=3072)
    parser.add_argument("--local-start", type=int, default=67)
    parser.add_argument("--gt-len", type=int, default=7)
    parser.add_argument("--prompt-logprobs", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--hidden-file-timeout", type=float, default=30.0)
    parser.add_argument(
        "--mode",
        choices=["sequential", "concurrent", "both"],
        default="both",
        help="How to replay the batch request set after references are built.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Thread count for concurrent mode. 0 means one worker per batch item.",
    )
    parser.add_argument("--logprob-tol", type=float, default=0.5)
    parser.add_argument("--hidden-tol", type=float, default=1e-2)
    parser.add_argument(
        "--compare-full-hidden",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also compute full-tensor hidden diffs, not only the local window.",
    )
    parser.add_argument(
        "--self-project",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Project each request's own connector hidden and compare with its own prompt_logprobs.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--keep-hidden-files", action="store_true")
    return parser


def _fmt(x: float) -> str:
    if math.isnan(x) or math.isinf(x):
        return str(x)
    if x == 0 or (1e-3 <= abs(x) < 1e4):
        return f"{x:.6f}"
    return f"{x:.6e}"


def _diff_stats(a: Any, b: Any) -> tuple[float, float]:
    diff = (a.detach().float().cpu() - b.detach().float().cpu()).abs()
    return float(diff.max().item()), float(diff.mean().item())


def _max_abs_delta(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b, strict=True)) if a else float("nan")


def _resolve_dtype(torch: Any, value: str) -> Any:
    if value == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def _wait_for_hidden_file(path_value: str, timeout: float) -> None:
    from speculators.data_generation.vllm_client import wait_for_lock

    path = Path(path_value)
    lock_path = Path(path_value + ".lock")
    deadline = time.monotonic() + timeout
    while lock_path.exists() or not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for hidden states file: {path}")
        if lock_path.exists():
            wait_for_lock(str(lock_path), timeout=max(deadline - time.monotonic(), 0.1))
            continue
        time.sleep(0.05)


def _delete_hidden_file(path_value: str | None) -> None:
    if path_value is None:
        return
    path = Path(path_value)
    path.unlink(missing_ok=True)
    Path(str(path) + ".lock").unlink(missing_ok=True)


def _make_openai_client(endpoint: str) -> tuple[Any, str]:
    import openai

    client = openai.OpenAI(base_url=endpoint, api_key="EMPTY", max_retries=0)
    model_id = client.models.list().data[0].id
    return client, model_id


def _load_hf_model(args: argparse.Namespace) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    dtype = _resolve_dtype(torch, args.dtype)
    kwargs: dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if dtype != "auto":
        kwargs["dtype"] = dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    except TypeError:
        if dtype != "auto":
            kwargs.pop("dtype", None)
            kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    model.to(torch.device(args.device))
    model.eval()
    return model


def _project_hidden_to_prompt_logprobs(
    *,
    hf_model: Any,
    hidden_states: Any,
    prompt: list[int],
    score_positions: list[int],
) -> list[float]:
    import torch

    device = next(hf_model.parameters()).device
    final_slot = hidden_states.shape[1] - 1
    hidden_positions = torch.tensor(
        [pos - 1 for pos in score_positions],
        dtype=torch.long,
        device=device,
    )
    target_ids = torch.tensor(
        [prompt[pos] for pos in score_positions],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        final_hidden = hidden_states[:, final_slot].to(device=device)
        selected_hidden = final_hidden.index_select(0, hidden_positions)
        selected_hidden = hf_model.model.norm(selected_hidden)
        logits = hf_model.lm_head(selected_hidden).float()
        logprobs = torch.log_softmax(logits, dim=-1)
        gathered = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return [float(x) for x in gathered.detach().cpu().tolist()]


def _request_scored_hidden(
    *,
    endpoint: str,
    model_id: str,
    prompt: list[int],
    score_positions: list[int],
    prompt_logprobs: int,
    request_timeout: float,
    hidden_file_timeout: float,
) -> dict[str, Any]:
    import openai
    from safetensors.torch import load_file
    from speculators.data_generation.vllm_client import (
        _extract_token_logprob,
        _kv_hidden_states_path,
        _prompt_logprobs,
        _prompt_token_ids,
    )

    client = openai.OpenAI(base_url=endpoint, api_key="EMPTY", max_retries=0)
    response = client.completions.create(
        model=model_id,
        prompt=prompt,
        max_tokens=1,
        extra_body={
            "return_token_ids": True,
            "prompt_logprobs": prompt_logprobs,
        },
        timeout=request_timeout,
    )

    response_prompt_ids = _prompt_token_ids(response)
    response_prompt_logprobs = _prompt_logprobs(response)
    hidden_path = _kv_hidden_states_path(response)
    if response_prompt_ids is None:
        raise RuntimeError("vLLM response missing prompt_token_ids")
    if response_prompt_logprobs is None:
        raise RuntimeError("vLLM response missing prompt_logprobs")
    if hidden_path is None:
        raise RuntimeError("vLLM response missing hidden_states_path")

    _wait_for_hidden_file(hidden_path, hidden_file_timeout)
    loaded = load_file(hidden_path)
    token_logprobs = [
        _extract_token_logprob(response_prompt_logprobs, pos, prompt[pos])
        for pos in score_positions
    ]
    return {
        "prompt_ids": list(response_prompt_ids),
        "hidden_path": hidden_path,
        "file_token_ids": loaded["token_ids"].detach().cpu().clone(),
        "hidden": loaded["hidden_states"].detach().cpu().clone(),
        "token_logprobs": token_logprobs,
    }


def _load_train_dataset(data_path: str) -> Any:
    from datasets import load_from_disk

    dataset = load_from_disk(data_path)
    split_idx = int(len(dataset) * 0.9)
    return dataset.select(range(split_idx))


def _item_token_ids(dataset: Any, index: int) -> list[int]:
    input_ids = dataset[int(index)]["input_ids"]
    if hasattr(input_ids, "tolist"):
        return [int(x) for x in input_ids.tolist()]
    return [int(x) for x in input_ids]


def _find_batch_indices(
    *,
    dataset: Any,
    dataset_index: int,
    total_seq_len: int,
) -> tuple[int, list[int]]:
    from speculators.train.distributed_batch_sampler import (
        MultipackDistributedBatchSamplerV2,
    )

    lengths = list(dataset.with_format(None)["seq_len"])
    sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=total_seq_len,
        lengths=lengths,
        num_replicas=1,
        rank=0,
    )
    for batch_index, batch in enumerate(list(sampler)):
        indices = [int(x) for x in batch.tolist()] if hasattr(batch, "tolist") else [int(x) for x in batch]
        if dataset_index in indices:
            return batch_index, indices
    raise RuntimeError(f"Dataset index {dataset_index} was not found in train batches")


def _score_positions_for_item(length: int, local_start: int, gt_len: int) -> list[int]:
    if local_start + gt_len >= length:
        raise ValueError(
            f"Item length {length} is too short for local_start={local_start}, gt_len={gt_len}"
        )
    return list(range(local_start + 1, local_start + gt_len + 1))


def _build_items(
    *,
    dataset: Any,
    indices: list[int],
    local_start: int,
    gt_len: int,
) -> list[dict[str, Any]]:
    items = []
    for doc_id, index in enumerate(indices):
        token_ids = _item_token_ids(dataset, index)
        score_positions = _score_positions_for_item(len(token_ids), local_start, gt_len)
        items.append(
            {
                "doc_id": doc_id,
                "dataset_index": int(index),
                "token_ids": token_ids,
                "score_positions": score_positions,
                "target_ids": [token_ids[pos] for pos in score_positions],
            }
        )
    return items


def _run_requests_sequential(
    *,
    items: list[dict[str, Any]],
    endpoint: str,
    model_id: str,
    prompt_logprobs: int,
    request_timeout: float,
    hidden_file_timeout: float,
) -> dict[int, dict[str, Any]]:
    out = {}
    for item in items:
        out[item["dataset_index"]] = _request_scored_hidden(
            endpoint=endpoint,
            model_id=model_id,
            prompt=item["token_ids"],
            score_positions=item["score_positions"],
            prompt_logprobs=prompt_logprobs,
            request_timeout=request_timeout,
            hidden_file_timeout=hidden_file_timeout,
        )
    return out


def _run_requests_concurrent(
    *,
    items: list[dict[str, Any]],
    endpoint: str,
    model_id: str,
    prompt_logprobs: int,
    request_timeout: float,
    hidden_file_timeout: float,
    max_workers: int,
) -> dict[int, dict[str, Any]]:
    out = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                _request_scored_hidden,
                endpoint=endpoint,
                model_id=model_id,
                prompt=item["token_ids"],
                score_positions=item["score_positions"],
                prompt_logprobs=prompt_logprobs,
                request_timeout=request_timeout,
                hidden_file_timeout=hidden_file_timeout,
            ): item["dataset_index"]
            for item in items
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            out[index] = future.result()
    return out


def _print_request_summary(
    *,
    label: str,
    items: list[dict[str, Any]],
    runs: dict[int, dict[str, Any]],
) -> None:
    print(f"TRACE {label}.requests")
    for item in items:
        run = runs[item["dataset_index"]]
        token_ids_match = bool(
            run["file_token_ids"].equal(
                __import__("torch").tensor(item["token_ids"], dtype=run["file_token_ids"].dtype)
            )
        )
        prompt_ids_match = run["prompt_ids"] == item["token_ids"]
        print(
            "  "
            f"doc_id={item['doc_id']} "
            f"dataset_index={item['dataset_index']} "
            f"prompt_len={len(item['token_ids'])} "
            f"hidden_path={run['hidden_path']} "
            f"prompt_ids_match={prompt_ids_match} "
            f"file_token_ids_match={token_ids_match} "
            f"hidden_shape={tuple(run['hidden'].shape)} "
            f"prompt_logprobs={[_fmt(x) for x in run['token_logprobs']]}"
        )


def _compare_mode(
    *,
    mode: str,
    items: list[dict[str, Any]],
    references: dict[int, dict[str, Any]],
    replays: dict[int, dict[str, Any]],
    local_start: int,
    gt_len: int,
    logprob_tol: float,
    hidden_tol: float,
    compare_full_hidden: bool,
) -> None:
    print(f"TRACE {mode}.compare")
    print(
        "  doc_id,dataset_index,plog_max_abs,final_window_hidden_max_abs,"
        "final_window_hidden_mean_abs,allslot_window_hidden_max_abs,"
        "allslot_window_hidden_mean_abs,full_hidden_max_abs,classification"
    )
    bad_plog = 0
    bad_hidden = 0
    for item in items:
        index = item["dataset_index"]
        ref = references[index]
        replay = replays[index]
        plog_diff = _max_abs_delta(
            replay["token_logprobs"],
            ref["token_logprobs"],
        )
        final_slot = ref["hidden"].shape[1] - 1
        local_slice = slice(local_start, local_start + gt_len)
        final_max, final_mean = _diff_stats(
            replay["hidden"][local_slice, final_slot],
            ref["hidden"][local_slice, final_slot],
        )
        allslot_max, allslot_mean = _diff_stats(
            replay["hidden"][local_slice],
            ref["hidden"][local_slice],
        )
        full_hidden = "<skipped>"
        if compare_full_hidden:
            full_max, _full_mean = _diff_stats(replay["hidden"], ref["hidden"])
            full_hidden = _fmt(full_max)

        plog_bad = plog_diff > logprob_tol
        hidden_bad = final_max > hidden_tol
        bad_plog += int(plog_bad)
        bad_hidden += int(hidden_bad)
        if plog_bad and hidden_bad:
            classification = "plog_bad_hidden_bad_target_forward_or_mask"
        elif not plog_bad and hidden_bad:
            classification = "plog_ok_hidden_bad_connector_or_binding"
        elif not plog_bad and not hidden_bad:
            classification = "plog_ok_hidden_ok"
        else:
            classification = "plog_bad_hidden_ok_unexpected"

        print(
            "  "
            f"{item['doc_id']},"
            f"{index},"
            f"{_fmt(plog_diff)},"
            f"{_fmt(final_max)},"
            f"{_fmt(final_mean)},"
            f"{_fmt(allslot_max)},"
            f"{_fmt(allslot_mean)},"
            f"{full_hidden},"
            f"{classification}"
        )

    print(f"TRACE {mode}.summary")
    print(f"  bad_plog_count={bad_plog}")
    print(f"  bad_hidden_count={bad_hidden}")
    if bad_plog and bad_hidden:
        print("  conclusion=some replay prompt_logprobs and hidden differ from reference; target forward/context/mask/batching is implicated")
    elif not bad_plog and bad_hidden:
        print("  conclusion=replay prompt_logprobs match but hidden differs; connector/export/request binding is implicated")
    elif not bad_plog and not bad_hidden:
        print("  conclusion=replay prompt_logprobs and hidden match references; earlier bad tensor likely came from client file lifecycle or downstream data path")
    else:
        print("  conclusion=mixed; inspect per-document classifications")


def _print_self_projection(
    *,
    label: str,
    items: list[dict[str, Any]],
    runs: dict[int, dict[str, Any]],
    hf_model: Any,
    logprob_tol: float,
) -> None:
    print(f"TRACE {label}.self_projection")
    print("  doc_id,dataset_index,self_projection_vs_own_prompt_max_abs,classification,projected_logprobs")
    bad = 0
    for item in items:
        index = item["dataset_index"]
        run = runs[index]
        projected = _project_hidden_to_prompt_logprobs(
            hf_model=hf_model,
            hidden_states=run["hidden"],
            prompt=item["token_ids"],
            score_positions=item["score_positions"],
        )
        max_abs = _max_abs_delta(projected, run["token_logprobs"])
        is_bad = max_abs > logprob_tol
        bad += int(is_bad)
        classification = "self_projection_bad" if is_bad else "self_projection_ok"
        print(
            "  "
            f"{item['doc_id']},"
            f"{index},"
            f"{_fmt(max_abs)},"
            f"{classification},"
            f"{[_fmt(x) for x in projected]}"
        )
    print(f"TRACE {label}.self_projection_summary")
    print(f"  bad_self_projection_count={bad}")
    if bad:
        print("  conclusion=some connector hidden tensors do not reconstruct their own request prompt_logprobs")
    else:
        print("  conclusion=connector hidden tensors reconstruct their own request prompt_logprobs")


def main() -> None:
    args = _parser().parse_args()

    dataset = _load_train_dataset(args.data_path)
    if args.batch_indices:
        batch_index = -1
        indices = [int(x) for x in args.batch_indices]
    else:
        batch_index, indices = _find_batch_indices(
            dataset=dataset,
            dataset_index=args.dataset_index,
            total_seq_len=args.total_seq_len,
        )
    if args.dataset_index not in indices:
        raise ValueError(
            f"--dataset-index {args.dataset_index} is not in batch indices {indices}"
        )

    items = _build_items(
        dataset=dataset,
        indices=indices,
        local_start=args.local_start,
        gt_len=args.gt_len,
    )
    _, model_id = _make_openai_client(args.vllm_endpoint)

    print("TRACE config")
    print(f"  repo={ROOT}")
    print(f"  model_path={args.model_path}")
    print(f"  data_path={args.data_path}")
    print(f"  dataset_index={args.dataset_index}")
    print(f"  batch_index={batch_index}")
    print(f"  batch_indices={indices}")
    print(f"  vllm_endpoint={args.vllm_endpoint}")
    print(f"  model_id={model_id}")
    print(f"  total_seq_len={args.total_seq_len}")
    print(f"  local_hidden_window={args.local_start}:{args.local_start + args.gt_len}")
    print(f"  local_score_positions={list(range(args.local_start + 1, args.local_start + args.gt_len + 1))}")
    print(f"  mode={args.mode}")
    print(f"  compare_full_hidden={args.compare_full_hidden}")
    print(f"  self_project={args.self_project}")

    print("TRACE batch_items")
    for item in items:
        print(
            "  "
            f"doc_id={item['doc_id']} "
            f"dataset_index={item['dataset_index']} "
            f"prompt_len={len(item['token_ids'])} "
            f"score_positions={item['score_positions']} "
            f"target_ids={item['target_ids']}"
        )

    references = _run_requests_sequential(
        items=items,
        endpoint=args.vllm_endpoint,
        model_id=model_id,
        prompt_logprobs=args.prompt_logprobs,
        request_timeout=args.request_timeout,
        hidden_file_timeout=args.hidden_file_timeout,
    )
    _print_request_summary(label="reference", items=items, runs=references)
    hf_model = _load_hf_model(args) if args.self_project else None
    if hf_model is not None:
        _print_self_projection(
            label="reference",
            items=items,
            runs=references,
            hf_model=hf_model,
            logprob_tol=args.logprob_tol,
        )

    replay_modes = ["sequential", "concurrent"] if args.mode == "both" else [args.mode]
    replay_runs: list[tuple[str, dict[int, dict[str, Any]]]] = []
    for mode in replay_modes:
        if mode == "sequential":
            runs = _run_requests_sequential(
                items=items,
                endpoint=args.vllm_endpoint,
                model_id=model_id,
                prompt_logprobs=args.prompt_logprobs,
                request_timeout=args.request_timeout,
                hidden_file_timeout=args.hidden_file_timeout,
            )
        else:
            max_workers = args.max_workers if args.max_workers > 0 else len(items)
            runs = _run_requests_concurrent(
                items=items,
                endpoint=args.vllm_endpoint,
                model_id=model_id,
                prompt_logprobs=args.prompt_logprobs,
                request_timeout=args.request_timeout,
                hidden_file_timeout=args.hidden_file_timeout,
                max_workers=max_workers,
            )
        replay_runs.append((mode, runs))
        _print_request_summary(label=mode, items=items, runs=runs)
        if hf_model is not None:
            _print_self_projection(
                label=mode,
                items=items,
                runs=runs,
                hf_model=hf_model,
                logprob_tol=args.logprob_tol,
            )
        _compare_mode(
            mode=mode,
            items=items,
            references=references,
            replays=runs,
            local_start=args.local_start,
            gt_len=args.gt_len,
            logprob_tol=args.logprob_tol,
            hidden_tol=args.hidden_tol,
            compare_full_hidden=args.compare_full_hidden,
        )

    if not args.keep_hidden_files:
        for run in references.values():
            _delete_hidden_file(run.get("hidden_path"))
        for _mode, runs in replay_runs:
            for run in runs.values():
                _delete_hidden_file(run.get("hidden_path"))


if __name__ == "__main__":
    main()
