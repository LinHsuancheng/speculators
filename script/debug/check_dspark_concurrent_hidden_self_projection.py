#!/usr/bin/env python3
"""Check whether concurrent vLLM connector hidden states are self-consistent.

For a DSpark batch's document prompts, this script sends requests with both
``prompt_logprobs`` and connector hidden-state export enabled. It then projects
each request's own exported final hidden states through the target model's final
norm and lm_head, and compares that result with the same response's
``prompt_logprobs``.

This intentionally does not compare against a previous run. It only checks the
closed loop for each individual response:

    response hidden -> final norm -> lm_head -> target-token logprob
    vs
    response prompt_logprobs
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-project concurrent vLLM hidden-state responses."
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
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--keep-hidden-files", action="store_true")
    return parser.parse_args()


def fmt(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        return str(value)
    if value == 0 or (1e-3 <= abs(value) < 1e4):
        return f"{value:.6f}"
    return f"{value:.6e}"


def torch_dtype(torch: Any, name: str) -> Any:
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def load_train_split(data_path: str) -> Any:
    from datasets import load_from_disk

    dataset = load_from_disk(data_path)
    return dataset.select(range(int(len(dataset) * 0.9)))


def token_ids_for(dataset: Any, index: int) -> list[int]:
    value = dataset[int(index)]["input_ids"]
    if hasattr(value, "tolist"):
        return [int(x) for x in value.tolist()]
    return [int(x) for x in value]


def find_batch_indices(dataset: Any, dataset_index: int, total_seq_len: int) -> tuple[int, list[int]]:
    from speculators.train.distributed_batch_sampler import (
        MultipackDistributedBatchSamplerV2,
    )

    sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=total_seq_len,
        lengths=list(dataset.with_format(None)["seq_len"]),
        num_replicas=1,
        rank=0,
    )
    for batch_index, batch in enumerate(list(sampler)):
        values = [int(x) for x in batch.tolist()] if hasattr(batch, "tolist") else [int(x) for x in batch]
        if dataset_index in values:
            return batch_index, values
    raise RuntimeError(f"dataset index {dataset_index} not found in train batches")


def build_items(dataset: Any, indices: list[int], local_start: int, gt_len: int) -> list[dict[str, Any]]:
    items = []
    for doc_id, index in enumerate(indices):
        tokens = token_ids_for(dataset, index)
        if local_start + gt_len >= len(tokens):
            raise ValueError(
                f"index {index} length {len(tokens)} is too short for requested window"
            )
        score_positions = list(range(local_start + 1, local_start + gt_len + 1))
        items.append(
            {
                "doc_id": doc_id,
                "dataset_index": int(index),
                "token_ids": tokens,
                "score_positions": score_positions,
                "target_ids": [tokens[pos] for pos in score_positions],
            }
        )
    return items


def wait_for_hidden_file(path_value: str, timeout: float) -> None:
    from speculators.data_generation.vllm_client import wait_for_lock

    path = Path(path_value)
    lock = Path(path_value + ".lock")
    deadline = time.monotonic() + timeout
    while lock.exists() or not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        if lock.exists():
            wait_for_lock(str(lock), timeout=max(deadline - time.monotonic(), 0.1))
        else:
            time.sleep(0.05)


def delete_hidden_file(path_value: str | None) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.unlink(missing_ok=True)
    Path(str(path) + ".lock").unlink(missing_ok=True)


def model_id_for(endpoint: str) -> str:
    import openai

    client = openai.OpenAI(base_url=endpoint, api_key="EMPTY", max_retries=0)
    return client.models.list().data[0].id


def request_one(
    *,
    endpoint: str,
    model_id: str,
    item: dict[str, Any],
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
        prompt=item["token_ids"],
        max_tokens=1,
        extra_body={
            "return_token_ids": True,
            "prompt_logprobs": prompt_logprobs,
        },
        timeout=request_timeout,
    )
    prompt_ids = _prompt_token_ids(response)
    prompt_logprob_obj = _prompt_logprobs(response)
    hidden_path = _kv_hidden_states_path(response)
    if prompt_ids is None:
        raise RuntimeError("response missing prompt_token_ids")
    if prompt_logprob_obj is None:
        raise RuntimeError("response missing prompt_logprobs")
    if hidden_path is None:
        raise RuntimeError("response missing hidden_states_path")

    wait_for_hidden_file(hidden_path, hidden_file_timeout)
    tensors = load_file(hidden_path)
    return {
        "doc_id": item["doc_id"],
        "dataset_index": item["dataset_index"],
        "prompt_ids": list(prompt_ids),
        "file_token_ids": tensors["token_ids"].detach().cpu().clone(),
        "hidden": tensors["hidden_states"].detach().cpu().clone(),
        "hidden_path": hidden_path,
        "prompt_logprobs": [
            _extract_token_logprob(prompt_logprob_obj, pos, item["token_ids"][pos])
            for pos in item["score_positions"]
        ],
    }


def request_batch_concurrently(
    *,
    endpoint: str,
    model_id: str,
    items: list[dict[str, Any]],
    prompt_logprobs: int,
    request_timeout: float,
    hidden_file_timeout: float,
    max_workers: int,
) -> dict[int, dict[str, Any]]:
    workers = max_workers if max_workers > 0 else len(items)
    outputs = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                request_one,
                endpoint=endpoint,
                model_id=model_id,
                item=item,
                prompt_logprobs=prompt_logprobs,
                request_timeout=request_timeout,
                hidden_file_timeout=hidden_file_timeout,
            ): item["dataset_index"]
            for item in items
        }
        for future in as_completed(futures):
            outputs[futures[future]] = future.result()
    return outputs


def load_target_model(args: argparse.Namespace) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch_dtype(torch, args.dtype)
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


def project_response(model: Any, item: dict[str, Any], response: dict[str, Any]) -> list[float]:
    import torch

    device = next(model.parameters()).device
    final_slot = response["hidden"].shape[1] - 1
    hidden_positions = torch.tensor(
        [pos - 1 for pos in item["score_positions"]],
        dtype=torch.long,
        device=device,
    )
    target_ids = torch.tensor(item["target_ids"], dtype=torch.long, device=device)
    with torch.no_grad():
        final_hidden = response["hidden"][:, final_slot].to(device=device)
        selected = final_hidden.index_select(0, hidden_positions)
        selected = model.model.norm(selected)
        logits = model.lm_head(selected).float()
        logprobs = torch.log_softmax(logits, dim=-1)
        values = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return [float(x) for x in values.detach().cpu().tolist()]


def max_abs_delta(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def main() -> None:
    args = parse_args()
    dataset = load_train_split(args.data_path)
    if args.batch_indices:
        batch_index = -1
        indices = [int(x) for x in args.batch_indices]
    else:
        batch_index, indices = find_batch_indices(
            dataset,
            args.dataset_index,
            args.total_seq_len,
        )
    items = build_items(dataset, indices, args.local_start, args.gt_len)
    model_id = model_id_for(args.vllm_endpoint)

    print("TRACE config")
    print(f"  repo={ROOT}")
    print(f"  model_path={args.model_path}")
    print(f"  data_path={args.data_path}")
    print(f"  dataset_index={args.dataset_index}")
    print(f"  batch_index={batch_index}")
    print(f"  batch_indices={indices}")
    print(f"  vllm_endpoint={args.vllm_endpoint}")
    print(f"  model_id={model_id}")
    print(f"  device={args.device}")
    print(f"  dtype={args.dtype}")
    print(f"  rounds={args.rounds}")
    print(f"  tolerance={args.tolerance}")

    print("TRACE items")
    for item in items:
        print(
            "  "
            f"doc_id={item['doc_id']} "
            f"dataset_index={item['dataset_index']} "
            f"prompt_len={len(item['token_ids'])} "
            f"score_positions={item['score_positions']} "
            f"target_ids={item['target_ids']}"
        )

    model = load_target_model(args)
    all_paths: list[str] = []
    for round_idx in range(args.rounds):
        responses = request_batch_concurrently(
            endpoint=args.vllm_endpoint,
            model_id=model_id,
            items=items,
            prompt_logprobs=args.prompt_logprobs,
            request_timeout=args.request_timeout,
            hidden_file_timeout=args.hidden_file_timeout,
            max_workers=args.max_workers,
        )
        print(f"TRACE round{round_idx}.self_projection")
        print("  doc_id,dataset_index,prompt_ids_match,file_token_ids_match,hidden_shape,max_abs_diff,classification,projected,reported")
        bad = 0
        for item in items:
            response = responses[item["dataset_index"]]
            all_paths.append(response["hidden_path"])
            projected = project_response(model, item, response)
            reported = response["prompt_logprobs"]
            max_abs = max_abs_delta(projected, reported)
            is_bad = max_abs > args.tolerance
            bad += int(is_bad)
            prompt_ids_match = response["prompt_ids"] == item["token_ids"]
            file_token_ids_match = response["file_token_ids"].tolist() == item["token_ids"]
            classification = "bad" if is_bad else "ok"
            print(
                "  "
                f"{item['doc_id']},"
                f"{item['dataset_index']},"
                f"{prompt_ids_match},"
                f"{file_token_ids_match},"
                f"{tuple(response['hidden'].shape)},"
                f"{fmt(max_abs)},"
                f"{classification},"
                f"{[fmt(x) for x in projected]},"
                f"{[fmt(x) for x in reported]}"
            )
        print(f"TRACE round{round_idx}.summary")
        print(f"  bad_count={bad}")
        if bad:
            print("  conclusion=some concurrent hidden tensors do not reconstruct their own prompt_logprobs")
        else:
            print("  conclusion=all concurrent hidden tensors reconstruct their own prompt_logprobs")

    if not args.keep_hidden_files:
        for path in all_paths:
            delete_hidden_file(path)


if __name__ == "__main__":
    main()
