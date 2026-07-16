#!/usr/bin/env python3
"""Debug vLLM hidden-state/token length mismatches for Arrow training data.

This script intentionally mirrors the online training dataloader path:
it loads the preprocessed Arrow dataset, builds the same vLLM request payload,
requests hidden states from an already-running server, then prints detailed
diagnostics instead of swallowing the mismatch as a dataloader warning.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 3


def _parse_indices(value: str | None) -> list[int]:
    if not value:
        return []
    parts = value.replace(",", " ").split()
    return [int(part) for part in parts]


def _short_list(values: list[int], n: int) -> dict[str, list[int]]:
    if len(values) <= 2 * n:
        return {"all": values}
    return {"head": values[:n], "tail": values[-n:]}


def _first_diff(left: list[int], right: list[int]) -> dict[str, Any] | None:
    limit = min(len(left), len(right))
    for idx in range(limit):
        if left[idx] != right[idx]:
            return {"index": idx, "expected": left[idx], "actual": right[idx]}
    if len(left) != len(right):
        return {
            "index": limit,
            "expected": left[limit] if limit < len(left) else None,
            "actual": right[limit] if limit < len(right) else None,
        }
    return None


def _summarize_value(value: Any) -> Any:
    if hasattr(value, "shape"):
        return {
            "type": type(value).__name__,
            "shape": list(value.shape),
            "dtype": str(getattr(value, "dtype", "")),
        }
    if isinstance(value, list):
        summary: dict[str, Any] = {"type": "list", "len": len(value)}
        if value:
            summary["first_type"] = type(value[0]).__name__
            if isinstance(value[0], dict):
                summary["first_keys"] = sorted(value[0].keys())
        return summary
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def _sample_indices(args: argparse.Namespace, dataset_len: int) -> list[int]:
    indices = _parse_indices(args.indices)
    if args.scan:
        scan_indices = list(range(dataset_len))
        if args.shuffle:
            rng = random.Random(args.seed)
            rng.shuffle(scan_indices)
        indices.extend(scan_indices[: args.scan])
    if not indices:
        raise ValueError("Provide --indices and/or --scan")
    for index in indices:
        if index < 0 or index >= dataset_len:
            raise IndexError(f"Index {index} outside dataset length {dataset_len}")
    return indices


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _inspect_one(
    *,
    dataset,
    index: int,
    client,
    model: str,
    args: argparse.Namespace,
) -> bool:
    from safetensors.torch import load_file  # noqa: PLC0415
    from speculators.train.data import build_client_item  # noqa: PLC0415

    item = dataset[index]
    expected_ids = item["input_ids"].tolist()
    client_item = build_client_item(item)
    uses_messages = "messages" in client_item
    messages = client_item.get("messages")

    print(
        f"[sample {index}] request | expected_tokens={len(expected_ids)} "
        f"| uses_messages={uses_messages}"
    )
    if messages is None:
        response = client.completions.create(
            model=model,
            prompt=client_item["input_ids"],
            max_tokens=1,
            extra_body={"return_token_ids": True},
            timeout=args.request_timeout,
        )
        response_prompt_ids = getattr(response.choices[0], "prompt_token_ids", None)
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1,
            extra_body={"add_generation_prompt": False, "return_token_ids": True},
            timeout=args.request_timeout,
        )
        response_prompt_ids = getattr(response, "prompt_token_ids", None)
    kv_transfer_params = getattr(response, "kv_transfer_params", None)
    hs_path = None if kv_transfer_params is None else kv_transfer_params.get(
        "hidden_states_path"
    )
    if hs_path is None:
        raise ValueError("Response missing kv_transfer_params.hidden_states_path")
    loaded = load_file(hs_path)
    actual_ids = loaded["token_ids"].tolist()
    hidden = loaded["hidden_states"]

    token_match = actual_ids == expected_ids
    response_prompt_match = response_prompt_ids == expected_ids
    hidden_len_match = hidden.shape[0] == len(expected_ids)
    ok = token_match and hidden_len_match and not hidden.isnan().any().item()

    payload: dict[str, Any] = {
        "sample": index,
        "ok": ok,
        "hidden_states_path": hs_path,
        "dataset_fields": {
            key: _summarize_value(value) for key, value in sorted(item.items())
        },
        "request": {
            "uses_messages": uses_messages,
            "input_ids_len": len(client_item["input_ids"]),
            "input_ids": _short_list(client_item["input_ids"], args.token_preview),
        },
        "response": {
            "prompt_token_ids_len": (
                None if response_prompt_ids is None else len(response_prompt_ids)
            ),
            "prompt_token_ids_match": response_prompt_match,
            "prompt_token_ids": (
                None
                if response_prompt_ids is None
                else _short_list(response_prompt_ids, args.token_preview)
            ),
            "token_ids_len": len(actual_ids),
            "hidden_states_shape": list(hidden.shape),
            "hidden_states_dtype": str(hidden.dtype),
            "hidden_has_nan": bool(hidden.isnan().any().item()),
            "token_ids": _short_list(actual_ids, args.token_preview),
        },
        "expected": {
            "input_ids_len": len(expected_ids),
            "input_ids": _short_list(expected_ids, args.token_preview),
        },
        "checks": {
            "token_ids_match": token_match,
            "hidden_seq_len_match": hidden_len_match,
            "response_prompt_ids_match": response_prompt_match,
            "first_token_diff": _first_diff(expected_ids, actual_ids),
            "first_response_prompt_diff": (
                None
                if response_prompt_ids is None
                else _first_diff(expected_ids, response_prompt_ids)
            ),
            "hidden_minus_expected_len": int(hidden.shape[0] - len(expected_ids)),
            "response_tokens_minus_expected_len": len(actual_ids) - len(expected_ids),
        },
    }
    _print_json(payload)

    if args.delete_generated:
        path = Path(hs_path)
        path.unlink(missing_ok=True)
        Path(str(path) + ".lock").unlink(missing_ok=True)
        print(f"[sample {index}] deleted generated hidden states: {path}")
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose hidden-state length mismatches from a live vLLM server.",
    )
    parser.add_argument("--data-path", required=True, help="Preprocessed Arrow dataset")
    parser.add_argument("--vllm-endpoint", default="http://localhost:8000/v1")
    parser.add_argument(
        "--model",
        default=None,
        help="Expected vLLM model id. Defaults to first model reported by server.",
    )
    parser.add_argument(
        "--indices",
        default=None,
        help="Comma/space-separated dataset indices to inspect.",
    )
    parser.add_argument(
        "--scan",
        type=int,
        default=0,
        help="Also inspect the first N samples, or N shuffled samples with --shuffle.",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--token-preview", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--keep-generated",
        action="store_true",
        help="Do not delete hidden-state files generated by the server.",
    )
    args = parser.parse_args()
    args.delete_generated = not args.keep_generated
    return args


def main() -> None:
    args = parse_args()

    import openai  # noqa: PLC0415
    from datasets import load_from_disk  # noqa: PLC0415

    dataset = load_from_disk(args.data_path)
    indices = _sample_indices(args, len(dataset))

    client = openai.OpenAI(
        base_url=args.vllm_endpoint,
        api_key="EMPTY",
        max_retries=0,
    )
    model_id = client.models.list().data[0].id
    if args.model is not None and args.model != model_id:
        raise ValueError(
            f"--model {args.model!r} does not match vLLM model id {model_id!r}"
        )

    print(
        f"dataset={args.data_path} | len={len(dataset)} | "
        f"endpoint={args.vllm_endpoint} | model={model_id} | indices={indices}"
    )
    failures = 0
    for index in indices:
        try:
            if not _inspect_one(
                dataset=dataset,
                index=index,
                client=client,
                model=model_id,
                args=args,
            ):
                failures += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[sample {index}] ERROR: {type(exc).__name__}: {exc}")

    if failures:
        raise SystemExit(f"{failures}/{len(indices)} sample(s) failed diagnostics")


if __name__ == "__main__":
    main()
