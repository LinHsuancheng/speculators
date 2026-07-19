#!/usr/bin/env python3
"""Strict H0/H1 ablation for vLLM hidden-state extraction.

The only request-body difference is whether ``prompt_logprobs`` is present:

H0a/H0b: no prompt_logprobs
H1a/H1b: prompt_logprobs=1

Each hidden-state file is loaded and cloned immediately after the request. The
script compares H0/H1 repeat stability, H0/H1 against HF hidden states, and H0
against H1 across either the full sequence or selected windows.
"""

from __future__ import annotations

import argparse
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
        description="Strict prompt_logprobs ablation for vLLM hidden export."
    )
    parser.add_argument("--model-path", default="/models/Qwen3-4B")
    parser.add_argument("--data-path", default="/data/open_perfectblend_qwen3_4b_100k")
    parser.add_argument("--dataset-index", type=int, default=45760)
    parser.add_argument("--vllm-endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--device", default="npu:15")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--target-layer-ids",
        type=int,
        nargs="+",
        default=[1, 9, 17, 25, 33],
    )
    parser.add_argument("--final-layer-id", type=int, default=None)
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=0,
        help="0 means use the full dataset item.",
    )
    parser.add_argument(
        "--prefix-lens",
        type=int,
        nargs="+",
        default=[32, 68, 128],
        help="Windows to inspect; the request prompt is unchanged.",
    )
    parser.add_argument("--gt-len", type=int, default=7)
    parser.add_argument("--prompt-logprobs", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--hidden-file-timeout", type=float, default=30.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--keep-hidden-files", action="store_true")
    parser.add_argument(
        "--full-sequence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also compute per-layer full-sequence max/mean diffs.",
    )
    return parser


def _fmt(x: float) -> str:
    if math.isnan(x) or math.isinf(x):
        return str(x)
    if x == 0 or (1e-3 <= abs(x) < 1e4):
        return f"{x:.6f}"
    return f"{x:.6e}"


def _resolve_dtype(torch: Any, value: str) -> Any:
    if value == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def _load_dataset_token_ids(data_path: str, dataset_index: int) -> list[int]:
    from datasets import load_from_disk

    dataset = load_from_disk(data_path)
    item = dataset[int(dataset_index)]
    input_ids = item["input_ids"]
    if hasattr(input_ids, "tolist"):
        token_ids = input_ids.tolist()
    else:
        token_ids = list(input_ids)
    return [int(x) for x in token_ids]


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


def _request_vllm_variant(
    *,
    client: Any,
    model_id: str,
    prompt: list[int],
    prompt_logprobs: int | None,
    request_timeout: float,
    hidden_file_timeout: float,
) -> dict[str, Any]:
    from safetensors.torch import load_file
    from speculators.data_generation.vllm_client import (
        _kv_hidden_states_path,
        _prompt_logprobs,
        _prompt_token_ids,
    )

    extra_body = {"return_token_ids": True}
    if prompt_logprobs is not None:
        extra_body["prompt_logprobs"] = prompt_logprobs

    response = client.completions.create(
        model=model_id,
        prompt=prompt,
        max_tokens=1,
        extra_body=extra_body,
        timeout=request_timeout,
    )
    response_prompt_ids = _prompt_token_ids(response)
    hidden_path = _kv_hidden_states_path(response)
    response_prompt_logprobs = _prompt_logprobs(response)
    if response_prompt_ids is None:
        raise RuntimeError("vLLM response missing prompt_token_ids")
    if hidden_path is None:
        raise RuntimeError("vLLM response missing hidden_states_path")

    _wait_for_hidden_file(hidden_path, timeout=hidden_file_timeout)
    loaded = load_file(hidden_path)
    # Clone immediately so later requests/file cleanup cannot affect comparisons.
    hidden = loaded["hidden_states"].detach().cpu().clone()
    token_ids = loaded["token_ids"].detach().cpu().clone()
    return {
        "prompt_logprobs": prompt_logprobs,
        "prompt_ids": list(response_prompt_ids),
        "response_has_prompt_logprobs": response_prompt_logprobs is not None,
        "hidden_path": hidden_path,
        "file_token_ids": token_ids,
        "hidden": hidden,
    }


def _load_hf(args: argparse.Namespace, torch: Any) -> Any:
    from transformers import AutoModelForCausalLM

    dtype = _resolve_dtype(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(torch.device(args.device))
    model.eval()
    return model


def _run_hf_hidden(model: Any, torch: Any, prompt: list[int], device: str) -> tuple[Any, ...]:
    input_ids = torch.tensor([prompt], dtype=torch.long, device=torch.device(device))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    return tuple(x.detach().cpu().clone() for x in outputs.hidden_states)


def _diff_stats(a: Any, b: Any) -> tuple[float, float]:
    diff = (a.detach().float() - b.detach().float()).abs()
    return float(diff.max().item()), float(diff.mean().item())


def _print_run_summary(
    *,
    label: str,
    run: dict[str, Any],
    prompt: list[int],
    expected_shape: tuple[int, int, int],
) -> None:
    print(f"TRACE run.{label}")
    print(f"  prompt_logprobs={run['prompt_logprobs']}")
    print(f"  prompt_ids_match={run['prompt_ids'] == prompt}")
    print(f"  file_token_ids_match={run['file_token_ids'].tolist() == prompt}")
    print(f"  response_has_prompt_logprobs={run['response_has_prompt_logprobs']}")
    print(f"  hidden_path={run['hidden_path']}")
    print(f"  hidden_shape={tuple(run['hidden'].shape)}")
    print(f"  hidden_shape_ok={tuple(run['hidden'].shape) == expected_shape}")


def _print_pair_full(
    *,
    name: str,
    left: Any,
    right: Any,
    layer_ids: list[int],
) -> None:
    print(f"TRACE pair_full.{name}")
    print("  layer,slot,max_abs_diff,mean_abs_diff")
    for slot, layer_id in enumerate(layer_ids):
        max_diff, mean_diff = _diff_stats(left[:, slot], right[:, slot])
        print(f"  {layer_id},{slot},{_fmt(max_diff)},{_fmt(mean_diff)}")


def _print_pair_windows(
    *,
    name: str,
    left: Any,
    right: Any,
    layer_ids: list[int],
    prefix_lens: list[int],
    gt_len: int,
    prompt_len: int,
) -> None:
    print(f"TRACE pair_windows.{name}")
    print("  prefix_len,layer,slot,hidden_pos,max_abs_diff,mean_abs_diff")
    for prefix_len in prefix_lens:
        if prefix_len + gt_len > prompt_len:
            print(f"  {prefix_len},<skip_prefix_out_of_range>")
            continue
        for slot, layer_id in enumerate(layer_ids):
            for score_pos in range(prefix_len, prefix_len + gt_len):
                hidden_pos = score_pos - 1
                max_diff, mean_diff = _diff_stats(
                    left[hidden_pos, slot],
                    right[hidden_pos, slot],
                )
                print(
                    f"  {prefix_len},{layer_id},{slot},{hidden_pos},"
                    f"{_fmt(max_diff)},{_fmt(mean_diff)}"
                )


def _hf_connector_hidden(
    *,
    hf_hidden: tuple[Any, ...],
    layer_ids: list[int],
) -> Any:
    import torch

    return torch.stack([hf_hidden[layer_id][0] for layer_id in layer_ids], dim=1)


def main() -> None:
    args = _parser().parse_args()

    import torch
    from transformers import AutoConfig

    token_ids = _load_dataset_token_ids(args.data_path, args.dataset_index)
    prompt_len = len(token_ids) if args.prompt_len <= 0 else min(args.prompt_len, len(token_ids))
    prompt = token_ids[:prompt_len]

    config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if hasattr(config, "text_config"):
        config = config.text_config
    final_layer_id = (
        int(args.final_layer_id)
        if args.final_layer_id is not None
        else int(config.num_hidden_layers)
    )
    layer_ids = list(args.target_layer_ids)
    if final_layer_id not in layer_ids:
        layer_ids.append(final_layer_id)
    expected_shape = (len(prompt), len(layer_ids), int(config.hidden_size))

    print("TRACE config")
    print(f"  repo={ROOT}")
    print(f"  model_path={args.model_path}")
    print(f"  data_path={args.data_path}")
    print(f"  dataset_index={args.dataset_index}")
    print(f"  item_len={len(token_ids)}")
    print(f"  prompt_len={len(prompt)}")
    print(f"  vllm_endpoint={args.vllm_endpoint}")
    print(f"  device={args.device}")
    print(f"  dtype={args.dtype}")
    print(f"  layer_ids={layer_ids}")
    print(f"  expected_shape={expected_shape}")
    print(f"  prefix_lens={args.prefix_lens}")
    print(f"  gt_len={args.gt_len}")

    client, model_id = _make_openai_client(args.vllm_endpoint)
    print("TRACE vllm")
    print(f"  model_id={model_id}")

    runs: dict[str, dict[str, Any]] = {}
    for label, prompt_logprobs in (
        ("H0a", None),
        ("H0b", None),
        ("H1a", int(args.prompt_logprobs)),
        ("H1b", int(args.prompt_logprobs)),
    ):
        runs[label] = _request_vllm_variant(
            client=client,
            model_id=model_id,
            prompt=prompt,
            prompt_logprobs=prompt_logprobs,
            request_timeout=args.request_timeout,
            hidden_file_timeout=args.hidden_file_timeout,
        )
        _print_run_summary(
            label=label,
            run=runs[label],
            prompt=prompt,
            expected_shape=expected_shape,
        )

    hf_model = _load_hf(args, torch)
    hf_hidden = _run_hf_hidden(hf_model, torch, prompt, args.device)
    hf_connector = _hf_connector_hidden(hf_hidden=hf_hidden, layer_ids=layer_ids)
    print("TRACE hf")
    print(f"  hidden_count={len(hf_hidden)}")
    print(f"  hidden_shapes={[tuple(x.shape) for x in hf_hidden]}")
    print(f"  connector_hidden_shape={tuple(hf_connector.shape)}")

    pairs = (
        ("H0a_vs_H0b", runs["H0a"]["hidden"], runs["H0b"]["hidden"]),
        ("H1a_vs_H1b", runs["H1a"]["hidden"], runs["H1b"]["hidden"]),
        ("H0a_vs_H1a", runs["H0a"]["hidden"], runs["H1a"]["hidden"]),
        ("H0b_vs_H1b", runs["H0b"]["hidden"], runs["H1b"]["hidden"]),
        ("H0a_vs_HF", runs["H0a"]["hidden"], hf_connector),
        ("H0b_vs_HF", runs["H0b"]["hidden"], hf_connector),
        ("H1a_vs_HF", runs["H1a"]["hidden"], hf_connector),
        ("H1b_vs_HF", runs["H1b"]["hidden"], hf_connector),
    )
    for name, left, right in pairs:
        if args.full_sequence:
            _print_pair_full(
                name=name,
                left=left,
                right=right,
                layer_ids=layer_ids,
            )
        _print_pair_windows(
            name=name,
            left=left,
            right=right,
            layer_ids=layer_ids,
            prefix_lens=args.prefix_lens,
            gt_len=args.gt_len,
            prompt_len=len(prompt),
        )

    print("TRACE conclusion_hints")
    print("  H0a_vs_H0b checks hidden-only repeat stability.")
    print("  H1a_vs_H1b checks prompt_logprobs repeat stability.")
    print("  H0*_vs_H1* isolates the prompt_logprobs request-body field.")
    print("  H0*_vs_HF and H1*_vs_HF identify which path matches direct HF.")

    if not args.keep_hidden_files:
        for run in runs.values():
            _delete_hidden_file(run["hidden_path"])


if __name__ == "__main__":
    main()
