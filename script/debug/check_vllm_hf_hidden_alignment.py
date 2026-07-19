#!/usr/bin/env python3
"""Compare vLLM connector hidden states with direct HF verifier hidden states.

This is a focused target-only check. It bypasses DSpark training and sends the
same token prompt to:

1. vLLM OpenAI completions, to collect prompt logprobs and connector safetensors.
2. Transformers AutoModelForCausalLM, to collect logits and hidden_states.

It also repeats the vLLM request to detect connector instability for identical
token_ids.
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
        description="Check vLLM connector hidden states against direct HF."
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
    parser.add_argument("--target-layer-ids", type=int, nargs="+", default=[1, 9, 17, 25, 33])
    parser.add_argument("--final-layer-id", type=int, default=None)
    parser.add_argument("--prefix-len", type=int, default=68)
    parser.add_argument("--gt-len", type=int, default=7)
    parser.add_argument(
        "--raw-max-len",
        type=int,
        default=0,
        help="0 means use the full dataset item for the raw_doc case.",
    )
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument(
        "--warmup-generate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "First issue a dataloader-style hidden-only raw-doc request "
            "without prompt_logprobs, then compare later requests against it."
        ),
    )
    parser.add_argument("--prompt-logprobs", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--hidden-file-timeout", type=float, default=30.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--keep-hidden-files", action="store_true")
    parser.add_argument("--print-token-window", type=int, default=4)
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


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        piece = tokenizer.convert_ids_to_tokens([token_id])[0]
    except Exception:  # noqa: BLE001
        piece = "<convert_error>"
    try:
        text = tokenizer.decode([token_id])
    except Exception:  # noqa: BLE001
        text = "<decode_error>"
    return f"{piece!r} text={text!r}"


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


def _request_vllm(
    *,
    client: Any,
    model_id: str,
    prompt: list[int],
    score_positions: list[int],
    prompt_logprobs: int,
    request_timeout: float,
    hidden_file_timeout: float,
) -> dict[str, Any]:
    from safetensors.torch import load_file
    from speculators.data_generation.vllm_client import (
        _extract_token_logprob,
        _kv_hidden_states_path,
        _prompt_logprobs,
        _prompt_token_ids,
    )

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
    if response_prompt_ids is None:
        raise RuntimeError("vLLM response missing prompt_token_ids")
    if response_prompt_logprobs is None:
        raise RuntimeError("vLLM response missing prompt_logprobs")
    hidden_path = _kv_hidden_states_path(response)
    if hidden_path is None:
        raise RuntimeError("vLLM response missing hidden_states_path")

    _wait_for_hidden_file(hidden_path, timeout=hidden_file_timeout)
    loaded = load_file(hidden_path)
    token_logprobs = [
        _extract_token_logprob(response_prompt_logprobs, pos, prompt[pos])
        for pos in score_positions
    ]
    return {
        "prompt_ids": response_prompt_ids,
        "hidden_path": hidden_path,
        "file_token_ids": loaded["token_ids"].detach().cpu().tolist(),
        "hidden": loaded["hidden_states"],
        "token_logprobs": token_logprobs,
    }


def _request_vllm_hidden_only(
    *,
    client: Any,
    model_id: str,
    prompt: list[int],
    request_timeout: float,
    hidden_file_timeout: float,
) -> dict[str, Any]:
    from safetensors.torch import load_file
    from speculators.data_generation.vllm_client import (
        _kv_hidden_states_path,
        _prompt_token_ids,
    )

    response = client.completions.create(
        model=model_id,
        prompt=prompt,
        max_tokens=1,
        extra_body={"return_token_ids": True},
        timeout=request_timeout,
    )
    response_prompt_ids = _prompt_token_ids(response)
    if response_prompt_ids is None:
        raise RuntimeError("vLLM hidden-only response missing prompt_token_ids")
    hidden_path = _kv_hidden_states_path(response)
    if hidden_path is None:
        raise RuntimeError("vLLM hidden-only response missing hidden_states_path")

    _wait_for_hidden_file(hidden_path, timeout=hidden_file_timeout)
    loaded = load_file(hidden_path)
    return {
        "prompt_ids": response_prompt_ids,
        "hidden_path": hidden_path,
        "file_token_ids": loaded["token_ids"].detach().cpu().tolist(),
        "hidden": loaded["hidden_states"],
    }


def _load_hf_model(args: argparse.Namespace, torch: Any) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _resolve_dtype(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(torch.device(args.device))
    model.eval()
    return model, tokenizer


def _run_hf(model: Any, torch: Any, prompt: list[int], device: str) -> Any:
    input_ids = torch.tensor([prompt], dtype=torch.long, device=torch.device(device))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )


def _gather_hf_logprobs(
    *,
    torch: Any,
    logits: Any,
    prompt: list[int],
    score_positions: list[int],
) -> list[float]:
    pred_positions = torch.tensor(
        [pos - 1 for pos in score_positions],
        dtype=torch.long,
        device=logits.device,
    )
    target_ids = torch.tensor(
        [prompt[pos] for pos in score_positions],
        dtype=torch.long,
        device=logits.device,
    )
    with torch.no_grad():
        logprobs = torch.log_softmax(logits[0, pred_positions].float(), dim=-1)
        gathered = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return [float(x) for x in gathered.detach().cpu().tolist()]


def _project_vllm_final(
    *,
    torch: Any,
    hf_model: Any,
    final_hidden: Any,
    prompt: list[int],
    score_positions: list[int],
    apply_norm: bool,
) -> list[float]:
    device = next(hf_model.parameters()).device
    hidden = final_hidden.to(device=device)
    if apply_norm:
        hidden = hf_model.model.norm(hidden)
    logits = hf_model.lm_head(hidden).float()
    pred_positions = torch.tensor(
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
        logprobs = torch.log_softmax(logits[pred_positions], dim=-1)
        gathered = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return [float(x) for x in gathered.detach().cpu().tolist()]


def _max_abs(values: list[float]) -> str:
    return _fmt(max(abs(x) for x in values)) if values else "<empty>"


def _print_token_window(
    *,
    tokenizer: Any,
    prompt: list[int],
    score_positions: list[int],
    window: int,
) -> None:
    shown = set()
    for pos in score_positions:
        for p in range(pos - window, pos + window + 1):
            if 0 <= p < len(prompt):
                shown.add(p)
    print("  token_window pos,token_id,predicts,next_token,token")
    for pos in sorted(shown):
        next_id = prompt[pos + 1] if pos + 1 < len(prompt) else None
        predicts = "yes" if pos + 1 in score_positions else "no"
        print(
            f"  {pos},{prompt[pos]},{predicts},{next_id},"
            f"{_decode_token(tokenizer, prompt[pos])}"
        )


def _compare_case(
    *,
    args: argparse.Namespace,
    torch: Any,
    tokenizer: Any,
    hf_model: Any,
    client: Any,
    model_id: str,
    name: str,
    prompt: list[int],
    score_positions: list[int],
    connector_layer_ids: list[int],
    reference_hidden: dict[str, Any] | None = None,
    reference_label: str = "reference",
) -> None:
    from transformers import AutoConfig

    print(f"TRACE case.{name}.setup")
    print(f"  prompt_len={len(prompt)}")
    print(f"  score_positions={score_positions}")
    print(f"  hidden_positions={[pos - 1 for pos in score_positions]}")
    print(f"  target_ids={[prompt[pos] for pos in score_positions]}")
    print(f"  connector_layer_ids={connector_layer_ids}")
    _print_token_window(
        tokenizer=tokenizer,
        prompt=prompt,
        score_positions=score_positions,
        window=args.print_token_window,
    )

    hf_outputs = _run_hf(hf_model, torch, prompt, args.device)
    hf_shapes = [tuple(x.shape) for x in hf_outputs.hidden_states]
    hf_logprobs = _gather_hf_logprobs(
        torch=torch,
        logits=hf_outputs.logits,
        prompt=prompt,
        score_positions=score_positions,
    )
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    hidden_size = int(getattr(config, "hidden_size", getattr(config, "text_config", config).hidden_size))

    print(f"TRACE case.{name}.hf")
    print(f"  logits_shape={tuple(hf_outputs.logits.shape)}")
    print(f"  hidden_count={len(hf_outputs.hidden_states)}")
    print(f"  hidden_shapes={hf_shapes}")
    print(f"  expected_hidden_size={hidden_size}")
    print(f"  hf_logits_logprobs={[_fmt(x) for x in hf_logprobs]}")

    if reference_hidden is not None:
        print(f"TRACE case.{name}.{reference_label}_compare")
        print(f"  prompt_ids_match={reference_hidden['prompt_ids'] == prompt}")
        print(f"  file_token_ids_match={reference_hidden['file_token_ids'] == prompt}")
        print(f"  hidden_path={reference_hidden['hidden_path']}")
        print(f"  hidden_shape={tuple(reference_hidden['hidden'].shape)}")
        print(
            "  layer,slot,hidden_pos,ref_vs_hf_max_abs,ref_vs_hf_mean_abs,"
            "ref_normed_vs_hf_max_abs,ref_normed_vs_hf_mean_abs"
        )
        final_layer_id = connector_layer_ids[-1]
        for slot, layer_id in enumerate(connector_layer_ids):
            for score_pos in score_positions:
                hidden_pos = score_pos - 1
                ref_vec = reference_hidden["hidden"][
                    hidden_pos,
                    slot,
                ].detach().float().cpu()
                hf_vec = hf_outputs.hidden_states[
                    layer_id
                ][0, hidden_pos].detach().float().cpu()
                raw_diff = (ref_vec - hf_vec).abs()
                normed_max = "<na>"
                normed_mean = "<na>"
                if layer_id == final_layer_id:
                    device = next(hf_model.parameters()).device
                    normed = hf_model.model.norm(
                        reference_hidden["hidden"][hidden_pos, slot].to(device=device)
                    ).detach().float().cpu()
                    normed_diff = (normed - hf_vec).abs()
                    normed_max = _fmt(float(normed_diff.max().item()))
                    normed_mean = _fmt(float(normed_diff.mean().item()))
                print(
                    f"  {layer_id},{slot},{hidden_pos},"
                    f"{_fmt(float(raw_diff.max().item()))},"
                    f"{_fmt(float(raw_diff.mean().item()))},"
                    f"{normed_max},{normed_mean}"
                )

    runs = []
    for run_idx in range(max(args.repeat, 1)):
        run = _request_vllm(
            client=client,
            model_id=model_id,
            prompt=prompt,
            score_positions=score_positions,
            prompt_logprobs=args.prompt_logprobs,
            request_timeout=args.request_timeout,
            hidden_file_timeout=args.hidden_file_timeout,
        )
        runs.append(run)
        shape = tuple(run["hidden"].shape)
        shape_ok = (
            len(shape) == 3
            and shape[0] == len(prompt)
            and shape[1] == len(connector_layer_ids)
            and shape[2] == hidden_size
        )
        print(f"TRACE case.{name}.vllm_run{run_idx}")
        print(f"  prompt_ids_match={run['prompt_ids'] == prompt}")
        print(f"  file_token_ids_match={run['file_token_ids'] == prompt}")
        print(f"  hidden_path={run['hidden_path']}")
        print(f"  hidden_shape={shape}")
        print(f"  hidden_shape_ok={shape_ok}")
        print(f"  prompt_logprobs={[_fmt(x) for x in run['token_logprobs']]}")

    print(f"TRACE case.{name}.logprob_compare")
    print("  gt_index,score_pos,target_id,hf_logits,vllm_prompt,diff")
    for i, score_pos in enumerate(score_positions):
        diff = runs[0]["token_logprobs"][i] - hf_logprobs[i]
        print(
            f"  {i + 1},{score_pos},{prompt[score_pos]},"
            f"{_fmt(hf_logprobs[i])},"
            f"{_fmt(runs[0]['token_logprobs'][i])},"
            f"{_fmt(diff)}"
        )
    print(
        "  hf_logits_vs_vllm_prompt_max_abs_diff="
        f"{_max_abs([a - b for a, b in zip(hf_logprobs, runs[0]['token_logprobs'], strict=True)])}"
    )

    print(f"TRACE case.{name}.hidden_compare_vllm0_vs_hf")
    print("  layer,slot,hidden_pos,raw_max_abs,raw_mean_abs,normed_max_abs,normed_mean_abs")
    final_layer_id = connector_layer_ids[-1]
    for slot, layer_id in enumerate(connector_layer_ids):
        for score_pos in score_positions:
            hidden_pos = score_pos - 1
            vllm_vec = runs[0]["hidden"][hidden_pos, slot].detach().float().cpu()
            hf_vec = hf_outputs.hidden_states[layer_id][0, hidden_pos].detach().float().cpu()
            raw_diff = (vllm_vec - hf_vec).abs()
            normed_max = "<na>"
            normed_mean = "<na>"
            if layer_id == final_layer_id:
                device = next(hf_model.parameters()).device
                normed = hf_model.model.norm(
                    runs[0]["hidden"][hidden_pos, slot].to(device=device)
                ).detach().float().cpu()
                normed_diff = (normed - hf_vec).abs()
                normed_max = _fmt(float(normed_diff.max().item()))
                normed_mean = _fmt(float(normed_diff.mean().item()))
            print(
                f"  {layer_id},{slot},{hidden_pos},"
                f"{_fmt(float(raw_diff.max().item()))},"
                f"{_fmt(float(raw_diff.mean().item()))},"
                f"{normed_max},{normed_mean}"
            )

    final_slot = len(connector_layer_ids) - 1
    vllm_final = runs[0]["hidden"][:, final_slot]
    vllm_no_norm = _project_vllm_final(
        torch=torch,
        hf_model=hf_model,
        final_hidden=vllm_final,
        prompt=prompt,
        score_positions=score_positions,
        apply_norm=False,
    )
    vllm_with_norm = _project_vllm_final(
        torch=torch,
        hf_model=hf_model,
        final_hidden=vllm_final,
        prompt=prompt,
        score_positions=score_positions,
        apply_norm=True,
    )
    print(f"TRACE case.{name}.vllm_final_projection")
    print(f"  no_norm_logprobs={[_fmt(x) for x in vllm_no_norm]}")
    print(f"  with_norm_logprobs={[_fmt(x) for x in vllm_with_norm]}")
    print(
        "  with_norm_vs_vllm_prompt_max_abs_diff="
        f"{_max_abs([a - b for a, b in zip(vllm_with_norm, runs[0]['token_logprobs'], strict=True)])}"
    )

    if len(runs) > 1:
        print(f"TRACE case.{name}.vllm_repeat_diff")
        print("  run,layer,slot,hidden_pos,max_abs_diff,mean_abs_diff")
        base = runs[0]["hidden"]
        for run_idx, run in enumerate(runs[1:], start=1):
            if tuple(run["hidden"].shape) != tuple(base.shape):
                print(
                    f"  {run_idx},<shape_mismatch>,"
                    f"{tuple(base.shape)}!={tuple(run['hidden'].shape)}"
                )
                continue
            for slot, layer_id in enumerate(connector_layer_ids):
                for score_pos in score_positions:
                    hidden_pos = score_pos - 1
                    diff = (
                        base[hidden_pos, slot].detach().float().cpu()
                        - run["hidden"][hidden_pos, slot].detach().float().cpu()
                    ).abs()
                    print(
                        f"  {run_idx},{layer_id},{slot},{hidden_pos},"
                        f"{_fmt(float(diff.max().item()))},"
                        f"{_fmt(float(diff.mean().item()))}"
                    )

    if reference_hidden is not None:
        print(f"TRACE case.{name}.{reference_label}_vs_vllm_run0")
        print("  layer,slot,hidden_pos,max_abs_diff,mean_abs_diff")
        ref = reference_hidden["hidden"]
        cur = runs[0]["hidden"]
        if tuple(ref.shape) != tuple(cur.shape):
            print(f"  <shape_mismatch>,{tuple(ref.shape)}!={tuple(cur.shape)}")
        else:
            for slot, layer_id in enumerate(connector_layer_ids):
                for score_pos in score_positions:
                    hidden_pos = score_pos - 1
                    diff = (
                        ref[hidden_pos, slot].detach().float().cpu()
                        - cur[hidden_pos, slot].detach().float().cpu()
                    ).abs()
                    print(
                        f"  {layer_id},{slot},{hidden_pos},"
                        f"{_fmt(float(diff.max().item()))},"
                        f"{_fmt(float(diff.mean().item()))}"
                    )

    if not args.keep_hidden_files:
        for run in runs:
            _delete_hidden_file(run["hidden_path"])
        if reference_hidden is not None:
            _delete_hidden_file(reference_hidden["hidden_path"])


def main() -> None:
    args = _parser().parse_args()

    import torch
    from transformers import AutoConfig

    token_ids = _load_dataset_token_ids(args.data_path, args.dataset_index)
    if args.prefix_len + args.gt_len > len(token_ids):
        raise ValueError(
            f"prefix_len + gt_len exceeds item length: "
            f"{args.prefix_len}+{args.gt_len}>{len(token_ids)}"
        )

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
    connector_layer_ids = list(args.target_layer_ids)
    if final_layer_id not in connector_layer_ids:
        connector_layer_ids.append(final_layer_id)

    print("TRACE config")
    print(f"  repo={ROOT}")
    print(f"  model_path={args.model_path}")
    print(f"  data_path={args.data_path}")
    print(f"  dataset_index={args.dataset_index}")
    print(f"  item_len={len(token_ids)}")
    print(f"  vllm_endpoint={args.vllm_endpoint}")
    print(f"  device={args.device}")
    print(f"  dtype={args.dtype}")
    print(f"  target_layer_ids={args.target_layer_ids}")
    print(f"  final_layer_id={final_layer_id}")
    print(f"  connector_layer_ids={connector_layer_ids}")
    print(f"  prefix_len={args.prefix_len}")
    print(f"  gt_len={args.gt_len}")
    print(f"  repeat={args.repeat}")
    print(f"  warmup_generate={args.warmup_generate}")

    client, model_id = _make_openai_client(args.vllm_endpoint)
    print("TRACE vllm")
    print(f"  model_id={model_id}")

    hf_model, tokenizer = _load_hf_model(args, torch)

    warmup_hidden = None
    raw_len = len(token_ids) if args.raw_max_len <= 0 else min(args.raw_max_len, len(token_ids))
    if args.warmup_generate:
        warmup_prompt = token_ids[:raw_len]
        warmup_hidden = _request_vllm_hidden_only(
            client=client,
            model_id=model_id,
            prompt=warmup_prompt,
            request_timeout=args.request_timeout,
            hidden_file_timeout=args.hidden_file_timeout,
        )
        print("TRACE warmup_generate")
        print(f"  prompt_len={len(warmup_prompt)}")
        print(f"  prompt_ids_match={warmup_hidden['prompt_ids'] == warmup_prompt}")
        print(f"  file_token_ids_match={warmup_hidden['file_token_ids'] == warmup_prompt}")
        print(f"  hidden_path={warmup_hidden['hidden_path']}")
        print(f"  hidden_shape={tuple(warmup_hidden['hidden'].shape)}")

    doc_prompt_len = args.prefix_len + args.gt_len
    doc_prompt = token_ids[:doc_prompt_len]
    doc_score_positions = list(range(args.prefix_len, doc_prompt_len))
    _compare_case(
        args=args,
        torch=torch,
        tokenizer=tokenizer,
        hf_model=hf_model,
        client=client,
        model_id=model_id,
        name="doc_prefix",
        prompt=doc_prompt,
        score_positions=doc_score_positions,
        connector_layer_ids=connector_layer_ids,
    )

    if raw_len >= doc_prompt_len:
        raw_prompt = token_ids[:raw_len]
        raw_score_positions = list(range(args.prefix_len, args.prefix_len + args.gt_len))
        _compare_case(
            args=args,
            torch=torch,
            tokenizer=tokenizer,
            hf_model=hf_model,
            client=client,
            model_id=model_id,
            name="raw_doc",
            prompt=raw_prompt,
            score_positions=raw_score_positions,
            connector_layer_ids=connector_layer_ids,
            reference_hidden=warmup_hidden,
            reference_label="warmup_generate",
        )
    else:
        print("TRACE case.raw_doc skipped: raw_len shorter than prefix+gt")


if __name__ == "__main__":
    main()
