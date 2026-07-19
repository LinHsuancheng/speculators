#!/usr/bin/env python3
"""Four-way DSpark hidden-state pipeline alignment check.

This is a narrow debug script for the Qwen3-4B DSpark hidden mismatch around
dataset index 45760. It compares one document-local window through four paths:

A. fresh vLLM connector hidden -> verifier final norm -> full lm_head
B. raw dataset item hidden as returned by the DSpark dataloader dataset
C. collated/packed batch hidden slice
D. vLLM prompt_logprobs for the same prompt positions

The goal is to separate connector/request issues from post-generation split,
collate, and packing issues.
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
        description="Check DSpark raw/collate/fresh hidden-state alignment."
    )
    parser.add_argument("--model-path", default="/models/Qwen3-4B")
    parser.add_argument("--data-path", default="/data/open_perfectblend_qwen3_4b_100k")
    parser.add_argument("--hidden-states-path", default=None)
    parser.add_argument("--dataset-index", type=int, default=45760)
    parser.add_argument("--vllm-endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--device", default="npu:15")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--hidden-states-dtype", default="bfloat16")
    parser.add_argument("--target-layer-ids", type=int, nargs="+", default=[1, 9, 17, 25, 33])
    parser.add_argument("--final-layer-id", type=int, default=None)
    parser.add_argument("--local-start", type=int, default=67)
    parser.add_argument("--gt-len", type=int, default=7)
    parser.add_argument(
        "--raw-max-len",
        type=int,
        default=0,
        help="0 means use the full dataset item for the fresh vLLM request.",
    )
    parser.add_argument("--total-seq-len", type=int, default=3072)
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Default 0 avoids hiding packing bugs behind train-time hidden noise.",
    )
    parser.add_argument("--on-missing", choices=["generate", "skip", "warn", "raise"], default="generate")
    parser.add_argument("--on-generate", choices=["cache", "delete"], default="delete")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--hidden-file-timeout", type=float, default=30.0)
    parser.add_argument("--prompt-logprobs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--keep-hidden-files", action="store_true")
    parser.add_argument("--hidden-tol", type=float, default=1e-2)
    parser.add_argument("--logprob-tol", type=float, default=0.5)
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


def _diff_stats(a: Any, b: Any) -> tuple[float, float]:
    diff = (a.detach().float().cpu() - b.detach().float().cpu()).abs()
    return float(diff.max().item()), float(diff.mean().item())


def _max_abs_delta(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b, strict=True)) if a else float("nan")


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


def _request_vllm_with_hidden(
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
    hidden_path = _kv_hidden_states_path(response)
    if response_prompt_ids is None:
        raise RuntimeError("vLLM response missing prompt_token_ids")
    if response_prompt_logprobs is None:
        raise RuntimeError("vLLM response missing prompt_logprobs")
    if hidden_path is None:
        raise RuntimeError("vLLM response missing hidden_states_path")

    _wait_for_hidden_file(hidden_path, timeout=hidden_file_timeout)
    loaded = load_file(hidden_path)
    token_logprobs = [
        _extract_token_logprob(response_prompt_logprobs, pos, prompt[pos])
        for pos in score_positions
    ]
    return {
        "prompt_ids": list(response_prompt_ids),
        "hidden_path": hidden_path,
        "file_token_ids": loaded["token_ids"].detach().cpu().tolist(),
        "hidden": loaded["hidden_states"].detach().cpu().clone(),
        "token_logprobs": token_logprobs,
    }


def _load_hf_model(args: argparse.Namespace, torch: Any) -> Any:
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


def _project_full_lm_head(
    *,
    torch: Any,
    hf_model: Any,
    final_hidden: Any,
    prompt: list[int],
    score_positions: list[int],
) -> list[float]:
    device = next(hf_model.parameters()).device
    hidden = final_hidden.to(device=device)
    hidden = hf_model.model.norm(hidden)
    logits = hf_model.lm_head(hidden).float()
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
        logprobs = torch.log_softmax(logits[hidden_positions], dim=-1)
        gathered = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return [float(x) for x in gathered.detach().cpu().tolist()]


def _batch_indices_for_loader(train_loader: Any) -> list[list[int]]:
    batches = list(train_loader.batch_sampler)
    out: list[list[int]] = []
    for batch in batches:
        if hasattr(batch, "tolist"):
            out.append([int(x) for x in batch.tolist()])
        else:
            out.append([int(x) for x in batch])
    return out


def _find_target_batch(train_loader: Any, dataset_index: int) -> tuple[int, int, list[int]]:
    for batch_index, indices in enumerate(_batch_indices_for_loader(train_loader)):
        if dataset_index in indices:
            return batch_index, indices.index(dataset_index), indices
    raise RuntimeError(f"Dataset index {dataset_index} was not found in train batches")


def _clone_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().cpu().clone() if hasattr(value, "detach") else value
        for key, value in item.items()
    }


def _clone_item_with_source(item: dict[str, Any], source: Any) -> dict[str, Any]:
    cloned = _clone_item(item)
    if source is not None:
        cloned["_hidden_state_source"] = dict(source)
    return cloned


def _load_batch_and_raw_item(
    *,
    args: argparse.Namespace,
    torch: Any,
    hidden_size: int,
    num_target_layers: int,
) -> tuple[Any, dict[str, Any], dict[str, Any], int, int, list[int]]:
    from speculators.train.dataloader import create_train_val_loaders

    hidden_states_dtype = getattr(torch, args.hidden_states_dtype)
    captured_items: list[dict[str, Any]] = []

    def capture(item: dict[str, Any]) -> dict[str, Any]:
        source = item.pop("_hidden_state_source", None)
        captured_items.append(_clone_item_with_source(item, source))
        return item

    train_loader, _ = create_train_val_loaders(
        data_path=args.data_path,
        total_seq_len=args.total_seq_len,
        hidden_states_dtype=hidden_states_dtype,
        noise_std=args.noise_std,
        legacy_data=False,
        hidden_states_path=args.hidden_states_path,
        vllm_endpoint=args.vllm_endpoint,
        on_missing=args.on_missing,
        on_generate=args.on_generate,
        verifier_name_or_path=args.model_path,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        hidden_size=hidden_size,
        num_target_layers=num_target_layers,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        preprocess=capture,
    )

    batch_index, doc_id, indices = _find_target_batch(train_loader, args.dataset_index)
    if args.num_workers != 0:
        raise ValueError("This debug script requires --num-workers 0 for raw capture order")

    batch = None
    for idx, candidate in enumerate(train_loader):
        if idx == batch_index:
            batch = candidate
            break
    if batch is None:
        raise RuntimeError(f"Failed to read target batch {batch_index}")

    offset = sum(len(x) for x in _batch_indices_for_loader(train_loader)[:batch_index])
    capture_index = offset + doc_id
    if capture_index >= len(captured_items):
        raise RuntimeError(
            f"Captured raw item missing: capture_index={capture_index}, "
            f"captured={len(captured_items)}"
        )
    return train_loader, batch, captured_items[capture_index], batch_index, doc_id, indices


def _packed_doc_start(batch: dict[str, Any], doc_id: int) -> int:
    document_ids = batch["document_ids"][0]
    positions = (document_ids == doc_id).nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        raise RuntimeError(f"doc_id {doc_id} not present in packed batch")
    return int(positions[0].item())


def _print_logprob_table(
    *,
    score_positions: list[int],
    target_ids: list[int],
    fresh: list[float],
    raw: list[float],
    batch: list[float],
    vllm: list[float],
) -> None:
    print("TRACE logprob_table")
    print("  local_score_pos,target_id,A_fresh,D_vllm,A_minus_D,B_raw,B_minus_A,C_batch,C_minus_B")
    for i, pos in enumerate(score_positions):
        print(
            f"  {pos},"
            f"{target_ids[i]},"
            f"{_fmt(fresh[i])},"
            f"{_fmt(vllm[i])},"
            f"{_fmt(fresh[i] - vllm[i])},"
            f"{_fmt(raw[i])},"
            f"{_fmt(raw[i] - fresh[i])},"
            f"{_fmt(batch[i])},"
            f"{_fmt(batch[i] - raw[i])}"
        )


def main() -> None:
    args = _parser().parse_args()

    import torch
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if hasattr(config, "text_config"):
        config = config.text_config
    hidden_size = int(config.hidden_size)
    final_layer_id = (
        int(args.final_layer_id)
        if args.final_layer_id is not None
        else int(config.num_hidden_layers)
    )
    connector_layer_ids = list(args.target_layer_ids)
    if final_layer_id not in connector_layer_ids:
        connector_layer_ids.append(final_layer_id)
    num_target_layers = len(args.target_layer_ids)

    print("TRACE config")
    print(f"  repo={ROOT}")
    print(f"  model_path={args.model_path}")
    print(f"  data_path={args.data_path}")
    print(f"  hidden_states_path={args.hidden_states_path}")
    print(f"  dataset_index={args.dataset_index}")
    print(f"  vllm_endpoint={args.vllm_endpoint}")
    print(f"  device={args.device}")
    print(f"  dtype={args.dtype}")
    print(f"  hidden_states_dtype={args.hidden_states_dtype}")
    print(f"  target_layer_ids={args.target_layer_ids}")
    print(f"  final_layer_id={final_layer_id}")
    print(f"  assumed_connector_layer_ids={connector_layer_ids}")
    print(f"  local_start={args.local_start}")
    print(f"  gt_len={args.gt_len}")
    print(f"  total_seq_len={args.total_seq_len}")
    print(f"  noise_std={args.noise_std}")
    print(f"  on_missing={args.on_missing}")
    print(f"  on_generate={args.on_generate}")

    _, batch, raw_item, batch_index, doc_id, batch_indices = _load_batch_and_raw_item(
        args=args,
        torch=torch,
        hidden_size=hidden_size,
        num_target_layers=num_target_layers,
    )
    doc_start = _packed_doc_start(batch, doc_id)
    packed_hidden_start = doc_start + args.local_start
    packed_hidden_end = packed_hidden_start + args.gt_len
    raw_hidden_end = args.local_start + args.gt_len
    score_positions = list(range(args.local_start + 1, raw_hidden_end + 1))

    raw_input_ids = raw_item["input_ids"].detach().cpu().tolist()
    if raw_hidden_end + 1 > len(raw_input_ids):
        raise ValueError(
            f"Requested local window exceeds raw item length: {raw_hidden_end + 1}>{len(raw_input_ids)}"
        )
    if packed_hidden_end + 1 > int(batch["input_ids"].shape[1]):
        raise ValueError(
            f"Requested packed window exceeds batch length: {packed_hidden_end + 1}>{batch['input_ids'].shape[1]}"
        )
    packed_doc_ids = batch["document_ids"][0, packed_hidden_start:packed_hidden_end].detach().cpu()
    if not bool((packed_doc_ids == doc_id).all().item()):
        raise RuntimeError(
            "Packed comparison window crosses a document boundary: "
            f"doc_ids={packed_doc_ids.tolist()}, expected={doc_id}"
        )

    prompt_len = len(raw_input_ids) if args.raw_max_len <= 0 else min(args.raw_max_len, len(raw_input_ids))
    min_prompt_len = args.local_start + args.gt_len + 1
    if prompt_len < min_prompt_len:
        raise ValueError(f"fresh prompt length {prompt_len} is shorter than {min_prompt_len}")
    prompt = raw_input_ids[:prompt_len]

    print("TRACE batch_locator")
    print(f"  batch_index={batch_index}")
    print(f"  batch_indices={batch_indices}")
    print(f"  doc_id={doc_id}")
    print(f"  doc_start={doc_start}")
    print(f"  packed_hidden_window={packed_hidden_start}:{packed_hidden_end}")
    print(f"  local_hidden_window={args.local_start}:{raw_hidden_end}")
    print(f"  local_score_positions={score_positions}")
    print(f"  raw_item_len={len(raw_input_ids)}")
    print(f"  fresh_prompt_len={len(prompt)}")
    print(f"  raw_hidden_shape={tuple(raw_item['hidden_states'].shape)}")
    print(f"  raw_last_shape={tuple(raw_item['verifier_last_hidden_states'].shape)}")
    print(f"  batch_hidden_shape={tuple(batch['hidden_states'].shape)}")
    print(f"  batch_last_shape={tuple(batch['verifier_last_hidden_states'].shape)}")
    print(f"  hidden_state_source={raw_item.get('_hidden_state_source')}")

    packed_tokens = batch["input_ids"][0, packed_hidden_start:packed_hidden_end + 1].detach().cpu().tolist()
    raw_tokens = raw_input_ids[args.local_start:raw_hidden_end + 1]
    print("TRACE token_alignment")
    print(f"  raw_tokens_{args.local_start}_{raw_hidden_end}={raw_tokens}")
    print(f"  packed_tokens_{packed_hidden_start}_{packed_hidden_end}={packed_tokens}")
    print(f"  packed_matches_raw={packed_tokens == raw_tokens}")

    client, model_id = _make_openai_client(args.vllm_endpoint)
    print("TRACE vllm")
    print(f"  model_id={model_id}")
    fresh_run = _request_vllm_with_hidden(
        client=client,
        model_id=model_id,
        prompt=prompt,
        score_positions=score_positions,
        prompt_logprobs=args.prompt_logprobs,
        request_timeout=args.request_timeout,
        hidden_file_timeout=args.hidden_file_timeout,
    )
    compare_len = args.local_start + args.gt_len + 1
    prompt_ids_prefix_match = fresh_run["prompt_ids"][:compare_len] == raw_input_ids[:compare_len]
    file_ids_prefix_match = fresh_run["file_token_ids"][:compare_len] == raw_input_ids[:compare_len]
    if not prompt_ids_prefix_match or not file_ids_prefix_match:
        raise RuntimeError(
            "fresh vLLM token ids do not match raw input prefix: "
            f"prompt_match={prompt_ids_prefix_match}, file_match={file_ids_prefix_match}"
        )

    print("TRACE fresh_vllm")
    print(f"  hidden_path={fresh_run['hidden_path']}")
    print(f"  prompt_ids_prefix_match_{compare_len}={prompt_ids_prefix_match}")
    print(f"  file_token_ids_prefix_match_{compare_len}={file_ids_prefix_match}")
    print(f"  hidden_shape={tuple(fresh_run['hidden'].shape)}")
    print(f"  hidden_shape_slots_expected={len(connector_layer_ids)}")
    print(f"  prompt_logprobs={[_fmt(x) for x in fresh_run['token_logprobs']]}")

    hf_model = _load_hf_model(args, torch)

    fresh_hidden = fresh_run["hidden"]
    if fresh_hidden.ndim != 3:
        raise RuntimeError(f"Expected fresh hidden shape [seq, slots, hidden], got {tuple(fresh_hidden.shape)}")
    if fresh_hidden.shape[1] < 2:
        raise RuntimeError(f"Fresh hidden has too few slots: {tuple(fresh_hidden.shape)}")

    raw_aux = raw_item["hidden_states"].reshape(len(raw_input_ids), num_target_layers, hidden_size)
    raw_last = raw_item["verifier_last_hidden_states"]
    batch_aux = batch["hidden_states"][0].reshape(args.total_seq_len, num_target_layers, hidden_size)
    batch_last = batch["verifier_last_hidden_states"][0]

    local_slice = slice(args.local_start, raw_hidden_end)
    packed_slice = slice(packed_hidden_start, packed_hidden_end)
    final_slot = fresh_hidden.shape[1] - 1
    slot33 = num_target_layers - 1

    fresh_final_window = fresh_hidden[local_slice, final_slot]
    fresh_slot33_window = fresh_hidden[local_slice, slot33]
    raw_last_window = raw_last[local_slice]
    raw_slot33_window = raw_aux[local_slice, slot33]
    batch_last_window = batch_last[packed_slice]
    batch_slot33_window = batch_aux[packed_slice, slot33]

    print("TRACE hidden_diff")
    for label, a, b in (
        ("raw_last_vs_fresh_final", raw_last_window, fresh_final_window),
        ("packed_last_vs_raw_last", batch_last_window, raw_last_window),
        ("packed_last_vs_fresh_final", batch_last_window, fresh_final_window),
        ("raw_slot33_vs_fresh_slot33", raw_slot33_window, fresh_slot33_window),
        ("packed_slot33_vs_raw_slot33", batch_slot33_window, raw_slot33_window),
    ):
        max_abs, mean_abs = _diff_stats(a, b)
        print(f"  {label}_max_abs={_fmt(max_abs)}")
        print(f"  {label}_mean_abs={_fmt(mean_abs)}")

    print("TRACE slot_probe")
    probe_raw_slot33 = raw_aux[args.local_start, slot33]
    probe_raw_final = raw_last[args.local_start]
    probe_batch_final = batch_last[packed_hidden_start]
    probe_fresh_slot33 = fresh_hidden[args.local_start, slot33]
    probe_fresh_final = fresh_hidden[args.local_start, final_slot]
    for label, a, b in (
        ("batch_last_vs_raw_slot33", probe_batch_final, probe_raw_slot33),
        ("batch_last_vs_raw_final", probe_batch_final, probe_raw_final),
        ("batch_last_vs_fresh_slot33", probe_batch_final, probe_fresh_slot33),
        ("batch_last_vs_fresh_final", probe_batch_final, probe_fresh_final),
    ):
        max_abs, mean_abs = _diff_stats(a, b)
        print(f"  {label}_max_abs={_fmt(max_abs)}")
        print(f"  {label}_mean_abs={_fmt(mean_abs)}")

    fresh_logprobs = _project_full_lm_head(
        torch=torch,
        hf_model=hf_model,
        final_hidden=fresh_hidden[:, final_slot],
        prompt=prompt,
        score_positions=score_positions,
    )
    raw_logprobs = _project_full_lm_head(
        torch=torch,
        hf_model=hf_model,
        final_hidden=raw_last,
        prompt=raw_input_ids,
        score_positions=score_positions,
    )
    batch_prompt = batch["input_ids"][0].detach().cpu().tolist()
    batch_score_positions = [doc_start + pos for pos in score_positions]
    batch_logprobs = _project_full_lm_head(
        torch=torch,
        hf_model=hf_model,
        final_hidden=batch_last,
        prompt=batch_prompt,
        score_positions=batch_score_positions,
    )
    slot33_logprobs = _project_full_lm_head(
        torch=torch,
        hf_model=hf_model,
        final_hidden=raw_aux[:, slot33],
        prompt=raw_input_ids,
        score_positions=score_positions,
    )

    target_ids = [raw_input_ids[pos] for pos in score_positions]
    _print_logprob_table(
        score_positions=score_positions,
        target_ids=target_ids,
        fresh=fresh_logprobs,
        raw=raw_logprobs,
        batch=batch_logprobs,
        vllm=fresh_run["token_logprobs"],
    )

    print("TRACE slot33_projection")
    print(f"  raw_slot33_logprobs={[_fmt(x) for x in slot33_logprobs]}")
    print(
        "  raw_slot33_vs_vllm_prompt_max_abs_diff="
        f"{_fmt(_max_abs_delta(slot33_logprobs, fresh_run['token_logprobs']))}"
    )

    fresh_vs_prompt = _max_abs_delta(fresh_logprobs, fresh_run["token_logprobs"])
    raw_vs_fresh_plog = _max_abs_delta(raw_logprobs, fresh_logprobs)
    batch_vs_raw_plog = _max_abs_delta(batch_logprobs, raw_logprobs)
    raw_vs_fresh_hidden, _ = _diff_stats(raw_last_window, fresh_final_window)
    packed_vs_raw_hidden, _ = _diff_stats(batch_last_window, raw_last_window)

    print("TRACE interpretation")
    print(f"  A_fresh_projection_vs_D_vllm_prompt_max_abs={_fmt(fresh_vs_prompt)}")
    print(f"  B_raw_projection_vs_A_fresh_projection_max_abs={_fmt(raw_vs_fresh_plog)}")
    print(f"  C_batch_projection_vs_B_raw_projection_max_abs={_fmt(batch_vs_raw_plog)}")
    print(f"  raw_vs_fresh_hidden_max_abs={_fmt(raw_vs_fresh_hidden)}")
    print(f"  packed_vs_raw_hidden_max_abs={_fmt(packed_vs_raw_hidden)}")
    if fresh_vs_prompt <= args.logprob_tol and raw_vs_fresh_hidden > args.hidden_tol:
        print("  conclusion=A ~= D, B != A: dataloader raw/generated hidden differs from fresh vLLM")
    elif raw_vs_fresh_hidden <= args.hidden_tol and packed_vs_raw_hidden > args.hidden_tol:
        print("  conclusion=A ~= B, C != B: collate/packing hidden slice differs from raw item")
    elif (
        fresh_vs_prompt <= args.logprob_tol
        and raw_vs_fresh_hidden <= args.hidden_tol
        and packed_vs_raw_hidden <= args.hidden_tol
    ):
        print("  conclusion=A ~= B ~= C: check hidden/target position or gather indexing next")
    else:
        print("  conclusion=mixed: inspect TRACE logprob_table and TRACE slot_probe")

    if not args.keep_hidden_files:
        _delete_hidden_file(fresh_run["hidden_path"])


if __name__ == "__main__":
    main()
