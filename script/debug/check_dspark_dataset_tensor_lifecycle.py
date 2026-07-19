#!/usr/bin/env python3
"""Trace DSpark training dataloader hidden-state tensor lifecycle.

This script intentionally focuses on the data path, not on a draft checkpoint:

    vLLM/cache safetensors
    -> ArrowDataset._get_raw_data()
    -> BaseDataset.__getitem__ dtype conversion
    -> Dataset transform
    -> DSpark collate/packing

It uses the real training dataset, sampler, transform, and collate function, but
keeps local snapshots at each stage so a clean vLLM hidden tensor is never
silently compared with a transformed or shifted tensor.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Any
import warnings

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace the DSpark dataset hidden-state tensor lifecycle."
    )
    parser.add_argument("--verifier-name-or-path", default="/models/Qwen3-4B")
    parser.add_argument("--data-path", default="/data/open_perfectblend_qwen3_4b_100k")
    parser.add_argument("--hidden-states-path", default=None)
    parser.add_argument("--vllm-endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--dataset-index", type=int, default=45760)
    parser.add_argument("--batch-index", type=int, default=None)
    parser.add_argument("--total-seq-len", type=int, default=3072)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--num-target-layers", type=int, default=5)
    parser.add_argument("--target-layer-ids", type=int, nargs="*", default=[1, 9, 17, 25, 33])
    parser.add_argument(
        "--hidden-states-dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--on-missing", choices=["generate", "skip", "warn", "raise"], default="generate")
    parser.add_argument("--on-generate", choices=["cache", "delete"], default="delete")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--hidden-file-timeout", type=float, default=30.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-start", type=int, default=67)
    parser.add_argument("--gt-len", type=int, default=7)
    parser.add_argument("--prompt-logprobs", type=int, default=1)
    parser.add_argument("--skip-direct-vllm", action="store_true")
    parser.add_argument("--keep-direct-hidden-file", action="store_true")
    return parser.parse_args()


def fmt(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        return str(value)
    if value == 0 or (1e-3 <= abs(value) < 1e4):
        return f"{value:.6f}"
    return f"{value:.6e}"


def dtype_from_name(torch: Any, name: str) -> Any:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def clone_cpu(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return dict(value)
    return value


def tensor_window(tensor: Any, start: int, end: int) -> Any:
    if tensor is None:
        return None
    return tensor[start:end].detach().cpu().clone()


def diff_stats(left: Any, right: Any) -> tuple[float, float]:
    import torch

    if left is None or right is None:
        return math.nan, math.nan
    if tuple(left.shape) != tuple(right.shape):
        return math.nan, math.nan
    delta = (left.float() - right.float()).abs()
    if delta.numel() == 0:
        return 0.0, 0.0
    return float(delta.max().item()), float(delta.mean().item())


def print_diff(label: str, left: Any, right: Any) -> None:
    max_abs, mean_abs = diff_stats(left, right)
    print(f"  {label}_max_abs={fmt(max_abs)}")
    print(f"  {label}_mean_abs={fmt(mean_abs)}")


def print_tensor(label: str, tensor: Any) -> None:
    if tensor is None:
        print(f"  {label}: <missing>")
        return
    print(f"  {label}: shape={tuple(tensor.shape)} dtype={tensor.dtype}")


def to_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        return [int(x) for x in value.tolist()]
    return [int(x) for x in value]


def wait_for_hidden_file(path_value: str, timeout: float) -> None:
    from speculators.data_generation.vllm_client import wait_for_lock

    path = Path(path_value)
    lock = Path(path_value + ".lock")
    deadline = time.monotonic() + timeout
    while lock.exists() or not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for hidden-state file: {path}")
        if lock.exists():
            wait_for_lock(str(lock), timeout=max(deadline - time.monotonic(), 0.1))
        else:
            time.sleep(0.05)


def unlink_hidden_file(path_value: str | None) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.unlink(missing_ok=True)
    Path(str(path) + ".lock").unlink(missing_ok=True)


class TracingArrowDataset:
    """Small wrapper around ArrowDataset with explicit lifecycle snapshots."""

    def __init__(self, *args: Any, trace_indices: set[int], **kwargs: Any) -> None:
        from speculators.train.data import ArrowDataset

        class _Dataset(ArrowDataset):
            def __init__(self, *inner_args: Any, trace_indices: set[int], **inner_kwargs: Any) -> None:
                self.trace_indices = set(trace_indices)
                self.snapshots: dict[int, dict[str, Any]] = {}
                super().__init__(*inner_args, **inner_kwargs)

            def _snap(self, index: int) -> dict[str, Any] | None:
                if int(index) not in self.trace_indices:
                    return None
                return self.snapshots.setdefault(int(index), {})

            def _record_loaded(self, index: int, loaded_hs: dict[str, Any], source: dict[str, Any]) -> None:
                snap = self._snap(index)
                if snap is None:
                    return
                hidden = loaded_hs["hidden_states"].detach().cpu().clone()
                token_ids = loaded_hs["token_ids"].detach().cpu().clone()
                snap["file_all"] = hidden
                snap["file_aux"] = hidden[:, :-1].flatten(1).clone()
                snap["file_last"] = hidden[:, -1].clone()
                snap["file_token_ids"] = token_ids
                snap["file_source"] = dict(source)

            def _get_raw_data(self, index: int) -> dict[str, Any] | None:
                import torch
                from speculators.train.data import _maybe_load_hs_file

                file_idx = self._map_to_file_idx(index)
                candidate_path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
                loaded_hs = _maybe_load_hs_file(candidate_path)
                hidden_state_source = {
                    "source": "cache",
                    "path": str(candidate_path),
                    "index": int(index),
                    "file_idx": int(file_idx),
                }

                if loaded_hs is None:
                    match self.on_missing:
                        case "generate":
                            loaded_hs = self._maybe_generate_hs(index)
                            hidden_state_source = {
                                "source": "generated",
                                "path": None
                                if loaded_hs is None
                                else loaded_hs.get("_hidden_state_source", {}).get("path"),
                                "index": int(index),
                            }
                        case "skip":
                            return None
                        case "warn":
                            warnings.warn(
                                f"Failed to load hidden states for sample {index}. Skipping...",
                                stacklevel=1,
                            )
                            return None
                        case "raise":
                            raise RuntimeError(
                                f"Failed to load hidden states for sample {index}."
                            )

                if loaded_hs is None:
                    return None

                self._record_loaded(index, loaded_hs, hidden_state_source)

                dataset_ids = self.data[index]["input_ids"]
                if not torch.equal(loaded_hs["token_ids"], dataset_ids):
                    snap = self._snap(index)
                    if snap is not None:
                        snap["dataset_token_ids"] = dataset_ids.detach().cpu().clone()
                    warnings.warn(
                        f"Loaded token ids for index {index} do not match dataset input ids",
                        stacklevel=1,
                    )
                    return None

                out = {
                    "hidden_states": loaded_hs["hidden_states"][:, :-1].flatten(1),
                    "input_ids": loaded_hs["token_ids"],
                    "verifier_last_hidden_states": loaded_hs["hidden_states"][:, -1],
                    "loss_mask": self.data[index]["loss_mask"],
                    "_hidden_state_source": hidden_state_source,
                }

                snap = self._snap(index)
                if snap is not None:
                    snap["getraw_aux"] = out["hidden_states"].detach().cpu().clone()
                    snap["getraw_last"] = out["verifier_last_hidden_states"].detach().cpu().clone()
                    snap["getraw_input_ids"] = out["input_ids"].detach().cpu().clone()
                    snap["getraw_loss_mask"] = out["loss_mask"].detach().cpu().clone()
                    snap["getraw_source"] = dict(hidden_state_source)
                return out

            def __getitem__(self, index: int) -> dict[str, Any] | None:
                import torch

                data = self._get_raw_data(index)
                if data is None:
                    return None

                data = {
                    k: v.to(self.hidden_states_dtype) if "hidden_states" in k else v
                    for k, v in data.items()
                }

                seq_len = data["input_ids"].shape[0]
                data["lengths"] = torch.tensor([seq_len], dtype=torch.long)
                data["position_ids"] = torch.arange(seq_len, dtype=torch.long)

                snap = self._snap(index)
                if snap is not None:
                    snap["post_dtype_aux"] = data["hidden_states"].detach().cpu().clone()
                    snap["post_dtype_last"] = data["verifier_last_hidden_states"].detach().cpu().clone()
                    snap["post_dtype_input_ids"] = data["input_ids"].detach().cpu().clone()

                if self.transform:
                    data = self.transform(data)

                snap = self._snap(index)
                if snap is not None:
                    snap["getitem_aux"] = data["hidden_states"].detach().cpu().clone()
                    snap["getitem_last"] = data["verifier_last_hidden_states"].detach().cpu().clone()
                    snap["getitem_input_ids"] = data["input_ids"].detach().cpu().clone()
                    snap["getitem_loss_mask"] = data["loss_mask"].detach().cpu().clone()
                    snap["getitem_lengths"] = data["lengths"].detach().cpu().clone()
                    snap["getitem_position_ids"] = data["position_ids"].detach().cpu().clone()
                    snap["getitem_source"] = dict(data.get("_hidden_state_source", {}))

                return data

        self.dataset = _Dataset(*args, trace_indices=trace_indices, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index]


def build_sampler(dataset: Any, total_seq_len: int) -> Any:
    from speculators.train.distributed_batch_sampler import MultipackDistributedBatchSamplerV2

    return MultipackDistributedBatchSamplerV2(
        batch_max_length=total_seq_len,
        lengths=dataset.approx_lengths,
        num_replicas=1,
        rank=0,
    )


def sampler_batches(dataset: Any, total_seq_len: int) -> list[list[int]]:
    batches = []
    for batch in build_sampler(dataset, total_seq_len):
        if hasattr(batch, "tolist"):
            batches.append([int(x) for x in batch.tolist()])
        else:
            batches.append([int(x) for x in batch])
    return batches


def request_direct_vllm(
    *,
    endpoint: str,
    model_id: str,
    token_ids: list[int],
    score_positions: list[int],
    prompt_logprobs: int | None,
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
    extra_body: dict[str, Any] = {"return_token_ids": True}
    if prompt_logprobs is not None:
        extra_body["prompt_logprobs"] = prompt_logprobs

    response = client.completions.create(
        model=model_id,
        prompt=token_ids,
        max_tokens=1,
        extra_body=extra_body,
        timeout=request_timeout,
    )
    response_ids = _prompt_token_ids(response)
    prompt_logprob_obj = _prompt_logprobs(response)
    hidden_path = _kv_hidden_states_path(response)
    if response_ids is None:
        raise RuntimeError("direct vLLM response missing prompt_token_ids")
    if prompt_logprobs is not None and prompt_logprob_obj is None:
        raise RuntimeError("direct vLLM response missing prompt_logprobs")
    if hidden_path is None:
        raise RuntimeError("direct vLLM response missing hidden_states_path")

    wait_for_hidden_file(hidden_path, hidden_file_timeout)
    loaded = load_file(hidden_path)
    reported = None
    if prompt_logprobs is not None:
        reported = [
            _extract_token_logprob(prompt_logprob_obj, pos, token_ids[pos])
            for pos in score_positions
        ]
    return {
        "prompt_token_ids": [int(x) for x in response_ids],
        "hidden_path": hidden_path,
        "hidden": loaded["hidden_states"].detach().cpu().clone(),
        "file_token_ids": loaded["token_ids"].detach().cpu().clone(),
        "prompt_logprobs": reported,
    }


def print_stage_shapes(snap: dict[str, Any]) -> None:
    print("TRACE stage_shapes")
    for key in (
        "file_all",
        "file_aux",
        "file_last",
        "getraw_aux",
        "getraw_last",
        "post_dtype_aux",
        "post_dtype_last",
        "getitem_aux",
        "getitem_last",
        "batch_aux",
        "batch_last",
    ):
        print_tensor(key, snap.get(key))


def print_lifecycle_diffs(snap: dict[str, Any], start: int, end: int) -> None:
    file_aux = tensor_window(snap.get("file_aux"), start, end)
    getraw_aux = tensor_window(snap.get("getraw_aux"), start, end)
    post_dtype_aux = tensor_window(snap.get("post_dtype_aux"), start, end)
    getitem_aux = tensor_window(snap.get("getitem_aux"), start, end)
    batch_aux = tensor_window(snap.get("batch_aux"), start, end)

    file_last = tensor_window(snap.get("file_last"), start, end)
    getraw_last = tensor_window(snap.get("getraw_last"), start, end)
    post_dtype_last = tensor_window(snap.get("post_dtype_last"), start, end)
    getitem_last = tensor_window(snap.get("getitem_last"), start, end)
    batch_last = tensor_window(snap.get("batch_last"), start, end)

    print("TRACE aux_lifecycle_window_diff")
    print_diff("file_vs_getraw", file_aux, getraw_aux)
    print_diff("getraw_vs_post_dtype", getraw_aux, post_dtype_aux)
    print_diff("post_dtype_vs_getitem", post_dtype_aux, getitem_aux)
    print_diff("getitem_vs_batch", getitem_aux, batch_aux)
    print_diff("file_vs_batch", file_aux, batch_aux)

    print("TRACE last_lifecycle_window_diff")
    print_diff("file_vs_getraw", file_last, getraw_last)
    print_diff("getraw_vs_post_dtype", getraw_last, post_dtype_last)
    print_diff("post_dtype_vs_getitem", post_dtype_last, getitem_last)
    print_diff("getitem_vs_batch", getitem_last, batch_last)
    print_diff("file_vs_batch", file_last, batch_last)


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader
    from speculators.train.data import create_collate_fn
    from speculators.train.noise_transforms import AddUniformNoise

    if args.num_workers != 0:
        raise RuntimeError(
            "This lifecycle tracer requires --num-workers 0 so snapshots remain "
            "in the main process. Run the real trainer separately for worker races."
        )

    torch.manual_seed(args.seed)
    hidden_dtype = dtype_from_name(torch, args.hidden_states_dtype)
    transform = AddUniformNoise(std=args.noise_std)

    probe_dataset = TracingArrowDataset(
        max_len=args.total_seq_len,
        datapath=args.data_path,
        hidden_states_path=args.hidden_states_path,
        vllm_endpoint=args.vllm_endpoint,
        on_missing=args.on_missing,
        on_generate=args.on_generate,
        split_ratio=0.9,
        transform=transform,
        hidden_states_dtype=hidden_dtype,
        model=args.verifier_name_or_path,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        trace_indices=set(),
    )
    batches = sampler_batches(probe_dataset, args.total_seq_len)
    if args.batch_index is None:
        for idx, batch_indices in enumerate(batches):
            if args.dataset_index in batch_indices:
                batch_index = idx
                break
        else:
            raise RuntimeError(f"dataset index {args.dataset_index} not found in sampler")
    else:
        batch_index = int(args.batch_index)
        if not (0 <= batch_index < len(batches)):
            raise RuntimeError(f"batch index {batch_index} out of range 0..{len(batches) - 1}")

    batch_indices = batches[batch_index]
    if args.dataset_index not in batch_indices:
        raise RuntimeError(
            f"dataset index {args.dataset_index} not present in batch {batch_index}: "
            f"{batch_indices}"
        )
    doc_id = batch_indices.index(args.dataset_index)
    prior_batches = batches[: batch_index + 1]
    request_order_to_batch = [int(index) for group in prior_batches for index in group]
    target_request_ordinal = request_order_to_batch.index(args.dataset_index)

    dataset = TracingArrowDataset(
        max_len=args.total_seq_len,
        datapath=args.data_path,
        hidden_states_path=args.hidden_states_path,
        vllm_endpoint=args.vllm_endpoint,
        on_missing=args.on_missing,
        on_generate=args.on_generate,
        split_ratio=0.9,
        transform=transform,
        hidden_states_dtype=hidden_dtype,
        model=args.verifier_name_or_path,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        trace_indices=set(batch_indices),
    )

    def sanitize_hidden_state_source(item: dict[str, Any]) -> dict[str, Any]:
        item.pop("_hidden_state_source", None)
        return item

    loader = DataLoader(
        dataset,
        batch_sampler=build_sampler(dataset, args.total_seq_len),
        num_workers=args.num_workers,
        prefetch_factor=None,
        pin_memory=True,
        collate_fn=create_collate_fn(
            args.total_seq_len,
            args.hidden_size,
            num_target_layers=args.num_target_layers,
            dtype=hidden_dtype,
            preprocess=sanitize_hidden_state_source,
        ),
        persistent_workers=False,
    )

    batch = None
    for idx, candidate in enumerate(loader):
        if idx == batch_index:
            batch = candidate
            break
    if batch is None:
        raise RuntimeError(f"failed to load batch {batch_index}")

    snap = dataset.snapshots.get(args.dataset_index)
    if snap is None:
        raise RuntimeError(f"no lifecycle snapshot captured for index {args.dataset_index}")

    document_ids = batch["document_ids"]
    doc_positions = (document_ids[0] == doc_id).nonzero(as_tuple=False).flatten()
    if doc_positions.numel() == 0:
        raise RuntimeError(f"doc_id {doc_id} not present in collated document_ids")
    doc_start = int(doc_positions[0].item())
    doc_end = int(doc_positions[-1].item()) + 1
    local_start = args.local_start
    local_end = args.local_start + args.gt_len
    score_positions = list(range(args.local_start + 1, args.local_start + args.gt_len + 1))
    packed_start = doc_start + local_start
    packed_end = doc_start + local_end

    snap["batch_aux"] = batch["hidden_states"][0, doc_start:doc_end].detach().cpu().clone()
    snap["batch_last"] = batch["verifier_last_hidden_states"][0, doc_start:doc_end].detach().cpu().clone()
    snap["batch_input_ids"] = batch["input_ids"][0, doc_start:doc_end].detach().cpu().clone()
    snap["batch_loss_mask"] = batch["loss_mask"][0, doc_start:doc_end].detach().cpu().clone()
    snap["batch_position_ids"] = batch["position_ids"][0, doc_start:doc_end].detach().cpu().clone()

    raw_ids = to_ids(snap["getitem_input_ids"])
    raw_tokens = raw_ids[local_start : local_end + 1]
    packed_tokens = to_ids(batch["input_ids"][0, packed_start : packed_end + 1])
    target_ids = [raw_ids[pos] for pos in score_positions]

    print("TRACE config")
    print(f"  repo={ROOT}")
    print(f"  verifier_name_or_path={args.verifier_name_or_path}")
    print(f"  data_path={args.data_path}")
    print(f"  hidden_states_path={args.hidden_states_path}")
    print(f"  dataset_index={args.dataset_index}")
    print(f"  vllm_endpoint={args.vllm_endpoint}")
    print(f"  total_seq_len={args.total_seq_len}")
    print(f"  hidden_size={args.hidden_size}")
    print(f"  num_target_layers={args.num_target_layers}")
    print(f"  target_layer_ids={args.target_layer_ids}")
    print(f"  hidden_states_dtype={hidden_dtype}")
    print(f"  noise_std={args.noise_std}")
    print(f"  seed={args.seed}")
    print(f"  num_workers={args.num_workers}")
    print(f"  on_missing={args.on_missing}")
    print(f"  on_generate={args.on_generate}")
    print("  speculator_type=dspark")
    print("  train_preprocess=None")
    print("  collate_sanitize_hidden_state_source=True")

    print("TRACE transform")
    print(f"  class={type(dataset.transform).__name__ if dataset.transform else None}")
    print(f"  repr={dataset.transform!r}")
    print(f"  std={getattr(dataset.transform, 'std', None)}")
    print(f"  tensors={list(getattr(dataset.transform, 'tensors', ())) if dataset.transform else []}")

    print("TRACE batch_locator")
    print(f"  batch_index={batch_index}")
    print(f"  batch_indices={batch_indices}")
    print(f"  target_doc_id={doc_id}")
    print(f"  doc_start={doc_start}")
    print(f"  doc_end={doc_end}")
    print(f"  raw_item_len={len(raw_ids)}")
    print(f"  packed_hidden_window={packed_start}:{packed_end}")
    print(f"  local_hidden_window={local_start}:{local_end}")
    print(f"  local_score_positions={score_positions}")
    print(f"  packed_score_positions={[doc_start + pos for pos in score_positions]}")
    print(f"  target_ids={target_ids}")
    print(f"  hidden_state_source={snap.get('getitem_source')}")

    print("TRACE request_sequence")
    print(f"  batches_0_to_target_batch={prior_batches}")
    print(f"  flattened_indices_0_to_target_batch={request_order_to_batch}")
    print(f"  target_request_ordinal_0_based={target_request_ordinal}")
    print(f"  target_request_ordinal_1_based={target_request_ordinal + 1}")

    print("TRACE token_alignment")
    print(f"  raw_tokens_{local_start}_{local_end + 1}={raw_tokens}")
    print(f"  packed_tokens_{packed_start}_{packed_end + 1}={packed_tokens}")
    print(f"  packed_matches_raw={packed_tokens == raw_tokens}")
    print(f"  file_token_ids_match_getitem={bool(torch.equal(snap['file_token_ids'], snap['getitem_input_ids']))}")
    print(f"  getraw_token_ids_match_getitem={bool(torch.equal(snap['getraw_input_ids'], snap['getitem_input_ids']))}")

    print_stage_shapes(snap)
    print_lifecycle_diffs(snap, local_start, local_end)

    print("TRACE batch_tensors")
    for key, value in batch.items():
        print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")

    direct_hidden_only = None
    direct_scored = None
    if not args.skip_direct_vllm:
        import openai

        client = openai.OpenAI(base_url=args.vllm_endpoint, api_key="EMPTY", max_retries=0)
        model_id = client.models.list().data[0].id
        direct_hidden_only = request_direct_vllm(
            endpoint=args.vllm_endpoint,
            model_id=model_id,
            token_ids=raw_ids,
            score_positions=score_positions,
            prompt_logprobs=None,
            request_timeout=args.request_timeout,
            hidden_file_timeout=args.hidden_file_timeout,
        )
        direct_scored = request_direct_vllm(
            endpoint=args.vllm_endpoint,
            model_id=model_id,
            token_ids=raw_ids,
            score_positions=score_positions,
            prompt_logprobs=args.prompt_logprobs,
            request_timeout=args.request_timeout,
            hidden_file_timeout=args.hidden_file_timeout,
        )
        hidden_only_last = direct_hidden_only["hidden"][:, -1]
        hidden_only_aux = direct_hidden_only["hidden"][:, :-1].flatten(1)
        scored_last = direct_scored["hidden"][:, -1]
        scored_aux = direct_scored["hidden"][:, :-1].flatten(1)
        print("TRACE direct_hidden_only")
        print(f"  model_id={model_id}")
        print(f"  hidden_path={direct_hidden_only['hidden_path']}")
        print(f"  prompt_ids_match={direct_hidden_only['prompt_token_ids'] == raw_ids}")
        print(f"  file_token_ids_match={bool(torch.equal(direct_hidden_only['file_token_ids'], snap['getitem_input_ids']))}")
        print(f"  hidden_shape={tuple(direct_hidden_only['hidden'].shape)}")

        print("TRACE direct_scored")
        print(f"  model_id={model_id}")
        print(f"  hidden_path={direct_scored['hidden_path']}")
        print(f"  prompt_ids_match={direct_scored['prompt_token_ids'] == raw_ids}")
        print(f"  file_token_ids_match={bool(torch.equal(direct_scored['file_token_ids'], snap['getitem_input_ids']))}")
        print(f"  hidden_shape={tuple(direct_scored['hidden'].shape)}")
        print(f"  prompt_logprobs={[fmt(x) for x in direct_scored['prompt_logprobs']]}")

        print("TRACE direct_window_diff")
        print_diff(
            "file_last_vs_hidden_only_last",
            tensor_window(snap.get("file_last"), local_start, local_end),
            tensor_window(hidden_only_last, local_start, local_end),
        )
        print_diff(
            "hidden_only_last_vs_scored_last",
            tensor_window(hidden_only_last, local_start, local_end),
            tensor_window(scored_last, local_start, local_end),
        )
        print_diff(
            "file_last_vs_scored_last",
            tensor_window(snap.get("file_last"), local_start, local_end),
            tensor_window(scored_last, local_start, local_end),
        )
        print_diff(
            "getitem_last_vs_scored_last",
            tensor_window(snap.get("getitem_last"), local_start, local_end),
            tensor_window(scored_last, local_start, local_end),
        )
        print_diff(
            "batch_last_vs_scored_last",
            tensor_window(snap.get("batch_last"), local_start, local_end),
            tensor_window(scored_last, local_start, local_end),
        )
        print_diff(
            "file_aux_vs_hidden_only_aux",
            tensor_window(snap.get("file_aux"), local_start, local_end),
            tensor_window(hidden_only_aux, local_start, local_end),
        )
        print_diff(
            "hidden_only_aux_vs_scored_aux",
            tensor_window(hidden_only_aux, local_start, local_end),
            tensor_window(scored_aux, local_start, local_end),
        )
        print_diff(
            "file_aux_vs_scored_aux",
            tensor_window(snap.get("file_aux"), local_start, local_end),
            tensor_window(scored_aux, local_start, local_end),
        )
        print_diff(
            "getitem_aux_vs_scored_aux",
            tensor_window(snap.get("getitem_aux"), local_start, local_end),
            tensor_window(scored_aux, local_start, local_end),
        )
        print_diff(
            "batch_aux_vs_scored_aux",
            tensor_window(snap.get("batch_aux"), local_start, local_end),
            tensor_window(scored_aux, local_start, local_end),
        )

        if not args.keep_direct_hidden_file:
            unlink_hidden_file(direct_hidden_only["hidden_path"])
            unlink_hidden_file(direct_scored["hidden_path"])

    print("TRACE interpretation")
    post_dtype_vs_getitem_last, _ = diff_stats(
        tensor_window(snap.get("post_dtype_last"), local_start, local_end),
        tensor_window(snap.get("getitem_last"), local_start, local_end),
    )
    getitem_vs_batch_last, _ = diff_stats(
        tensor_window(snap.get("getitem_last"), local_start, local_end),
        tensor_window(snap.get("batch_last"), local_start, local_end),
    )
    post_dtype_vs_getitem_aux, _ = diff_stats(
        tensor_window(snap.get("post_dtype_aux"), local_start, local_end),
        tensor_window(snap.get("getitem_aux"), local_start, local_end),
    )
    if post_dtype_vs_getitem_last and post_dtype_vs_getitem_last > 0:
        conclusion = "transform_or_getitem_changed_verifier_last_hidden_states"
    elif getitem_vs_batch_last and getitem_vs_batch_last > 0:
        conclusion = "collate_changed_verifier_last_hidden_states"
    elif args.noise_std > 0 and post_dtype_vs_getitem_aux and post_dtype_vs_getitem_aux > 0:
        conclusion = "aux_hidden_noise_observed_last_hidden_unchanged"
    else:
        conclusion = "lifecycle_preserves_traced_window"
    print(f"  conclusion={conclusion}")
    if direct_scored is not None:
        print("  note=direct_vllm_is_a_later_request; lifecycle diffs identify client-side mutation separately")


if __name__ == "__main__":
    main()
