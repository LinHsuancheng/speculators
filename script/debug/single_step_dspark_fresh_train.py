#!/usr/bin/env python3
"""Run one fresh DSpark training step with focused hidden-state checks.

This script is intentionally independent from the older trace/debug scripts.
It uses the normal training modules to do one real single-process step:

1. initialize a DSpark draft from training arguments, without loading a draft
   checkpoint;
2. build the normal train dataloader with online hidden-state generation;
3. read one real multipack batch;
4. compare the selected document's packed verifier-last hidden states against a
   direct vLLM scored hidden request;
5. run SampledAcceptanceAugmentor;
6. run model forward, backward, grad clip, optimizer step.

The main diagnostic is whether the batch hidden states used by training
reconstruct the same target prompt logprobs as a direct vLLM request for the
same raw document positions.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one fresh DSpark training step and check hidden alignment."
    )
    parser.add_argument("--verifier-name-or-path", default="/models/Qwen3-4B")
    parser.add_argument("--data-path", default="/data/open_perfectblend_qwen3_4b_100k")
    parser.add_argument("--hidden-states-path", default=None)
    parser.add_argument("--vllm-endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--device", default="npu:15")
    parser.add_argument("--dataset-index", type=int, default=45760)
    parser.add_argument("--batch-index", type=int, default=None)
    parser.add_argument("--total-seq-len", type=int, default=3072)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--max-anchors", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--draft-vocab-size", type=int, default=32000)
    parser.add_argument("--target-layer-ids", type=int, nargs="+", default=[1, 9, 17, 25, 33])
    parser.add_argument("--draft-attn-impl", choices=["simple_flex_attention", "sdpa", "eager"], default="sdpa")
    parser.add_argument("--markov-rank", type=int, default=256)
    parser.add_argument("--markov-head-type", choices=["vanilla", "gated", "rnn"], default="vanilla")
    parser.add_argument("--loss-fn", default='{"ce": 0.1, "tv": 0.9}')
    parser.add_argument("--confidence-head-alpha", type=float, default=1.0)
    parser.add_argument("--sampled-acceptance-loss-alpha", type=float, default=1.0)
    parser.add_argument("--hidden-states-dtype", default="bfloat16")
    parser.add_argument("--on-missing", choices=["generate", "skip", "warn", "raise"], default="generate")
    parser.add_argument("--on-generate", choices=["cache", "delete"], default="delete")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--hidden-file-timeout", type=float, default=30.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--local-start", type=int, default=67)
    parser.add_argument("--gt-len", type=int, default=7)
    parser.add_argument("--prompt-logprobs", type=int, default=1)
    parser.add_argument("--hidden-tol", type=float, default=1e-2)
    parser.add_argument("--logprob-tol", type=float, default=0.5)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--preserve-hidden-state-source",
        action="store_true",
        help=(
            "Do not remove the debug _hidden_state_source key before collate. "
            "This follows the raw current dataloader output exactly and may fail "
            "if that non-tensor key reaches collate."
        ),
    )
    parser.add_argument("--skip-backward", action="store_true")
    parser.add_argument("--keep-direct-hidden-file", action="store_true")
    return parser.parse_args()


def fmt(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        return str(value)
    if value == 0 or (1e-3 <= abs(value) < 1e4):
        return f"{value:.6f}"
    return f"{value:.6e}"


def diff_stats(left: Any, right: Any) -> tuple[float, float]:
    diff = (left.detach().float().cpu() - right.detach().float().cpu()).abs()
    return float(diff.max().item()), float(diff.mean().item())


def max_abs_delta(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right, strict=True)) if left else float("nan")


def resolve_device(torch: Any, value: str) -> Any:
    if value:
        return torch.device(value)
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def make_train_args(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        {
            "speculator_type": "dspark",
            "from_pretrained": "",
            "draft_config": "",
            "draft_arch": "qwen3",
            "draft_hidden_act": None,
            "dry_run": False,
            "legacy_data": False,
            "token_freq_path": None,
            "d2t_path": None,
            "t2d_path": None,
            "mask_token_id": None,
            "sliding_window": 2048,
            "sliding_window_indices": [],
            "sliding_window_non_causal": False,
            "micro_block_size": 0,
            "anchor_len": 1,
            "micro_block_layer_growth": False,
            "max_prev_micro_blocks": None,
            "micro_token_layer_growth": False,
            "max_prev_micro_tokens": None,
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
            "cat_mode": "none",
            "dflash_decay_gamma": 4.0,
            "num_speculative_steps": 0,
            "norm_before_fc": False,
            "norm_output": False,
            "num_depths": 8,
            "down_sample_ratio": 0.7,
            "down_sample_ratio_min": 0.2,
            "epochs": 1,
            "save_path": "/tmp/dspark_single_step_debug_checkpoints",
            "log_dir": "/tmp/dspark_single_step_debug_logs",
            "run_name": None,
            "no_resume_from_checkpoint": True,
            "optimizer": "adamw",
            "muon_lr": 0.02,
            "muon_momentum": 0.95,
            "muon_weight_decay": 0.1,
            "muon_ns_steps": 5,
            "muon_adjust_lr_fn": "match_rms_adamw",
            "scheduler_type": "none",
            "scheduler_warmup_steps": None,
            "scheduler_total_steps": None,
            "scheduler_num_cosine_cycles": 0.5,
            "checkpoint_freq": 1.0,
            "save_best": False,
        }
    )
    return argparse.Namespace(**values)


def batch_indices_for_loader(loader: Any) -> list[list[int]]:
    batches = list(loader.batch_sampler)
    out: list[list[int]] = []
    for batch in batches:
        out.append([int(x) for x in batch.tolist()] if hasattr(batch, "tolist") else [int(x) for x in batch])
    return out


def find_target_batch(loader: Any, dataset_index: int, requested: int | None) -> tuple[int, int, list[int]]:
    batches = batch_indices_for_loader(loader)
    if requested is not None:
        indices = batches[int(requested)]
        if dataset_index not in indices:
            raise ValueError(f"dataset_index={dataset_index} not in batch {requested}: {indices}")
        return int(requested), indices.index(dataset_index), indices
    for batch_index, indices in enumerate(batches):
        if dataset_index in indices:
            return batch_index, indices.index(dataset_index), indices
    raise RuntimeError(f"dataset_index={dataset_index} not found in train batches")


def clone_item(item: dict[str, Any], source: Any) -> dict[str, Any]:
    out = {
        key: value.detach().cpu().clone() if hasattr(value, "detach") else value
        for key, value in item.items()
    }
    if source is not None:
        out["_hidden_state_source"] = dict(source)
    return out


def wait_for_hidden_file(path_value: str, timeout: float) -> None:
    from speculators.data_generation.vllm_client import wait_for_lock

    path = Path(path_value)
    lock_path = Path(path_value + ".lock")
    deadline = time.monotonic() + timeout
    while lock_path.exists() or not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for hidden states file {path}")
        if lock_path.exists():
            wait_for_lock(str(lock_path), timeout=max(deadline - time.monotonic(), 0.1))
        else:
            time.sleep(0.05)


def delete_hidden_file(path_value: str | None) -> None:
    if path_value is None:
        return
    path = Path(path_value)
    path.unlink(missing_ok=True)
    Path(str(path) + ".lock").unlink(missing_ok=True)


def request_direct_scored_hidden(
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
    prompt_ids = _prompt_token_ids(response)
    prompt_logprob_obj = _prompt_logprobs(response)
    hidden_path = _kv_hidden_states_path(response)
    if prompt_ids is None:
        raise RuntimeError("direct response missing prompt_token_ids")
    if prompt_logprob_obj is None:
        raise RuntimeError("direct response missing prompt_logprobs")
    if hidden_path is None:
        raise RuntimeError("direct response missing hidden_states_path")
    wait_for_hidden_file(hidden_path, hidden_file_timeout)
    tensors = load_file(hidden_path)
    return {
        "prompt_ids": list(prompt_ids),
        "file_token_ids": tensors["token_ids"].detach().cpu().clone(),
        "hidden": tensors["hidden_states"].detach().cpu().clone(),
        "hidden_path": hidden_path,
        "token_logprobs": [
            _extract_token_logprob(prompt_logprob_obj, pos, prompt[pos])
            for pos in score_positions
        ],
    }


class FullVerifierHead:
    def __init__(self, model_path: str, device: Any, dtype: Any) -> None:
        import torch
        from transformers import AutoConfig
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        from speculators.utils.loading import load_model_layers

        config = AutoConfig.from_pretrained(model_path)
        if hasattr(config, "text_config"):
            config = config.text_config
        weights = load_model_layers(
            ["embed_tokens.weight", "lm_head.weight", "model.norm.weight"],
            model_path,
        )
        # Match DraftVocabMixin.load_verifier_weights(): Qwen-style tied heads may
        # not have a separate lm_head.weight key.
        lm_head_source = (
            "lm_head.weight" if "lm_head.weight" in weights else "embed_tokens.weight"
        )
        lm_head_weight = weights[lm_head_source].to(device=device, dtype=dtype)
        self.lm_head_weight = lm_head_weight
        self.lm_head_source = lm_head_source
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(
            device=device,
            dtype=dtype,
        )
        self.norm.load_state_dict({"weight": weights["model.norm.weight"].to(device=device, dtype=dtype)})
        self.device = device
        self.dtype = dtype
        self.torch = torch

    def project(
        self,
        hidden: Any,
        prompt: list[int],
        score_positions: list[int],
    ) -> list[float]:
        torch = self.torch
        hidden_positions = torch.tensor(
            [pos - 1 for pos in score_positions],
            dtype=torch.long,
            device=self.device,
        )
        target_ids = torch.tensor(
            [prompt[pos] for pos in score_positions],
            dtype=torch.long,
            device=self.device,
        )
        with torch.no_grad():
            selected = hidden.to(device=self.device, dtype=self.dtype).index_select(
                0,
                hidden_positions,
            )
            normed = self.norm(selected)
            logits = torch.nn.functional.linear(normed, self.lm_head_weight).float()
            logprobs = torch.log_softmax(logits, dim=-1)
            values = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return [float(x) for x in values.detach().cpu().tolist()]


def target_to_training_gather_ids(model: Any, target_ids: Any, device: Any) -> tuple[Any, Any]:
    import torch

    if model.t2d is None:
        return target_ids.to(device=device), torch.ones_like(
            target_ids,
            dtype=torch.bool,
            device=device,
        )
    t2d = model.t2d.to(device=device)
    target_ids = target_ids.to(device=device)
    in_vocab = t2d[target_ids].bool()
    target_to_rank = torch.cumsum(t2d.to(dtype=torch.long), dim=0) - 1
    gather_ids = target_to_rank[target_ids].clamp_min(0)
    return gather_ids, in_vocab


def project_training_verifier_head(
    *,
    model: Any,
    hidden: Any,
    prompt: list[int],
    score_positions: list[int],
) -> tuple[list[float], list[bool], list[int]]:
    import torch

    device = next(model.parameters()).device
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
    gather_ids, in_vocab = target_to_training_gather_ids(model, target_ids, device)
    with torch.no_grad():
        selected = hidden.to(device=device, dtype=next(model.parameters()).dtype).index_select(
            0,
            hidden_positions,
        )
        logits = model.verifier_lm_head(model.verifier_norm(selected)).float()
        logprobs = torch.log_softmax(logits, dim=-1)
        gathered = logprobs.gather(-1, gather_ids.unsqueeze(-1)).squeeze(-1)
        gathered = torch.where(
            in_vocab,
            gathered,
            torch.full_like(gathered, float("nan")),
        )
    return (
        [float(x) for x in gathered.detach().cpu().tolist()],
        [bool(x) for x in in_vocab.detach().cpu().tolist()],
        [int(x) for x in gather_ids.detach().cpu().tolist()],
    )


def main() -> None:
    args = parse_args()

    import torch
    import scripts.train as train_script
    from speculators.model import SpeculatorModel
    from speculators.models.dspark.core import DSparkDraftModel
    from speculators.train.dataloader import create_train_val_loaders
    from speculators.train.optimizers import build_optimizers
    from speculators.train.sampled_acceptance import (
        SampledAcceptanceAugmentor,
        SampledAcceptanceConfig,
    )
    from speculators.train.trainer import TrainerConfig
    from speculators.train.utils import normalize_counted_metrics

    torch.manual_seed(args.seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(torch, args.device)
    hidden_dtype = getattr(torch, args.hidden_states_dtype)
    train_args = make_train_args(args)

    print("TRACE config")
    print(f"  repo={ROOT}")
    print(f"  verifier_name_or_path={args.verifier_name_or_path}")
    print(f"  data_path={args.data_path}")
    print(f"  dataset_index={args.dataset_index}")
    print(f"  vllm_endpoint={args.vllm_endpoint}")
    print(f"  device={device}")
    print(f"  hidden_dtype={hidden_dtype}")
    print("  from_pretrained=<none:fresh_training>")
    print(f"  total_seq_len={args.total_seq_len}")
    print(f"  num_workers={args.num_workers}")
    print(f"  noise_std={args.noise_std}")
    print(f"  preserve_hidden_state_source={args.preserve_hidden_state_source}")

    registry = SpeculatorModel.registry
    if registry is None or "dspark" not in registry:
        raise RuntimeError("DSpark model is not registered")
    model_class = registry["dspark"]
    d2t, t2d, draft_vocab_size = train_script.parse_vocab_mappings(train_args)
    model = train_script.build_draft_model(
        train_args,
        model_class,
        t2d,
        d2t,
        draft_vocab_size,
    )
    if not isinstance(model, DSparkDraftModel):
        raise TypeError(f"expected DSparkDraftModel, got {type(model).__name__}")
    model.to(device=device, dtype=hidden_dtype)
    model.train()
    print("TRACE model")
    print(f"  class={type(model).__name__}")
    print(f"  block_size={int(model.block_size)}")
    print(f"  max_anchors={int(model.config.max_anchors)}")
    print(f"  draft_vocab_size={draft_vocab_size}")
    print(f"  target_layer_ids={list(model.target_layer_ids)}")
    print(f"  attn_impl={getattr(model, '_attn_impl', None)}")
    print(f"  mask_token_id={getattr(model.config, 'mask_token_id', None)}")
    print(f"  verifier_vocab_size={getattr(model, 'verifier_vocab_size', None)}")
    print(f"  use_draft_vocab={getattr(model, 'use_draft_vocab', None)}")
    if model.t2d is not None:
        print(f"  t2d_shape={tuple(model.t2d.shape)}")
        print(f"  t2d_selected_count={int(model.t2d.sum(dtype=torch.long).item())}")
    if model.d2t is not None:
        print(f"  d2t_shape={tuple(model.d2t.shape)}")
        print(f"  d2t_head={model.d2t[:16].detach().cpu().tolist()}")
    print(f"  verifier_lm_head_shape={tuple(model.verifier_lm_head.weight.shape)}")
    print(f"  verifier_norm_shape={tuple(model.verifier_norm.weight.shape)}")

    captured_items: list[dict[str, Any]] = []

    def capture_and_sanitize(item: dict[str, Any]) -> dict[str, Any]:
        source = item.get("_hidden_state_source")
        if not args.preserve_hidden_state_source:
            source = item.pop("_hidden_state_source", None)
        captured_items.append(clone_item(item, source))
        return item

    train_loader, _ = create_train_val_loaders(
        data_path=args.data_path,
        total_seq_len=args.total_seq_len,
        hidden_states_dtype=hidden_dtype,
        noise_std=args.noise_std,
        legacy_data=False,
        hidden_states_path=args.hidden_states_path,
        vllm_endpoint=args.vllm_endpoint,
        on_missing=args.on_missing,
        on_generate=args.on_generate,
        verifier_name_or_path=args.verifier_name_or_path,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        hidden_size=model.config.transformer_layer_config.hidden_size,
        num_target_layers=len(model.target_layer_ids),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        preprocess=capture_and_sanitize,
    )
    if args.num_workers != 0:
        print("TRACE warning num_workers_nonzero raw item capture order may be unreliable")

    batch_index, doc_id, batch_indices = find_target_batch(
        train_loader,
        args.dataset_index,
        args.batch_index,
    )
    print("TRACE batch_locator")
    print(f"  batch_index={batch_index}")
    print(f"  batch_indices={batch_indices}")
    print(f"  target_doc_id={doc_id}")

    batch = None
    for idx, candidate in enumerate(train_loader):
        if idx == batch_index:
            batch = candidate
            break
    if batch is None:
        raise RuntimeError(f"failed to read batch {batch_index}")

    raw_offset = sum(len(x) for x in batch_indices_for_loader(train_loader)[:batch_index])
    raw_capture_index = raw_offset + doc_id
    raw_item = captured_items[raw_capture_index]
    raw_ids = raw_item["input_ids"].detach().cpu().tolist()
    document_ids = batch["document_ids"]
    doc_positions = (document_ids[0] == doc_id).nonzero(as_tuple=False).flatten()
    if doc_positions.numel() == 0:
        raise RuntimeError(f"doc_id={doc_id} not present in batch document_ids")
    doc_start = int(doc_positions[0].item())
    local_hidden_start = args.local_start
    local_hidden_end = args.local_start + args.gt_len
    packed_hidden_start = doc_start + local_hidden_start
    packed_hidden_end = doc_start + local_hidden_end
    score_positions = list(range(args.local_start + 1, args.local_start + args.gt_len + 1))
    packed_score_positions = [doc_start + pos for pos in score_positions]

    print("TRACE batch")
    for key, value in batch.items():
        if hasattr(value, "shape"):
            print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")
    print(f"  raw_item_len={len(raw_ids)}")
    print(f"  doc_start={doc_start}")
    print(f"  local_hidden_window={local_hidden_start}:{local_hidden_end}")
    print(f"  packed_hidden_window={packed_hidden_start}:{packed_hidden_end}")
    print(f"  local_score_positions={score_positions}")
    print(f"  packed_score_positions={packed_score_positions}")
    print(f"  hidden_state_source={raw_item.get('_hidden_state_source')}")
    print(f"  captured_items_count={len(captured_items)}")
    print(f"  raw_capture_index={raw_capture_index}")
    packed_tokens = batch["input_ids"][0, packed_hidden_start : packed_hidden_end + 1].detach().cpu().tolist()
    raw_tokens = raw_ids[local_hidden_start : local_hidden_end + 1]
    print("TRACE token_alignment")
    print(f"  packed_matches_raw={packed_tokens == raw_tokens}")
    print(f"  raw_tokens={raw_tokens}")
    print(f"  packed_tokens={packed_tokens}")

    import openai

    client = openai.OpenAI(base_url=args.vllm_endpoint, api_key="EMPTY", max_retries=0)
    model_id = client.models.list().data[0].id
    direct = request_direct_scored_hidden(
        endpoint=args.vllm_endpoint,
        model_id=model_id,
        prompt=raw_ids,
        score_positions=score_positions,
        prompt_logprobs=args.prompt_logprobs,
        request_timeout=args.request_timeout,
        hidden_file_timeout=args.hidden_file_timeout,
    )
    print("TRACE direct_vllm")
    print(f"  model_id={model_id}")
    print(f"  hidden_path={direct['hidden_path']}")
    print(f"  prompt_ids_match={direct['prompt_ids'] == raw_ids}")
    print(f"  file_token_ids_match={direct['file_token_ids'].tolist() == raw_ids}")
    print(f"  hidden_shape={tuple(direct['hidden'].shape)}")
    print(f"  prompt_logprobs={[fmt(x) for x in direct['token_logprobs']]}")

    batch_last = batch["verifier_last_hidden_states"][0]
    raw_last = raw_item["verifier_last_hidden_states"]
    direct_final = direct["hidden"][:, -1]
    batch_window = batch_last[packed_hidden_start:packed_hidden_end]
    raw_window = raw_last[local_hidden_start:local_hidden_end]
    direct_window = direct_final[local_hidden_start:local_hidden_end]
    raw_batch_max, raw_batch_mean = diff_stats(batch_window, raw_window)
    batch_direct_max, batch_direct_mean = diff_stats(batch_window, direct_window)
    raw_direct_max, raw_direct_mean = diff_stats(raw_window, direct_window)
    print("TRACE hidden_compare_before_step")
    print(f"  packed_vs_raw_max_abs={fmt(raw_batch_max)}")
    print(f"  packed_vs_raw_mean_abs={fmt(raw_batch_mean)}")
    print(f"  packed_vs_direct_max_abs={fmt(batch_direct_max)}")
    print(f"  packed_vs_direct_mean_abs={fmt(batch_direct_mean)}")
    print(f"  raw_vs_direct_max_abs={fmt(raw_direct_max)}")
    print(f"  raw_vs_direct_mean_abs={fmt(raw_direct_mean)}")

    batch_prompt = batch["input_ids"][0].detach().cpu().tolist()
    (
        train_batch_plog,
        train_in_vocab,
        train_gather_ids,
    ) = project_training_verifier_head(
        model=model,
        hidden=batch_last,
        prompt=batch_prompt,
        score_positions=packed_score_positions,
    )
    train_raw_plog, _, _ = project_training_verifier_head(
        model=model,
        hidden=raw_last,
        prompt=raw_ids,
        score_positions=score_positions,
    )
    train_direct_plog, _, _ = project_training_verifier_head(
        model=model,
        hidden=direct_final,
        prompt=raw_ids,
        score_positions=score_positions,
    )
    print("TRACE training_verifier_projection_before_step")
    print(
        "  local_score_pos,target_id,in_draft_vocab,gather_id,"
        "batch_pruned,raw_pruned,direct_pruned"
    )
    for i, pos in enumerate(score_positions):
        print(
            "  "
            f"{pos},"
            f"{raw_ids[pos]},"
            f"{train_in_vocab[i]},"
            f"{train_gather_ids[i]},"
            f"{fmt(train_batch_plog[i])},"
            f"{fmt(train_raw_plog[i])},"
            f"{fmt(train_direct_plog[i])}"
        )

    full_head = FullVerifierHead(args.verifier_name_or_path, device, hidden_dtype)
    print("TRACE full_verifier_head")
    print(f"  lm_head_source={full_head.lm_head_source}")
    print(f"  lm_head_shape={tuple(full_head.lm_head_weight.shape)}")
    raw_plog = full_head.project(raw_last, raw_ids, score_positions)
    batch_plog = full_head.project(batch_last, batch_prompt, packed_score_positions)
    direct_plog = full_head.project(direct_final, raw_ids, score_positions)
    print("TRACE full_verifier_projection_before_step")
    print("  local_score_pos,target_id,batch_full,raw_full,direct_hidden_full,direct_prompt,batch_minus_prompt")
    for i, pos in enumerate(score_positions):
        print(
            "  "
            f"{pos},"
            f"{raw_ids[pos]},"
            f"{fmt(batch_plog[i])},"
            f"{fmt(raw_plog[i])},"
            f"{fmt(direct_plog[i])},"
            f"{fmt(direct['token_logprobs'][i])},"
            f"{fmt(batch_plog[i] - direct['token_logprobs'][i])}"
        )
    print(f"  batch_projection_vs_direct_prompt_max_abs={fmt(max_abs_delta(batch_plog, direct['token_logprobs']))}")
    print(f"  direct_hidden_projection_vs_direct_prompt_max_abs={fmt(max_abs_delta(direct_plog, direct['token_logprobs']))}")

    gpu_batch = {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    print("TRACE gpu_batch")
    for key, value in gpu_batch.items():
        if hasattr(value, "shape"):
            print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype} device={value.device}")
    augmentor = SampledAcceptanceAugmentor(
        SampledAcceptanceConfig(
            vllm_endpoint=args.vllm_endpoint,
            model=args.verifier_name_or_path,
            prompt_logprobs=args.prompt_logprobs,
            request_timeout=args.request_timeout,
            hidden_states_file_timeout=args.hidden_file_timeout,
            temperature=args.temperature,
        )
    )
    before_keys = set(gpu_batch)
    gpu_batch = augmentor(model, gpu_batch)
    print("TRACE augmentor")
    print(f"  added_keys={sorted(set(gpu_batch) - before_keys)}")
    for key in sorted(set(gpu_batch) - before_keys):
        value = gpu_batch[key]
        if hasattr(value, "shape"):
            preview = value.detach().cpu().flatten()[:16].tolist()
            print(
                f"  {key}: shape={tuple(value.shape)} dtype={value.dtype} "
                f"device={value.device} head={preview}"
            )
        else:
            print(f"  {key}: {value}")
    if "sampled_anchor_pos" in gpu_batch:
        anchor_pos = int(gpu_batch["sampled_anchor_pos"][0].detach().cpu().item())
        anchor_doc = int(gpu_batch["document_ids"][0, anchor_pos].detach().cpu().item())
        anchor_doc_positions = (
            gpu_batch["document_ids"][0, : anchor_pos + 1] == anchor_doc
        ).nonzero(as_tuple=False).flatten()
        anchor_doc_start = int(anchor_doc_positions[0].detach().cpu().item())
        anchor_local_pos = anchor_pos - anchor_doc_start
        print(f"  sampled_anchor_pos={anchor_pos}")
        print(f"  sampled_anchor_doc_id={anchor_doc}")
        print(f"  sampled_anchor_doc_start={anchor_doc_start}")
        print(f"  sampled_anchor_local_pos={anchor_local_pos}")
        print(
            "  sampled_anchor_token="
            f"{int(gpu_batch['input_ids'][0, anchor_pos].detach().cpu().item())}"
        )
        print(f"  sampled_target_ids={gpu_batch['sampled_target_ids'][0].detach().cpu().tolist()}")
        print(f"  sampled_target_logprobs={[fmt(float(x)) for x in gpu_batch['sampled_target_logprobs'][0].detach().cpu().tolist()]}")

    train_call_kwargs, _ = model_class.get_trainer_kwargs(**vars(train_args))
    print("TRACE forward_kwargs")
    for key, value in train_call_kwargs.items():
        print(f"  {key}={value}")

    optim_config = TrainerConfig(
        lr=args.lr,
        num_epochs=1,
        save_path="/tmp/dspark_single_step_debug_checkpoints",
        resume_from_checkpoint=False,
        train_call_kwargs=train_call_kwargs,
        optimizer="adamw",
        weight_decay=args.weight_decay,
        scheduler_type="none",
        hidden_states_dtype=hidden_dtype,
        batch_augmentor=augmentor,
    )
    optimizers = build_optimizers(model, optim_config)
    print("TRACE optimizer")
    for idx, optimizer in enumerate(optimizers):
        param_count = sum(
            param.numel()
            for group in optimizer.param_groups
            for param in group["params"]
            if getattr(param, "requires_grad", False)
        )
        print(
            f"  optimizer_{idx}={type(optimizer).__name__} "
            f"lr={optimizer.param_groups[0]['lr']} "
            f"weight_decay={optimizer.param_groups[0].get('weight_decay')} "
            f"trainable_param_count={param_count}"
        )
    for optimizer in optimizers:
        optimizer.zero_grad()
    _draft_tokens, loss, metrics = model(**gpu_batch, **train_call_kwargs)
    print("TRACE forward")
    print(f"  loss={fmt(float(loss.detach().float().cpu().item()))}")
    normalized = normalize_counted_metrics(
        {key: float(value.detach().float().cpu().item()) for key, value in metrics.items()},
        world_size=1,
    )
    for key in sorted(normalized):
        if key == "loss" or key.startswith("sampled_") or key in {"ce_loss", "tv_loss", "confidence_loss"}:
            print(f"  metric/{key}={fmt(float(normalized[key]))}")

    if args.skip_backward:
        print("TRACE backward skipped=true")
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for optimizer in optimizers:
            optimizer.step()
        print("TRACE backward")
        print(f"  grad_norm={fmt(float(grad_norm.detach().float().cpu().item() if hasattr(grad_norm, 'detach') else grad_norm))}")
        print("  optimizer_step=true")

    print("TRACE conclusion")
    batch_vs_prompt = max_abs_delta(batch_plog, direct["token_logprobs"])
    if batch_vs_prompt <= args.logprob_tol and batch_direct_max <= args.hidden_tol:
        print("  status=batch_hidden_matches_direct_hidden_and_prompt")
    elif batch_vs_prompt <= args.logprob_tol:
        print("  status=batch_hidden_projects_to_prompt_but_differs_numerically")
    else:
        print("  status=batch_hidden_does_not_reconstruct_direct_prompt")

    if not args.keep_direct_hidden_file:
        delete_hidden_file(direct["hidden_path"])


if __name__ == "__main__":
    main()
