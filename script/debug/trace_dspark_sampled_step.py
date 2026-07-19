#!/usr/bin/env python3
"""Trace one real DSpark sampled-acceptance training step.

This intentionally follows the training path:

    dataloader -> batch.to(device) -> SampledAcceptanceAugmentor
    -> vLLM sampled-token scoring -> model.forward -> optional backward/step

The trace wraps the existing code at runtime and prints the variables that can
explain sampled-acceptance failures: selected anchors, sampled tokens, draft
q-logprobs during sampling, target p-logprobs returned by vLLM, replay q-logprobs
inside the training forward, and the exact raw/normalized train metrics.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
from pathlib import Path
import sys
from types import MethodType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_ACTIVE_TRACE_BATCH: dict[str, Any] = {}


def _fmt(x: float) -> str:
    if math.isnan(x) or math.isinf(x):
        return str(x)
    if x == 0 or (1e-3 <= abs(x) < 1e4):
        return f"{x:.6f}"
    return f"{x:.6e}"


def _scalar(x: Any) -> float:
    if hasattr(x, "detach"):
        return float(x.detach().float().cpu().item())
    return float(x)


def _values(t: Any, limit: int = 16) -> list[Any]:
    if t is None:
        return []
    return t.detach().cpu().flatten()[:limit].tolist()


def _shape(t: Any) -> str:
    return f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device}"


def _print_tensor(name: str, t: Any, *, limit: int = 16) -> None:
    print(f"  {name}: {_shape(t)} values={_values(t, limit)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and trace one real DSpark sampled-acceptance training step."
    )
    parser.add_argument("--verifier-name-or-path", default="/models/Qwen3-4B")
    parser.add_argument("--data-path", default="/data/open_perfectblend_qwen3_4b_100k")
    parser.add_argument("--hidden-states-path", default=None)
    parser.add_argument("--vllm-endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--save-path", default="/tmp/dspark_trace_checkpoints")
    parser.add_argument(
        "--from-pretrained",
        default="",
        help="Load a saved Speculators checkpoint instead of random draft weights.",
    )
    parser.add_argument(
        "--draft-config",
        default="",
        help="Optional decoder config path, matching scripts/train.py.",
    )
    parser.add_argument("--total-seq-len", type=int, default=3072)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--max-anchors", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--draft-vocab-size", type=int, default=32000)
    parser.add_argument("--target-layer-ids", type=int, nargs="+", default=[1, 9, 17, 25, 33])
    parser.add_argument("--draft-attn-impl", default="sdpa")
    parser.add_argument("--markov-rank", type=int, default=256)
    parser.add_argument("--markov-head-type", default="vanilla")
    parser.add_argument("--loss-fn", default='{"ce": 0.1, "tv": 0.9}')
    parser.add_argument("--confidence-head-alpha", type=float, default=1.0)
    parser.add_argument("--sampled-acceptance-loss-alpha", type=float, default=1.0)
    parser.add_argument("--hidden-states-dtype", default="bfloat16")
    parser.add_argument("--on-missing", choices=["generate", "skip", "warn", "raise"], default="generate")
    parser.add_argument("--on-generate", choices=["cache", "delete"], default="delete")
    parser.add_argument("--request-timeout", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--legacy-data", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Default: npu/cuda accelerator if available, else cpu")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--print-anchor-limit", type=int, default=24)
    parser.add_argument("--gt-compare-len", type=int, default=7)
    parser.add_argument(
        "--hf-hidden-compare",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare gt_doc_prefix vLLM connector hidden states against direct HF.",
    )
    parser.add_argument(
        "--vllm-hidden-repeat",
        type=int,
        default=1,
        help="Extra identical gt_doc_prefix hidden requests for repeat-diff checks.",
    )
    parser.add_argument(
        "--anchor-selection",
        choices=["first", "prefer-packed", "require-packed"],
        default="prefer-packed",
        help=(
            "Debug anchor choice. prefer-packed moves a doc_start>0 anchor to "
            "ordinal 0 when available; require-packed errors if none exists."
        ),
    )
    parser.add_argument("--skip-backward", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Print full trace log.")
    parser.add_argument("--lr", type=float, default=6e-4)
    return parser


def _make_train_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "speculator_type": "dspark",
        "draft_arch": "qwen3",
        "draft_hidden_act": None,
        "sliding_window": 2048,
        "sliding_window_indices": [],
        "sliding_window_non_causal": False,
        "mask_token_id": None,
        "d2t_path": None,
        "t2d_path": None,
        "epochs": 1,
        "log_dir": "/tmp/dspark_trace_logs",
        "run_name": None,
        "dry_run": False,
        "deterministic_cuda": False,
        "enable_confidence_head": True,
        "confidence_head_with_markov": True,
        "cat_mode": "none",
        "dflash_decay_gamma": 4.0,
        "norm_before_fc": False,
        "norm_output": False,
        "num_speculative_steps": 0,
        "ttt_steps": 3,
        "ttt_step_loss_decay": 1.0,
        "num_depths": 8,
        "down_sample_ratio": 0.7,
        "down_sample_ratio_min": 0.2,
        "checkpoint_freq": 1.0,
        "save_best": False,
        "scheduler_type": "linear",
        "scheduler_warmup_steps": None,
        "scheduler_total_steps": None,
        "scheduler_num_cosine_cycles": 0.5,
        "optimizer": "adamw",
        "weight_decay": 0.01,
        "muon_lr": 0.02,
        "muon_momentum": 0.95,
        "muon_weight_decay": 0.1,
        "muon_ns_steps": 5,
        "muon_adjust_lr_fn": "match_rms_adamw",
    }
    out = argparse.Namespace(**vars(args))
    for key, value in defaults.items():
        if not hasattr(out, key):
            setattr(out, key, value)
    return out


def _resolve_device(torch: Any, device_arg: str | None):
    if device_arg:
        return torch.device(device_arg)
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _force_npu_supported_attention(model: Any, device: Any) -> None:
    """Switch checkpoint flex attention to SDPA masks for NPU-only tracing."""
    if getattr(device, "type", str(device).split(":", 1)[0]) != "npu":
        return
    if getattr(model, "_attn_impl", None) != "simple_flex_attention":
        return

    from torch.nn.attention.flex_attention import create_mask

    print("TRACE npu_attention_override")
    print("  checkpoint_attn_impl=simple_flex_attention")
    print("  override_attn_impl=sdpa")
    print("  reason=PyTorch FlexAttention does not support npu tensors")

    model._attn_impl = "sdpa"
    model._create_mask_fn = create_mask
    model.config.transformer_layer_config._attn_implementation = "sdpa"  # noqa: SLF001
    for layer in getattr(model, "layers", []):
        if hasattr(layer, "config"):
            layer.config._attn_implementation = "sdpa"  # noqa: SLF001
        if hasattr(layer, "self_attn"):
            layer.self_attn.config._attn_implementation = "sdpa"  # noqa: SLF001


def _install_score_trace() -> tuple[list[dict[str, Any]], Any]:
    import speculators.train.sampled_acceptance as sampled_acceptance_mod

    calls: list[dict[str, Any]] = []
    original = sampled_acceptance_mod.score_sampled_tokens

    def wrapped(*args: Any, **kwargs: Any):
        prompt = [
            *list(kwargs.get("prefix_token_ids", [])),
            *list(kwargs.get("sampled_token_ids", [])),
        ]
        print("TRACE vllm.score_sampled_tokens request")
        print(f"  model={kwargs.get('model')}")
        print(f"  prefix_len={len(kwargs.get('prefix_token_ids', []))}")
        print(f"  sampled_len={len(kwargs.get('sampled_token_ids', []))}")
        print(f"  prompt_len={len(prompt)}")
        print(f"  prompt_tail={prompt[-24:]}")
        print(f"  sampled_token_ids={list(kwargs.get('sampled_token_ids', []))}")
        result = original(*args, **kwargs)
        calls.append({"kwargs": kwargs, "prompt": prompt, "result": result})
        print("TRACE vllm.score_sampled_tokens response")
        print(f"  score_positions={result['score_positions']}")
        print(f"  token_logprobs={[float(x) for x in result['token_logprobs']]}")
        print(f"  hidden_states_path={result['hidden_states_path']}")
        print(f"  hidden_states_deleted={result['hidden_states_deleted']}")
        return result

    sampled_acceptance_mod.score_sampled_tokens = wrapped
    return calls, original


def _restore_score_trace(original: Any) -> None:
    import speculators.train.sampled_acceptance as sampled_acceptance_mod

    sampled_acceptance_mod.score_sampled_tokens = original


def _doc_start_for_position(document_ids: Any, pos: int) -> int:
    anchor_doc = document_ids[0, pos]
    same_doc = document_ids[0, : pos + 1] == anchor_doc
    positions = same_doc.nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        return 0
    return int(positions[0].item())


def _block_stays_in_document(document_ids: Any, pos: int, block_size: int) -> bool:
    if pos + block_size > int(document_ids.shape[1]):
        return False
    anchor_doc = document_ids[0, pos]
    if int(anchor_doc.detach().cpu().item()) == -1:
        return False
    block_docs = document_ids[0, pos : pos + block_size]
    return bool((block_docs == anchor_doc).all().detach().cpu().item())


def _first_packed_anchor_position(
    *,
    loss_mask: Any,
    document_ids: Any,
    block_size: int,
) -> int | None:
    valid = loss_mask[0].bool().clone()
    if block_size > valid.numel():
        valid.zero_()
    else:
        valid[valid.numel() - block_size + 1 :] = False
    valid_positions = valid.nonzero(as_tuple=False).flatten()
    for pos_tensor in valid_positions.detach().cpu().tolist():
        pos = int(pos_tensor)
        if (
            _doc_start_for_position(document_ids, pos) > 0
            and _block_stays_in_document(document_ids, pos, block_size)
        ):
            return pos
    return None


def _install_anchor_selection_trace(mode: str) -> Any:
    import speculators.train.sampled_acceptance as sampled_acceptance_mod

    original = sampled_acceptance_mod.select_anchors

    def wrapped(loss_mask: Any, num_anchors: int, block_size: int, **kwargs: Any):
        anchors, anchor_valid = original(
            loss_mask,
            num_anchors,
            block_size,
            **kwargs,
        )
        if mode == "first":
            print("TRACE anchor_selection mode=first")
            return anchors, anchor_valid

        document_ids = _ACTIVE_TRACE_BATCH.get("document_ids")
        if document_ids is None:
            message = "TRACE anchor_selection no document_ids available"
            if mode == "require-packed":
                raise RuntimeError(message)
            print(message)
            return anchors, anchor_valid

        packed_anchor = _first_packed_anchor_position(
            loss_mask=loss_mask,
            document_ids=document_ids,
            block_size=block_size,
        )
        if packed_anchor is None:
            message = "TRACE anchor_selection no doc_start>0 valid anchor in batch"
            if mode == "require-packed":
                raise RuntimeError(message)
            print(f"{message}; falling back to ordinal 0")
            return anchors, anchor_valid

        device = anchors.device
        packed_anchor_tensor = anchors.new_tensor(packed_anchor)
        matches = (anchors == packed_anchor_tensor).nonzero(as_tuple=False).flatten()
        if matches.numel() > 0:
            src = int(matches[0].item())
            anchors[[0, src]] = anchors[[src, 0]]
            anchor_valid[[0, src]] = anchor_valid[[src, 0]]
        else:
            anchors[0] = packed_anchor_tensor.to(device=device)
            anchor_valid[0] = True

        doc_start = _doc_start_for_position(document_ids, packed_anchor)
        doc_id = int(document_ids[0, packed_anchor].detach().cpu().item())
        print("TRACE anchor_selection")
        print(f"  mode={mode}")
        print(f"  selected_anchor_pos={packed_anchor}")
        print(f"  selected_doc_id={doc_id}")
        print(f"  selected_doc_start={doc_start}")
        print(f"  selected_doc_prefix_len={packed_anchor - doc_start + 1}")
        return anchors, anchor_valid

    sampled_acceptance_mod.select_anchors = wrapped
    return original


def _restore_anchor_selection(original: Any) -> None:
    import speculators.train.sampled_acceptance as sampled_acceptance_mod

    sampled_acceptance_mod.select_anchors = original


def _install_model_traces(model: Any, *, topk: int) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "backbone_calls": [],
        "recompute_calls": [],
    }
    original_backbone = model._backbone_forward
    original_recompute = model._recompute_sampled_qlogp

    def traced_backbone(
        self: Any,
        hidden_states: Any,
        input_ids: Any,
        loss_mask: Any,
        verifier_last_hidden_states: Any,
        document_ids: Any,
        position_ids: Any = None,
        anchor_positions: Any = None,
        anchor_valid: Any = None,
        **kwargs: Any,
    ):
        call_id = len(trace["backbone_calls"]) + 1
        provided_positions = None if anchor_positions is None else anchor_positions.detach().cpu().view(-1)
        config_max_before = int(self.config.max_anchors)
        result = original_backbone(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            position_ids,
            anchor_positions=anchor_positions,
            anchor_valid=anchor_valid,
            **kwargs,
        )
        hidden, logits, targets, aligned_loss_mask, anchored_block_indices = result
        block = int(self.block_size)
        block_heads = anchored_block_indices.detach().cpu().view(-1, block)[:, 0]
        valid_count = None if anchor_valid is None else int(anchor_valid.detach().bool().sum().cpu().item())
        info = {
            "call_id": call_id,
            "config_max_anchors": config_max_before,
            "provided_positions": provided_positions,
            "provided_valid_count": valid_count,
            "output_block_heads": block_heads,
            "hidden_shape": tuple(hidden.shape),
            "logits_shape": tuple(logits.shape),
            "targets_shape": tuple(targets.shape),
            "aligned_loss_mask_sum": _scalar(aligned_loss_mask.sum()),
        }
        trace["backbone_calls"].append(info)
        print(f"TRACE backbone_forward[{call_id}]")
        print(f"  config.max_anchors={config_max_before} block_size={block}")
        print(f"  provided_anchor_positions={None if provided_positions is None else provided_positions[:24].tolist()}")
        print(f"  provided_anchor_valid_count={valid_count}")
        print(f"  output_block_heads={block_heads[:24].tolist()}")
        print(f"  aligned_loss_mask_sum={_fmt(info['aligned_loss_mask_sum'])}")
        print(f"  hidden={info['hidden_shape']} logits={info['logits_shape']} targets={info['targets_shape']}")
        return result

    def traced_recompute(self: Any, *args: Any, **kwargs: Any):
        hidden = kwargs["hidden"]
        logits_base = kwargs["logits_base"]
        input_ids = kwargs["input_ids"]
        anchored_block_indices = kwargs["anchored_block_indices"]
        sampled_draft_ids = kwargs["sampled_draft_ids"].to(device=hidden.device).long().view(1, -1)
        sampled_target_ids = kwargs["sampled_target_ids"].to(device=hidden.device).long().view(1, -1)
        sampled_target_logprobs = kwargs["sampled_target_logprobs"].to(device=hidden.device).float().view(1, -1)
        sampled_anchor_pos = kwargs["sampled_anchor_pos"].to(device=hidden.device).long().view(-1)[0]
        sampled_anchor_index = kwargs["sampled_anchor_index"].to(device=hidden.device).long().view(-1)[0]
        num_blocks = int(kwargs["num_blocks"])
        block = int(kwargs["block"])
        K = min(int(sampled_draft_ids.shape[-1]), block - 1)

        block_positions = anchored_block_indices.view(num_blocks, block)
        anchor_position = block_positions[sampled_anchor_index, 0]
        anchor_token = input_ids[0].gather(0, anchor_position.view(1))[0]
        prev_token_ids = anchor_token.expand(1, block).clone()
        if K > 1:
            prev_token_ids[:, 2 : K + 1] = sampled_target_ids[:, : K - 1]

        q_logp, p_logp = original_recompute(*args, **kwargs)
        log_alpha = (p_logp.float() - q_logp.float()).clamp(max=0)
        alpha = log_alpha.exp()
        survival = log_alpha.cumsum(dim=-1).exp()
        undercovered = (q_logp.float() < p_logp.float()).float()

        info = {
            "sampled_anchor_index": int(sampled_anchor_index.detach().cpu().item()),
            "sampled_anchor_pos": int(sampled_anchor_pos.detach().cpu().item()),
            "block_anchor_pos": int(anchor_position.detach().cpu().item()),
            "anchor_token": int(anchor_token.detach().cpu().item()),
            "prev_token_ids": prev_token_ids.detach().cpu(),
            "sampled_draft_ids": sampled_draft_ids[:, :K].detach().cpu(),
            "sampled_target_ids": sampled_target_ids[:, :K].detach().cpu(),
            "q_logp": q_logp.detach().cpu(),
            "p_logp": p_logp.detach().cpu(),
            "alpha": alpha.detach().cpu(),
            "survival": survival.detach().cpu(),
            "undercovered": undercovered.detach().cpu(),
            "logits_base_shape": tuple(logits_base.shape),
        }
        trace["recompute_calls"].append(info)
        print("TRACE training_forward.sampled_replay")
        print(f"  sampled_anchor_index={info['sampled_anchor_index']}")
        print(f"  sampled_anchor_pos={info['sampled_anchor_pos']}")
        print(f"  block_anchor_pos={info['block_anchor_pos']}")
        print(f"  anchor_token={info['anchor_token']}")
        print(f"  prev_token_ids={info['prev_token_ids'].flatten().tolist()}")
        print(f"  sampled_draft_ids={info['sampled_draft_ids'].flatten().tolist()}")
        print(f"  sampled_target_ids={info['sampled_target_ids'].flatten().tolist()}")
        print("  pos,replay_qlogp,target_plogp,alpha,survival,undercovered")
        for i in range(K):
            print(
                f"  {i + 1},"
                f"{_fmt(_scalar(q_logp[0, i]))},"
                f"{_fmt(_scalar(p_logp[0, i]))},"
                f"{_fmt(_scalar(alpha[0, i]))},"
                f"{_fmt(_scalar(survival[0, i]))},"
                f"{_fmt(_scalar(undercovered[0, i]))}"
            )
        return q_logp, p_logp

    model._backbone_forward = MethodType(traced_backbone, model)
    model._recompute_sampled_qlogp = MethodType(traced_recompute, model)
    trace["restore"] = lambda: (
        setattr(model, "_backbone_forward", original_backbone),
        setattr(model, "_recompute_sampled_qlogp", original_recompute),
    )
    return trace


def _install_sampling_trace(augmentor: Any, *, topk: int) -> dict[str, Any]:
    import torch

    trace: dict[str, Any] = {}

    def traced_sample_from_draft(
        self: Any,
        model: Any,
        batch: dict[str, Any],
        anchor_pos: int | None = None,
    ) -> dict[str, Any]:
        device = next(model.parameters()).device
        input_ids = batch["input_ids"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        hidden_states = batch["hidden_states"].to(device)
        verifier_last_hidden_states = batch["verifier_last_hidden_states"].to(device)
        document_ids = batch["document_ids"].to(device)
        position_ids = batch.get("position_ids")
        if position_ids is not None:
            position_ids = position_ids.to(device)

        block_size = int(model.block_size)
        if anchor_pos is None:
            anchor_pos = self._sample_anchor_position(loss_mask, block_size)
        sampled_len = block_size - 1
        anchor_loss_mask = torch.zeros_like(loss_mask, dtype=torch.bool)
        anchor_loss_mask[:, anchor_pos] = True
        anchor_positions = torch.tensor([anchor_pos], dtype=torch.long, device=device)
        anchor_valid = torch.ones(1, dtype=torch.bool, device=device)

        print("TRACE draft_sampling.begin")
        print(f"  model.config.max_anchors(before)={int(model.config.max_anchors)}")
        print(f"  augmentor.config.max_anchors={int(self.config.max_anchors)}")
        print(f"  anchor_pos={anchor_pos}")
        print(f"  anchor_token={int(input_ids[0, anchor_pos].detach().cpu().item())}")
        print(f"  prefix_tail={input_ids[0, max(0, anchor_pos - 16) : anchor_pos + 1].detach().cpu().tolist()}")

        old_max_anchors = model.config.max_anchors
        model.config.max_anchors = self.config.max_anchors
        try:
            hidden, logits, _, _, _ = model.get_backbone_outputs(
                hidden_states,
                input_ids,
                anchor_loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                anchor_positions=anchor_positions,
                anchor_valid=anchor_valid,
            )
        finally:
            model.config.max_anchors = old_max_anchors

        hidden_blocks = hidden.view(1, block_size, -1)
        logits_blocks = logits.view(1, block_size, -1)
        anchor_token_id = int(input_ids[0, anchor_pos].item())

        sampled_target_ids: list[int] = []
        sampled_draft_ids: list[int] = []
        draft_logprobs: list[Any] = []
        slot_trace: list[dict[str, Any]] = []
        for slot in range(1, sampled_len + 1):
            prev_token_ids = torch.full(
                (1, block_size),
                anchor_token_id,
                dtype=torch.long,
                device=device,
            )
            if sampled_target_ids:
                prev_token_ids[0, 2 : 2 + len(sampled_target_ids)] = torch.tensor(
                    sampled_target_ids[: block_size - 2],
                    dtype=torch.long,
                    device=device,
                )

            biased_logits = logits_blocks
            if model.markov_head is not None:
                prev_emb = model.markov_head.prev_embeddings(prev_token_ids)
                biased_logits = biased_logits + model.markov_head.block_bias(
                    prev_token_ids=prev_token_ids,
                    hidden_states=hidden_blocks,
                    prev_emb=prev_emb,
                )

            slot_logits = biased_logits[0, slot].float()
            if self.config.temperature <= 0:
                scaled = slot_logits
                log_probs = torch.log_softmax(slot_logits, dim=-1)
                draft_token_id = int(torch.argmax(slot_logits).item())
            else:
                scaled = slot_logits / self.config.temperature
                log_probs = torch.log_softmax(scaled, dim=-1)
                draft_token_id = int(
                    torch.multinomial(torch.softmax(scaled, dim=-1), 1).item()
                )
            target_token_id = self._target_token_id(model, draft_token_id)
            top_vals, top_ids = torch.topk(log_probs, k=min(topk, log_probs.numel()))
            q_logp = log_probs[draft_token_id]

            sampled_draft_ids.append(draft_token_id)
            sampled_target_ids.append(target_token_id)
            draft_logprobs.append(q_logp.to(logits.dtype))
            slot_info = {
                "slot": slot,
                "prev_token_ids": prev_token_ids.detach().cpu(),
                "draft_token_id": draft_token_id,
                "target_token_id": target_token_id,
                "q_logp": float(q_logp.detach().cpu().item()),
                "top_ids": top_ids.detach().cpu().tolist(),
                "top_logprobs": top_vals.detach().cpu().tolist(),
                "slot_logits_min": float(slot_logits.min().detach().cpu().item()),
                "slot_logits_max": float(slot_logits.max().detach().cpu().item()),
                "slot_logits_mean": float(slot_logits.mean().detach().cpu().item()),
                "scaled_logits_max": float(scaled.max().detach().cpu().item()),
            }
            slot_trace.append(slot_info)
            print(
                "TRACE draft_sampling.slot "
                f"slot={slot} prev={slot_info['prev_token_ids'].flatten().tolist()} "
                f"draft_id={draft_token_id} target_id={target_token_id} "
                f"q_logp={_fmt(slot_info['q_logp'])} "
                f"top_ids={slot_info['top_ids']} "
                f"top_logprobs={[float(x) for x in slot_info['top_logprobs']]}"
            )

        prefix_start = self._document_prefix_start(document_ids, anchor_pos)
        prefix_token_ids = input_ids[0, prefix_start : anchor_pos + 1].tolist()
        result = {
            "prefix_token_ids": prefix_token_ids,
            "sampled_target_token_ids": sampled_target_ids,
            "sampled_draft_token_ids": sampled_draft_ids,
            "draft_logprobs": torch.stack(draft_logprobs),
            "anchor_positions": anchor_positions,
            "anchor_valid": anchor_valid,
        }
        trace.update(
            {
                "anchor_pos": anchor_pos,
                "anchor_token": anchor_token_id,
                "prefix_start": prefix_start,
                "prefix_len": len(prefix_token_ids),
                "sampled_target_ids": list(sampled_target_ids),
                "sampled_draft_ids": list(sampled_draft_ids),
                "draft_logprobs": result["draft_logprobs"].detach().cpu(),
                "slot_trace": slot_trace,
            }
        )
        print("TRACE draft_sampling.end")
        print(f"  prefix_start={prefix_start}")
        print(f"  prefix_len={len(prefix_token_ids)}")
        print(f"  sampled_draft_ids={sampled_draft_ids}")
        print(f"  sampled_target_ids={sampled_target_ids}")
        print(f"  sampling_qlogps={[float(x) for x in trace['draft_logprobs'].tolist()]}")
        return result

    original = augmentor._sample_from_draft
    augmentor._sample_from_draft = MethodType(traced_sample_from_draft, augmentor)
    trace["restore"] = lambda: setattr(augmentor, "_sample_from_draft", original)
    return trace


def _print_batch(batch: dict[str, Any], *, limit: int) -> None:
    print("TRACE batch")
    for key, value in batch.items():
        if hasattr(value, "shape"):
            print(f"  {key}: {_shape(value)}")
    if "loss_mask" in batch:
        loss_mask = batch["loss_mask"]
        valid = loss_mask[0].bool().nonzero(as_tuple=False).flatten()
        print(f"  loss_mask_sum={_fmt(_scalar(loss_mask.sum()))}")
        print(f"  first_loss_positions={valid[:limit].detach().cpu().tolist()}")
    if "input_ids" in batch:
        print(f"  input_ids_head={batch['input_ids'][0, :limit].detach().cpu().tolist()}")


def _batch_indices_for_loader(train_loader: Any, batch_index: int) -> list[int] | None:
    batch_sampler = getattr(train_loader, "batch_sampler", None)
    if batch_sampler is None:
        return None
    try:
        batches = list(batch_sampler)
    except Exception:  # noqa: BLE001
        return None
    if batch_index < 0 or batch_index >= len(batches):
        return None
    batch = batches[batch_index]
    if hasattr(batch, "tolist"):
        return [int(x) for x in batch.tolist()]
    return [int(x) for x in batch]


def _clone_trace_item(item: dict[str, Any], source: Any = None) -> dict[str, Any]:
    cloned = {
        key: value.detach().cpu().clone() if hasattr(value, "detach") else value
        for key, value in item.items()
    }
    if source is not None:
        cloned["_hidden_state_source"] = dict(source)
    return cloned


def _captured_item_for_doc(
    *,
    train_loader: Any,
    captured_items: list[dict[str, Any]],
    batch_index: int,
    doc_id: int,
) -> dict[str, Any] | None:
    batch_sampler = getattr(train_loader, "batch_sampler", None)
    if batch_sampler is None:
        return None
    try:
        batches = list(batch_sampler)
    except Exception:  # noqa: BLE001
        return None
    if batch_index < 0 or batch_index >= len(batches):
        return None
    offset = sum(len(batch) for batch in batches[:batch_index])
    capture_index = offset + doc_id
    if capture_index < 0 or capture_index >= len(captured_items):
        return None
    return captured_items[capture_index]


def _print_packed_raw_alignment(
    *,
    train_loader: Any,
    captured_items: list[dict[str, Any]],
    batch: dict[str, Any],
    batch_index: int,
    anchor_pos: int,
) -> None:
    print("TRACE packed_raw_alignment")
    indices = _batch_indices_for_loader(train_loader, batch_index)
    if indices is None:
        print("  skipped=unavailable_batch_sampler_indices")
        return
    if "document_ids" not in batch:
        print("  skipped=no_document_ids")
        return

    document_ids = batch["document_ids"]
    doc_id = int(document_ids[0, anchor_pos].detach().cpu().item())
    if doc_id < 0 or doc_id >= len(indices):
        print(f"  skipped=doc_id_out_of_range doc_id={doc_id} indices={indices}")
        return

    same_doc = document_ids[0, : anchor_pos + 1] == document_ids[0, anchor_pos]
    doc_positions = same_doc.nonzero(as_tuple=False).flatten()
    if doc_positions.numel() == 0:
        print("  skipped=no_doc_positions")
        return

    doc_start = int(doc_positions[0].detach().cpu().item())
    local_pos = anchor_pos - doc_start
    dataset_index = indices[doc_id]
    raw_item = _captured_item_for_doc(
        train_loader=train_loader,
        captured_items=captured_items,
        batch_index=batch_index,
        doc_id=doc_id,
    )
    if raw_item is None:
        print(f"  skipped=no_captured_raw_item dataset_index={dataset_index}")
        return

    raw_len = int(raw_item["input_ids"].shape[0])
    if local_pos < 0 or local_pos >= raw_len:
        print(
            "  skipped=local_pos_out_of_range "
            f"dataset_index={dataset_index} raw_len={raw_len} local_pos={local_pos}"
        )
        return

    packed_token = int(batch["input_ids"][0, anchor_pos].detach().cpu().item())
    raw_token = int(raw_item["input_ids"][local_pos].detach().cpu().item())
    packed_hidden = batch["hidden_states"][0, anchor_pos].detach().float().cpu()
    raw_hidden = raw_item["hidden_states"][local_pos].detach().float().cpu()
    packed_last = (
        batch["verifier_last_hidden_states"][0, anchor_pos].detach().float().cpu()
    )
    raw_last = raw_item["verifier_last_hidden_states"][local_pos].detach().float().cpu()
    hidden_abs = (packed_hidden - raw_hidden).abs()
    last_abs = (packed_last - raw_last).abs()

    print(
        "  "
        f"batch_index={batch_index} doc_id={doc_id} dataset_index={dataset_index} "
        f"anchor_pos={anchor_pos} doc_start={doc_start} local_pos={local_pos}"
    )
    print(f"  hidden_state_source={raw_item.get('_hidden_state_source')}")
    print(
        "  "
        f"token_match={packed_token == raw_token} "
        f"packed_token={packed_token} raw_token={raw_token} raw_len={raw_len}"
    )
    print("  raw_source=captured_collate_preprocess_same_loader_pass")
    print(f"  hidden_max_abs_diff={_fmt(float(hidden_abs.max().item()))}")
    print(f"  verifier_last_max_abs_diff={_fmt(float(last_abs.max().item()))}")


def _print_raw_doc_regen_compare(
    *,
    augmentor: Any,
    raw_item: dict[str, Any] | None,
    local_pos: int,
    label: str,
) -> None:
    print(f"TRACE {label}.raw_doc_regen_compare")
    if raw_item is None:
        print("  skipped=no_raw_item")
        return
    try:
        import time
        from pathlib import Path

        from safetensors.torch import load_file
        from speculators.data_generation.vllm_client import wait_for_lock
        from speculators.data_generation.vllm_client import generate_hidden_states

        token_ids = raw_item["input_ids"].detach().cpu().tolist()
        path = generate_hidden_states(
            augmentor.client,
            augmentor.model_id,
            {"input_ids": token_ids},
            timeout=augmentor.config.request_timeout,
        )
        path_obj = Path(path)
        lock_path = Path(path + ".lock")
        deadline = time.monotonic() + augmentor.config.hidden_states_file_timeout
        while lock_path.exists() or not path_obj.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for hidden states file: {path}")
            if lock_path.exists():
                wait_for_lock(str(lock_path), timeout=max(deadline - time.monotonic(), 0.1))
                continue
            time.sleep(0.05)
        loaded = load_file(path)
        fresh = loaded["hidden_states"]
        token_match = loaded["token_ids"].detach().cpu().tolist() == token_ids
        cached = raw_item["verifier_last_hidden_states"][local_pos].detach().float().cpu()
        fresh_h = fresh[local_pos, -1].detach().float().cpu()
        print(f"  generated_path={path}")
        print(f"  token_ids_match={token_match}")
        print(f"  fresh_hidden_shape={tuple(fresh.shape)}")
        print(f"  local_pos={local_pos}")
        print(f"  verifier_last_max_abs_diff={_fmt(float((cached - fresh_h).abs().max().item()))}")
    except Exception as exc:  # noqa: BLE001
        print(f"  error={type(exc).__name__}: {exc}")


def _print_added_batch_keys(batch: dict[str, Any], before_keys: set[str], *, limit: int) -> None:
    added = sorted(set(batch) - before_keys)
    print(f"TRACE augmentor.added_keys={added}")
    for key in added:
        value = batch[key]
        if hasattr(value, "shape"):
            _print_tensor(key, value, limit=limit)
        else:
            print(f"  {key}: {value}")
    if "sampled_draft_ids" in batch and "sampled_target_ids" in batch:
        print("TRACE augmentor.sampled_id_mapping")
        draft_ids = batch["sampled_draft_ids"][0].detach().cpu().tolist()
        target_ids = batch["sampled_target_ids"][0].detach().cpu().tolist()
        print(f"  draft_ids={draft_ids}")
        print(f"  target_ids={target_ids}")


def _document_prefix_bounds(batch: dict[str, Any], anchor_pos: int) -> tuple[int, int]:
    document_ids = batch.get("document_ids")
    if document_ids is None:
        return 0, anchor_pos + 1
    anchor_doc = document_ids[0, anchor_pos]
    same_doc = document_ids[0, : anchor_pos + 1] == anchor_doc
    positions = same_doc.nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        return 0, anchor_pos + 1
    return int(positions[0].item()), anchor_pos + 1


def _score_with_vllm(
    *,
    augmentor: Any,
    prefix_token_ids: list[int],
    token_ids: list[int],
    label: str,
) -> dict[str, Any] | None:
    import speculators.train.sampled_acceptance as sampled_acceptance_mod

    try:
        scored = sampled_acceptance_mod.score_sampled_tokens(
            client=augmentor.client,
            model=augmentor.model_id,
            prefix_token_ids=prefix_token_ids,
            sampled_token_ids=token_ids,
            prompt_logprobs=augmentor.config.prompt_logprobs,
            timeout=augmentor.config.request_timeout,
            cleanup_hidden_states=True,
            hidden_states_file_timeout=augmentor.config.hidden_states_file_timeout,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"TRACE {label}.vllm_error={type(exc).__name__}: {exc}")
        return None
    out = [float(x) for x in scored["token_logprobs"]]
    print(f"TRACE {label}.vllm")
    print(f"  prefix_len={len(prefix_token_ids)}")
    print(f"  token_ids={token_ids}")
    print(f"  score_positions={scored['score_positions']}")
    print(f"  logprobs={out}")
    return scored | {"token_logprobs_float": out}


def _score_with_vllm_hidden_states(
    *,
    augmentor: Any,
    prefix_token_ids: list[int],
    token_ids: list[int],
    label: str,
) -> dict[str, Any] | None:
    import speculators.train.sampled_acceptance as sampled_acceptance_mod

    try:
        scored = sampled_acceptance_mod.score_sampled_tokens(
            client=augmentor.client,
            model=augmentor.model_id,
            prefix_token_ids=prefix_token_ids,
            sampled_token_ids=token_ids,
            prompt_logprobs=augmentor.config.prompt_logprobs,
            timeout=augmentor.config.request_timeout,
            cleanup_hidden_states=False,
            hidden_states_file_timeout=augmentor.config.hidden_states_file_timeout,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"TRACE {label}.vllm_error={type(exc).__name__}: {exc}")
        return None
    out = [float(x) for x in scored["token_logprobs"]]
    print(f"TRACE {label}.vllm_hidden_request")
    print(f"  prefix_len={len(prefix_token_ids)}")
    print(f"  token_ids={token_ids}")
    print(f"  score_positions={scored['score_positions']}")
    print(f"  logprobs={out}")
    print(f"  hidden_states_path={scored['hidden_states_path']}")
    return scored | {"token_logprobs_float": out}


def _target_to_draft_gather_ids(model: Any, target_ids: Any, device: Any) -> tuple[Any, Any]:
    import torch

    if model.t2d is None:
        return target_ids, torch.ones_like(target_ids, dtype=torch.bool)

    in_draft_vocab = model.t2d[target_ids.to(model.t2d.device)].to(device)
    target_to_draft_index = (
        torch.cumsum(model.t2d.to(device=device, dtype=torch.long), dim=0) - 1
    )
    return target_to_draft_index[target_ids], in_draft_vocab.bool()


def _local_verifier_gt_logprobs(
    *,
    model: Any,
    batch: dict[str, Any],
    token_positions: list[int],
    shift: int = -1,
) -> list[float]:
    import torch

    device = batch["input_ids"].device
    # For normal causal hidden states, shift=-1 means logits[:, pos-1] predict
    # input_ids[:, pos]. Other shifts are printed as a debugging sweep.
    logit_positions = torch.tensor(
        [pos + shift for pos in token_positions],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        verifier_logits = model.verifier_lm_head(
            model.verifier_norm(batch["verifier_last_hidden_states"])
        )
        target_ids = batch["input_ids"][0, token_positions].to(
            device=device, dtype=torch.long
        )
        gather_ids, in_draft_vocab = _target_to_draft_gather_ids(
            model, target_ids, device
        )
        if not bool(in_draft_vocab.all().item()):
            missing = target_ids[~in_draft_vocab].detach().cpu().tolist()
            print(f"TRACE gt_verifier_compare.local_missing_from_draft_vocab={missing}")
        logprobs = torch.log_softmax(
            verifier_logits[0, logit_positions].float(),
            dim=-1,
        )
        safe_gather_ids = gather_ids.clamp_min(0)
        gathered = logprobs.gather(-1, safe_gather_ids.unsqueeze(-1)).squeeze(-1)
        gathered = torch.where(
            in_draft_vocab.bool(),
            gathered,
            torch.full_like(gathered, float("nan")),
        )
    return [float(x) for x in gathered.detach().cpu().tolist()]


def _project_hidden_logprobs(
    *,
    model: Any,
    hidden_states: Any,
    token_ids: list[int],
    score_positions: list[int],
    shift: int = -1,
) -> list[float] | None:
    import torch

    device = next(model.parameters()).device
    hidden_states = hidden_states.to(device=device)
    if hidden_states.ndim == 3:
        hidden_states = hidden_states[:, -1]
    target_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    gather_ids, in_draft_vocab = _target_to_draft_gather_ids(model, target_ids, device)
    if not bool(in_draft_vocab.all().item()):
        missing = target_ids[~in_draft_vocab].detach().cpu().tolist()
        print(f"TRACE vllm_hidden_projection.missing_from_draft_vocab={missing}")
    positions = torch.tensor(
        [pos + shift for pos in score_positions],
        dtype=torch.long,
        device=device,
    )
    if bool(((positions < 0) | (positions >= hidden_states.shape[0])).any().item()):
        return None
    with torch.no_grad():
        verifier_logits = model.verifier_lm_head(
            model.verifier_norm(hidden_states.unsqueeze(0))
        )
        logprobs = torch.log_softmax(verifier_logits[0, positions].float(), dim=-1)
        gathered = logprobs.gather(
            -1,
            gather_ids.clamp_min(0).unsqueeze(-1),
        ).squeeze(-1)
        gathered = torch.where(
            in_draft_vocab.bool(),
            gathered,
            torch.full_like(gathered, float("nan")),
        )
    return [float(x) for x in gathered.detach().cpu().tolist()]


def _delete_trace_hidden_states(path_value: str | None) -> None:
    if path_value is None:
        return
    from pathlib import Path

    path = Path(path_value)
    path.unlink(missing_ok=True)
    Path(str(path) + ".lock").unlink(missing_ok=True)


def _compare_vllm_hidden_projection(
    *,
    model: Any,
    scored: dict[str, Any] | None,
    token_ids: list[int],
    label: str,
    cleanup: bool = True,
) -> list[float] | None:
    if scored is None:
        return None
    hidden_states_path = scored.get("hidden_states_path")
    if hidden_states_path is None:
        print(f"TRACE {label}.hidden_projection skipped: no hidden_states_path")
        return None
    try:
        from safetensors.torch import load_file

        loaded = load_file(hidden_states_path)
        hidden_states = loaded["hidden_states"]
        loaded_token_ids = loaded["token_ids"].detach().cpu().tolist()
        expected_token_ids = scored["prompt_token_ids"]
        token_ids_match = loaded_token_ids == expected_token_ids
        vals = _project_hidden_logprobs(
            model=model,
            hidden_states=hidden_states,
            token_ids=token_ids,
            score_positions=scored["score_positions"],
            shift=-1,
        )
        print(f"TRACE {label}.hidden_projection")
        print(f"  hidden_states_shape={tuple(hidden_states.shape)}")
        print("  shift=-1")
        print(f"  token_ids_match={token_ids_match}")
        print(f"  logprobs={vals}")
        return vals
    except Exception as exc:  # noqa: BLE001
        print(f"TRACE {label}.hidden_projection_error={type(exc).__name__}: {exc}")
        return None
    finally:
        if cleanup:
            _delete_trace_hidden_states(hidden_states_path)


def _cleanup_hidden_alignment(alignment: dict[str, Any] | None) -> None:
    if alignment is None:
        return
    _delete_trace_hidden_states(alignment.get("path"))


_HF_VERIFIER_CACHE: dict[tuple[str, str, str, bool], Any] = {}


def _load_hf_verifier(
    *,
    args: argparse.Namespace,
    device: Any,
    dtype: Any,
) -> Any:
    cache_key = (
        args.verifier_name_or_path,
        str(device),
        str(dtype),
        bool(args.trust_remote_code),
    )
    if cache_key not in _HF_VERIFIER_CACHE:
        from transformers import AutoModelForCausalLM

        print("TRACE hf_direct.load")
        print(f"  model_path={args.verifier_name_or_path}")
        print(f"  device={device}")
        print(f"  dtype={dtype}")
        verifier = AutoModelForCausalLM.from_pretrained(
            args.verifier_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        verifier.to(device)
        verifier.eval()
        _HF_VERIFIER_CACHE[cache_key] = verifier
    return _HF_VERIFIER_CACHE[cache_key]


def _run_hf_hidden_for_prompt(
    *,
    args: argparse.Namespace,
    prompt_token_ids: list[int],
    device: Any,
    dtype: Any,
) -> Any:
    import torch

    verifier = _load_hf_verifier(args=args, device=device, dtype=dtype)
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        return verifier(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )


def _score_hf_logprobs_with_dspark_head(
    *,
    model: Any,
    hf_hidden_states: Any,
    token_ids: list[int],
    score_positions: list[int],
) -> list[float] | None:
    import torch

    device = next(model.parameters()).device
    final_hidden = hf_hidden_states[-1][0].to(device=device)
    target_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    positions = torch.tensor(
        [pos - 1 for pos in score_positions],
        dtype=torch.long,
        device=device,
    )
    if bool(((positions < 0) | (positions >= final_hidden.shape[0])).any().item()):
        return None
    gather_ids, in_draft_vocab = _target_to_draft_gather_ids(model, target_ids, device)
    with torch.no_grad():
        logits = model.verifier_lm_head(model.verifier_norm(final_hidden.unsqueeze(0)))
        logprobs = torch.log_softmax(logits[0, positions].float(), dim=-1)
        gathered = logprobs.gather(
            -1,
            gather_ids.clamp_min(0).unsqueeze(-1),
        ).squeeze(-1)
        gathered = torch.where(
            in_draft_vocab.bool(),
            gathered,
            torch.full_like(gathered, float("nan")),
        )
    return [float(x) for x in gathered.detach().cpu().tolist()]


def _score_hf_output_logprobs(
    *,
    hf_logits: Any,
    token_ids: list[int],
    score_positions: list[int],
) -> list[float] | None:
    import torch

    logits = hf_logits[0].detach().float()
    positions = torch.tensor(
        [pos - 1 for pos in score_positions],
        dtype=torch.long,
        device=logits.device,
    )
    if bool(((positions < 0) | (positions >= logits.shape[0])).any().item()):
        return None
    target_ids = torch.tensor(token_ids, dtype=torch.long, device=logits.device)
    with torch.no_grad():
        logprobs = torch.log_softmax(logits[positions], dim=-1)
        gathered = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return [float(x) for x in gathered.detach().cpu().tolist()]


def _compare_vllm_hf_hidden_alignment(
    *,
    args: argparse.Namespace,
    model: Any,
    batch: dict[str, Any],
    scored: dict[str, Any] | None,
    prefix_token_ids: list[int],
    gt_ids: list[int],
    gt_positions: list[int],
    label: str,
) -> dict[str, Any] | None:
    print(f"TRACE {label}.hf_vllm_hidden_alignment")
    if not args.hf_hidden_compare:
        print("  skipped=disabled")
        return None
    if scored is None or scored.get("hidden_states_path") is None:
        print("  skipped=no_vllm_hidden_states_path")
        return None

    try:
        from safetensors.torch import load_file

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        prompt = [*prefix_token_ids, *gt_ids]
        score_positions = scored["score_positions"]
        hf_outputs = _run_hf_hidden_for_prompt(
            args=args,
            prompt_token_ids=prompt,
            device=device,
            dtype=dtype,
        )
        loaded = load_file(scored["hidden_states_path"])
        vllm_hidden = loaded["hidden_states"]
        vllm_token_ids = loaded["token_ids"].detach().cpu().tolist()
        expected_layers = list(model.target_layer_ids)
        final_layer_id = int(model.config.transformer_layer_config.num_hidden_layers)
        if final_layer_id not in expected_layers:
            expected_layers.append(final_layer_id)

        print(f"  prompt_len={len(prompt)}")
        print(f"  prefix_len={len(prefix_token_ids)}")
        print(f"  gt_ids={gt_ids}")
        print(f"  score_positions={score_positions}")
        print(f"  hidden_positions={[pos - 1 for pos in score_positions]}")
        print(f"  vllm_token_ids_match_prompt={vllm_token_ids == prompt}")
        print(
            "  scored_prompt_ids_match_prompt="
            f"{scored['prompt_token_ids'] == prompt}"
        )
        print(f"  vllm_hidden_shape={tuple(vllm_hidden.shape)}")
        print(f"  hf_logits_shape={tuple(hf_outputs.logits.shape)}")
        print(f"  hf_hidden_count={len(hf_outputs.hidden_states)}")
        print(
            "  hf_hidden_shapes="
            f"{[tuple(h.shape) for h in hf_outputs.hidden_states]}"
        )
        print(f"  expected_layer_ids={expected_layers}")

        shape_ok = (
            vllm_hidden.ndim == 3
            and int(vllm_hidden.shape[0]) == len(prompt)
            and int(vllm_hidden.shape[1]) == len(expected_layers)
            and int(vllm_hidden.shape[2])
            == int(hf_outputs.hidden_states[-1].shape[-1])
        )
        print(f"  vllm_shape_matches_expected={shape_ok}")

        hf_logprobs = _score_hf_logprobs_with_dspark_head(
            model=model,
            hf_hidden_states=hf_outputs.hidden_states,
            token_ids=gt_ids,
            score_positions=score_positions,
        )
        hf_output_logprobs = _score_hf_output_logprobs(
            hf_logits=hf_outputs.logits,
            token_ids=gt_ids,
            score_positions=score_positions,
        )
        print(f"  hf_output_logits_logprobs={hf_output_logprobs}")
        print(f"  hf_with_dspark_norm_logprobs={hf_logprobs}")
        print(f"  vllm_prompt_logprobs={scored['token_logprobs']}")
        if hf_output_logprobs is not None:
            diffs = [
                abs(float(a) - float(b))
                for a, b in zip(
                    hf_output_logprobs,
                    scored["token_logprobs"],
                    strict=True,
                )
            ]
            print(f"  hf_logits_vs_vllm_prompt_logprob_max_abs_diff={_fmt(max(diffs))}")
        if hf_logprobs is not None:
            diffs = [
                abs(float(a) - float(b))
                for a, b in zip(hf_logprobs, scored["token_logprobs"], strict=True)
            ]
            print(
                "  hf_dspark_norm_vs_vllm_prompt_logprob_max_abs_diff="
                f"{_fmt(max(diffs))}"
            )

        print("  layer,slot,gt_index,packed_pos,prompt_hidden_pos,cached_diff,hf_diff")
        for slot, layer_id in enumerate(expected_layers):
            if layer_id >= len(hf_outputs.hidden_states):
                print(f"  {layer_id},{slot},<skip_layer_out_of_range>")
                continue
            hf_layer = hf_outputs.hidden_states[layer_id][0].detach().float().cpu()
            for gt_index, packed_pos in enumerate(gt_positions, start=1):
                prompt_hidden_pos = score_positions[gt_index - 1] - 1
                if prompt_hidden_pos < 0 or prompt_hidden_pos >= vllm_hidden.shape[0]:
                    print(
                        f"  {layer_id},{slot},{gt_index},{packed_pos},"
                        f"{prompt_hidden_pos},<invalid>,<invalid>"
                    )
                    continue
                vllm_vec = vllm_hidden[
                    prompt_hidden_pos,
                    slot,
                ].detach().float().cpu()
                hf_vec = hf_layer[prompt_hidden_pos]
                hf_diff = float((vllm_vec - hf_vec).abs().max().item())
                cached_diff_value = "<na>"
                cached_pos = packed_pos - 1
                if layer_id == expected_layers[-1]:
                    cached = batch["verifier_last_hidden_states"][
                        0,
                        cached_pos,
                    ].detach().float().cpu()
                    cached_diff_value = _fmt(
                        float((vllm_vec - cached).abs().max().item())
                    )
                else:
                    hidden_slots = (
                        batch["hidden_states"].shape[-1] // vllm_hidden.shape[-1]
                    )
                    if slot < hidden_slots:
                        start = slot * vllm_hidden.shape[-1]
                        end = start + vllm_hidden.shape[-1]
                        cached = batch["hidden_states"][
                            0,
                            cached_pos,
                            start:end,
                        ].detach().float().cpu()
                        cached_diff_value = _fmt(
                            float((vllm_vec - cached).abs().max().item())
                        )
                print(
                    f"  {layer_id},{slot},{gt_index},{packed_pos},"
                    f"{prompt_hidden_pos},{cached_diff_value},{_fmt(hf_diff)}"
                )

        return {
            "path": scored["hidden_states_path"],
            "prompt": prompt,
            "hidden": vllm_hidden,
            "score_positions": list(score_positions),
            "expected_layers": expected_layers,
            "hf_outputs": hf_outputs,
        }
    except Exception as exc:  # noqa: BLE001
        print(f"  error={type(exc).__name__}: {exc}")
        return None


def _repeat_vllm_hidden_alignment(
    *,
    args: argparse.Namespace,
    augmentor: Any,
    first: dict[str, Any] | None,
    prefix_token_ids: list[int],
    gt_ids: list[int],
    label: str,
) -> None:
    print(f"TRACE {label}.vllm_hidden_repeat")
    if first is None:
        print("  skipped=no_first_hidden")
        return
    repeat = max(int(args.vllm_hidden_repeat), 0)
    if repeat <= 0:
        print("  skipped=repeat_0")
        _cleanup_hidden_alignment(first)
        return

    hidden_runs = [first["hidden"]]
    path_runs = [first["path"]]
    for run_idx in range(1, repeat + 1):
        scored = _score_with_vllm_hidden_states(
            augmentor=augmentor,
            prefix_token_ids=prefix_token_ids,
            token_ids=gt_ids,
            label=f"{label}_repeat{run_idx}",
        )
        if scored is None or scored.get("hidden_states_path") is None:
            print(f"  repeat={run_idx} skipped=no_hidden_path")
            continue
        try:
            from safetensors.torch import load_file

            loaded = load_file(scored["hidden_states_path"])
            hidden_runs.append(loaded["hidden_states"])
            path_runs.append(scored["hidden_states_path"])
            print(
                f"  repeat={run_idx} token_ids_match="
                f"{loaded['token_ids'].detach().cpu().tolist() == first['prompt']} "
                f"hidden_shape={tuple(loaded['hidden_states'].shape)}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  repeat={run_idx} error={type(exc).__name__}: {exc}")

    print("  compare_to_run0,layer,slot,prompt_hidden_pos,max_abs_diff,mean_abs_diff")
    for run_idx, hidden in enumerate(hidden_runs[1:], start=1):
        if tuple(hidden.shape) != tuple(hidden_runs[0].shape):
            print(
                f"  {run_idx},<shape_mismatch>,"
                f"{tuple(hidden_runs[0].shape)}!={tuple(hidden.shape)}"
            )
            continue
        for slot, layer_id in enumerate(first["expected_layers"]):
            for score_pos in first["score_positions"]:
                hidden_pos = score_pos - 1
                diff = (
                    hidden_runs[0][hidden_pos, slot].detach().float().cpu()
                    - hidden[hidden_pos, slot].detach().float().cpu()
                ).abs()
                print(
                    f"  {run_idx},{layer_id},{slot},{hidden_pos},"
                    f"{_fmt(float(diff.max().item()))},"
                    f"{_fmt(float(diff.mean().item()))}"
                )

    for path in path_runs:
        _delete_trace_hidden_states(path)


def _compare_cached_and_fresh_doc_hidden(
    *,
    batch: dict[str, Any],
    scored: dict[str, Any] | None,
    doc_start: int,
    gt_positions: list[int],
    label: str,
) -> None:
    if scored is None or scored.get("hidden_states_path") is None:
        print(f"TRACE {label}.cached_vs_fresh_hidden skipped")
        return
    try:
        from safetensors.torch import load_file

        loaded = load_file(scored["hidden_states_path"])
        fresh = loaded["hidden_states"]
        prompt_ids = scored["prompt_token_ids"]
        token_match = loaded["token_ids"].detach().cpu().tolist() == prompt_ids
        max_diffs = []
        for packed_pos in gt_positions:
            cached_pos = packed_pos - 1
            fresh_pos = packed_pos - doc_start - 1
            cached_h = batch["verifier_last_hidden_states"][0, cached_pos].detach().float().cpu()
            fresh_h = fresh[fresh_pos, -1].detach().float().cpu()
            max_diffs.append(float((cached_h - fresh_h).abs().max().item()))
        print(f"TRACE {label}.cached_vs_fresh_hidden")
        print(f"  token_ids_match={token_match}")
        print(f"  fresh_hidden_shape={tuple(fresh.shape)}")
        print(f"  max_abs_diffs={[_fmt(x) for x in max_diffs]}")
    except Exception as exc:  # noqa: BLE001
        print(f"TRACE {label}.cached_vs_fresh_hidden error={type(exc).__name__}: {exc}")


def _print_local_verifier_slot_diagnostics(
    *,
    model: Any,
    batch: dict[str, Any],
    anchor_pos: int,
    gt_positions: list[int],
    gt_ids: list[int],
    vllm_logprobs: list[float] | None,
) -> None:
    import torch

    device = batch["input_ids"].device
    with torch.no_grad():
        verifier_logits = model.verifier_lm_head(
            model.verifier_norm(batch["verifier_last_hidden_states"])
        ).float()
        logits = verifier_logits[0, anchor_pos : anchor_pos + len(gt_positions)]
        diff_from_slot0 = (logits - logits[:1]).abs().amax(dim=-1)

    print("TRACE gt_verifier_compare.local_slot_diffs")
    print("  slot,logit_pos,max_abs_diff_from_slot1")
    for i, diff in enumerate(diff_from_slot0.detach().cpu().tolist()):
        print(f"  {i + 1},{anchor_pos + i},{_fmt(float(diff))}")

    print("TRACE gt_verifier_compare.local_shift_sweep")
    print("  shift,mean_abs_diff_to_full_vllm,logprobs")
    for shift in (-2, -1, 0, 1):
        valid = all(0 <= pos + shift < batch["input_ids"].shape[1] for pos in gt_positions)
        if not valid:
            print(f"  {shift},<invalid>,[]")
            continue
        vals = _local_verifier_gt_logprobs(
            model=model,
            batch=batch,
            token_positions=gt_positions,
            shift=shift,
        )
        if vllm_logprobs is None:
            mean_diff = "<no-vllm>"
        else:
            finite = [
                abs(a - b)
                for a, b in zip(vals, vllm_logprobs, strict=True)
                if not math.isnan(a)
            ]
            mean_diff = _fmt(sum(finite) / len(finite)) if finite else "nan"
        print(f"  {shift},{mean_diff},{[_fmt(x) for x in vals]}")


def _print_vocab_mapping_invariants(model: Any, sampled_draft_ids: list[int]) -> None:
    import torch

    if model.d2t is None or model.t2d is None:
        print("TRACE vocab_mapping_invariants skipped: full verifier vocab")
        return

    d2t = model.d2t.detach().cpu().long()
    t2d = model.t2d.detach().cpu().bool()
    draft_ids = torch.arange(d2t.numel(), dtype=torch.long)
    mapped_target_ids = draft_ids + d2t
    selected_target_ids = torch.nonzero(t2d, as_tuple=False).flatten().long()
    unique_count = int(torch.unique(mapped_target_ids).numel())
    in_range = bool(
        (mapped_target_ids.min() >= 0)
        and (mapped_target_ids.max() < t2d.numel())
    )
    selected_equal = bool(torch.equal(mapped_target_ids, selected_target_ids))

    print("TRACE vocab_mapping_invariants")
    print(f"  draft_vocab_size={d2t.numel()}")
    print(f"  target_vocab_size={t2d.numel()}")
    print(f"  mapped_min={int(mapped_target_ids.min().item())}")
    print(f"  mapped_max={int(mapped_target_ids.max().item())}")
    print(f"  mapped_in_range={in_range}")
    print(f"  unique_mapped_count={unique_count}")
    print(f"  unique_matches_draft_vocab={unique_count == d2t.numel()}")
    print(f"  t2d_selected_count={int(t2d.sum(dtype=torch.long).item())}")
    print(f"  mapped_equals_t2d_selected_ordinals={selected_equal}")
    print("  sampled_draft_id,target_id,offset,t2d_contains,target_rank")
    target_to_rank = torch.full((t2d.numel(),), -1, dtype=torch.long)
    target_to_rank[selected_target_ids] = torch.arange(selected_target_ids.numel())
    for draft_id in sampled_draft_ids:
        target_id = int(mapped_target_ids[draft_id].item())
        print(
            f"  {draft_id},"
            f"{target_id},"
            f"{int(d2t[draft_id].item())},"
            f"{bool(t2d[target_id].item())},"
            f"{int(target_to_rank[target_id].item())}"
        )


def _print_tokenizer_mapping_check(
    *,
    args: argparse.Namespace,
    model: Any,
    sampled_draft_ids: list[int],
    sampled_target_ids: list[int],
) -> None:
    try:
        from speculators.data_generation.preprocessing import get_tokenizer, load_processor

        processor = load_processor(
            args.verifier_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
        target_tokenizer = get_tokenizer(processor)
    except Exception as exc:  # noqa: BLE001
        print(f"TRACE tokenizer_mapping_check skipped: {type(exc).__name__}: {exc}")
        return

    # DSpark currently samples from a pruned vocabulary whose ids are indices
    # into d2t, not ids for an independent tokenizer. Decode the mapped target ids
    # and re-encode the cumulative target text to catch non-1:1 assumptions.
    print("TRACE tokenizer_mapping_check")
    for i, (draft_id, target_id) in enumerate(
        zip(sampled_draft_ids, sampled_target_ids, strict=True),
        start=1,
    ):
        target_piece = target_tokenizer.decode(
            [target_id],
            skip_special_tokens=False,
        )
        reencoded = target_tokenizer.encode(target_piece, add_special_tokens=False)
        print(
            f"  pos={i} draft_id={draft_id} target_id={target_id} "
            f"target_piece={target_piece!r} reencoded_piece={reencoded}"
        )

    target_text = target_tokenizer.decode(
        sampled_target_ids,
        skip_special_tokens=False,
    )
    target_reencoded = target_tokenizer.encode(target_text, add_special_tokens=False)
    print(f"  mapped_target_text={target_text!r}")
    print(f"  mapped_target_ids={sampled_target_ids}")
    print(f"  reencoded_target_ids={target_reencoded}")

    if model.d2t is not None:
        print("  note=draft ids are pruned-vocab ids; d2t maps them into target ids")


def _compare_gt_verifier_paths(
    *,
    args: argparse.Namespace,
    model: Any,
    batch: dict[str, Any],
    augmentor: Any,
) -> None:
    if "sampled_anchor_pos" not in batch:
        return

    anchor_pos = int(batch["sampled_anchor_pos"][0].detach().cpu().item())
    K = min(args.gt_compare_len, int(batch["input_ids"].shape[1]) - anchor_pos - 1)
    if K <= 0:
        print("TRACE gt_verifier_compare skipped: no continuation after anchor")
        return

    full_prefix = batch["input_ids"][0, : anchor_pos + 1].detach().cpu().tolist()
    doc_start, doc_end = _document_prefix_bounds(batch, anchor_pos)
    doc_prefix = batch["input_ids"][0, doc_start:doc_end].detach().cpu().tolist()
    gt_positions = list(range(anchor_pos + 1, anchor_pos + 1 + K))
    gt_ids = batch["input_ids"][0, gt_positions].detach().cpu().tolist()
    local_logprobs = _local_verifier_gt_logprobs(
        model=model,
        batch=batch,
        token_positions=gt_positions,
    )
    doc_id = int(batch["document_ids"][0, anchor_pos].detach().cpu().item())
    block_document_ids = batch["document_ids"][
        0, anchor_pos : anchor_pos + K + 1
    ].detach().cpu().tolist()
    position_ids = batch.get("position_ids")
    doc_pos_start = None
    doc_pos_anchor = None
    if position_ids is not None:
        doc_pos_start = int(position_ids[0, doc_start].detach().cpu().item())
        doc_pos_anchor = int(position_ids[0, anchor_pos].detach().cpu().item())

    print("TRACE gt_verifier_compare.setup")
    print(f"  anchor_pos={anchor_pos}")
    print(f"  anchor_doc_id={doc_id}")
    print(f"  full_prefix_len={len(full_prefix)}")
    print(f"  doc_start={doc_start}")
    print(f"  doc_prefix_len={len(doc_prefix)}")
    print(f"  packed_prefix_covered={doc_start > 0}")
    print(f"  block_document_ids={block_document_ids}")
    print(f"  position_id_doc_start={doc_pos_start}")
    print(f"  position_id_anchor={doc_pos_anchor}")
    print(f"  gt_positions={gt_positions}")
    print(f"  gt_ids={gt_ids}")
    print(f"  local_pruned_verifier_logprobs={local_logprobs}")

    full_vllm = _score_with_vllm(
        augmentor=augmentor,
        prefix_token_ids=full_prefix,
        token_ids=gt_ids,
        label="gt_full_prefix",
    )
    full_vllm_logprobs = None if full_vllm is None else full_vllm["token_logprobs_float"]
    _print_local_verifier_slot_diagnostics(
        model=model,
        batch=batch,
        anchor_pos=anchor_pos,
        gt_positions=gt_positions,
        gt_ids=gt_ids,
        vllm_logprobs=full_vllm_logprobs,
    )
    doc_vllm = _score_with_vllm_hidden_states(
        augmentor=augmentor,
        prefix_token_ids=doc_prefix,
        token_ids=gt_ids,
        label="gt_doc_prefix",
    )
    doc_vllm_logprobs = None if doc_vllm is None else doc_vllm["token_logprobs_float"]
    _compare_cached_and_fresh_doc_hidden(
        batch=batch,
        scored=doc_vllm,
        doc_start=doc_start,
        gt_positions=gt_positions,
        label="gt_doc_prefix",
    )
    hf_vllm_hidden = _compare_vllm_hf_hidden_alignment(
        args=args,
        model=model,
        batch=batch,
        scored=doc_vllm,
        prefix_token_ids=doc_prefix,
        gt_ids=gt_ids,
        gt_positions=gt_positions,
        label="gt_doc_prefix",
    )
    doc_hidden_logprobs = _compare_vllm_hidden_projection(
        model=model,
        scored=doc_vllm,
        token_ids=gt_ids,
        label="gt_doc_prefix",
        cleanup=False,
    )
    _repeat_vllm_hidden_alignment(
        args=args,
        augmentor=augmentor,
        first=hf_vllm_hidden,
        prefix_token_ids=doc_prefix,
        gt_ids=gt_ids,
        label="gt_doc_prefix",
    )
    if hf_vllm_hidden is None and doc_vllm is not None:
        _delete_trace_hidden_states(doc_vllm.get("hidden_states_path"))

    print("TRACE gt_verifier_compare.summary")
    if doc_start == 0:
        print("  packed_prefix_status=not_covered_doc_start_is_0")
    elif full_vllm_logprobs is not None and doc_vllm_logprobs is not None:
        full_doc_max_diff = max(
            abs(a - b) for a, b in zip(full_vllm_logprobs, doc_vllm_logprobs, strict=True)
        )
        print(f"  packed_prefix_status=covered_doc_start_gt_0")
        print(f"  full_vs_doc_max_abs_diff={_fmt(full_doc_max_diff)}")
    print(
        "  pos,gt_id,local_pruned,full_prefix_vllm,diff_full,"
        "doc_prefix_vllm,diff_doc,doc_hidden_projection,diff_hidden"
    )
    for i, local in enumerate(local_logprobs):
        full = None if full_vllm_logprobs is None else full_vllm_logprobs[i]
        doc = None if doc_vllm_logprobs is None else doc_vllm_logprobs[i]
        doc_hidden = None if doc_hidden_logprobs is None else doc_hidden_logprobs[i]
        print(
            f"  {i + 1},"
            f"{gt_ids[i]},"
            f"{_fmt(local)},"
            f"{'<err>' if full is None else _fmt(full)},"
            f"{'<err>' if full is None else _fmt(full - local)},"
            f"{'<err>' if doc is None else _fmt(doc)},"
            f"{'<err>' if doc is None else _fmt(doc - local)},"
            f"{'<err>' if doc_hidden is None else _fmt(doc_hidden)},"
            f"{'<err>' if doc_hidden is None else _fmt(doc_hidden - local)}"
        )


def _compare_sampled_verifier_paths(
    *,
    batch: dict[str, Any],
    augmentor: Any,
) -> None:
    if "sampled_anchor_pos" not in batch or "sampled_target_ids" not in batch:
        return

    anchor_pos = int(batch["sampled_anchor_pos"][0].detach().cpu().item())
    sampled_ids = batch["sampled_target_ids"][0].detach().cpu().tolist()
    actual_logprobs = batch["sampled_target_logprobs"][0].detach().cpu().tolist()
    full_prefix = batch["input_ids"][0, : anchor_pos + 1].detach().cpu().tolist()
    doc_start, doc_end = _document_prefix_bounds(batch, anchor_pos)
    doc_prefix = batch["input_ids"][0, doc_start:doc_end].detach().cpu().tolist()

    print("TRACE sampled_verifier_prefix_compare.setup")
    print(f"  anchor_pos={anchor_pos}")
    print(f"  full_prefix_len={len(full_prefix)}")
    print(f"  doc_start={doc_start}")
    print(f"  doc_prefix_len={len(doc_prefix)}")
    print(f"  packed_prefix_covered={doc_start > 0}")
    print(f"  sampled_target_ids={sampled_ids}")
    print(f"  actual_sampled_target_logprobs={actual_logprobs}")

    full_vllm = _score_with_vllm(
        augmentor=augmentor,
        prefix_token_ids=full_prefix,
        token_ids=sampled_ids,
        label="sampled_full_prefix",
    )
    doc_vllm = _score_with_vllm(
        augmentor=augmentor,
        prefix_token_ids=doc_prefix,
        token_ids=sampled_ids,
        label="sampled_doc_prefix",
    )
    full_vllm_logprobs = None if full_vllm is None else full_vllm["token_logprobs_float"]
    doc_vllm_logprobs = None if doc_vllm is None else doc_vllm["token_logprobs_float"]

    print("TRACE sampled_verifier_prefix_compare.summary")
    if doc_start == 0:
        print("  packed_prefix_status=not_covered_doc_start_is_0")
    elif full_vllm_logprobs is not None and doc_vllm_logprobs is not None:
        full_doc_max_diff = max(
            abs(a - b) for a, b in zip(full_vllm_logprobs, doc_vllm_logprobs, strict=True)
        )
        print("  packed_prefix_status=covered_doc_start_gt_0")
        print(f"  full_vs_doc_max_abs_diff={_fmt(full_doc_max_diff)}")
    print("  pos,target_id,actual,full_prefix_vllm,diff_full,doc_prefix_vllm,diff_doc")
    for i, actual in enumerate(actual_logprobs):
        full = None if full_vllm_logprobs is None else full_vllm_logprobs[i]
        doc = None if doc_vllm_logprobs is None else doc_vllm_logprobs[i]
        print(
            f"  {i + 1},"
            f"{sampled_ids[i]},"
            f"{_fmt(float(actual))},"
            f"{'<err>' if full is None else _fmt(full)},"
            f"{'<err>' if full is None else _fmt(full - float(actual))},"
            f"{'<err>' if doc is None else _fmt(doc)},"
            f"{'<err>' if doc is None else _fmt(doc - float(actual))}"
        )


def _print_metrics(raw: dict[str, float], normalized: dict[str, float], block_size: int) -> None:
    print("TRACE metrics.raw")
    for key in sorted(raw):
        if key.startswith("sampled_"):
            print(f"  {key}={_fmt(raw[key])}")

    print("TRACE metrics.normalized_core")
    for key in sorted(k for k in normalized if not k.startswith("sampled_")):
        print(f"  train/{key}={_fmt(normalized[key])}")

    print("TRACE metrics.normalized_sampled")
    for key in sorted(k for k in normalized if k.startswith("sampled_")):
        print(f"  train/{key}={_fmt(normalized[key])}")

    print("TRACE metrics.alpha_self_check")
    print("  pos,qlogp,plogp,logged_alpha,alpha_ref,diff,logged_survival,survival_ref,undercovered")
    log_alpha_sum = 0.0
    for pos in range(1, block_size):
        qk = f"sampled_pos{pos}_qlogp"
        pk = f"sampled_pos{pos}_plogp"
        ak = f"sampled_pos{pos}_alpha"
        sk = f"sampled_pos{pos}_survival"
        uk = f"sampled_pos{pos}_undercovered"
        if qk not in normalized:
            continue
        q = normalized[qk]
        p = normalized[pk]
        logged_alpha = normalized[ak]
        log_alpha = min(0.0, p - q)
        log_alpha_sum += log_alpha
        alpha_ref = math.exp(log_alpha)
        survival_ref = math.exp(log_alpha_sum)
        print(
            f"  {pos},"
            f"{_fmt(q)},"
            f"{_fmt(p)},"
            f"{_fmt(logged_alpha)},"
            f"{_fmt(alpha_ref)},"
            f"{_fmt(abs(logged_alpha - alpha_ref))},"
            f"{_fmt(normalized[sk])},"
            f"{_fmt(survival_ref)},"
            f"{_fmt(normalized[uk])}"
        )


def _compare_sampling_and_replay(sample_trace: dict[str, Any], model_trace: dict[str, Any]) -> None:
    if "draft_logprobs" not in sample_trace or not model_trace["recompute_calls"]:
        print("TRACE sampling_vs_training_replay unavailable")
        return
    sample_q = sample_trace["draft_logprobs"].float().view(-1)
    replay = model_trace["recompute_calls"][-1]
    replay_q = replay["q_logp"].float().view(-1)
    K = min(sample_q.numel(), replay_q.numel())
    print("TRACE sampling_vs_training_replay")
    print("  pos,sampling_qlogp,replay_qlogp,diff")
    max_diff = 0.0
    for i in range(K):
        diff = abs(float(sample_q[i].item()) - float(replay_q[i].item()))
        max_diff = max(max_diff, diff)
        print(
            f"  {i + 1},"
            f"{_fmt(float(sample_q[i].item()))},"
            f"{_fmt(float(replay_q[i].item()))},"
            f"{_fmt(diff)}"
        )
    print(f"  max_abs_diff={_fmt(max_diff)}")


def trace_real_step(args: argparse.Namespace) -> None:
    import torch

    import scripts.train as train_script
    from speculators.model import SpeculatorModel
    from speculators.models.dspark.core import DSparkDraftModel
    from speculators.train.dataloader import create_train_val_loaders
    from speculators.train.sampled_acceptance import (
        SampledAcceptanceAugmentor,
        SampledAcceptanceConfig,
    )
    from speculators.train.utils import normalize_counted_metrics

    torch.manual_seed(args.seed)
    train_args = _make_train_args(args)
    hidden_states_dtype = getattr(torch, args.hidden_states_dtype)
    device = _resolve_device(torch, args.device)

    print("TRACE setup")
    print(f"  repo={ROOT}")
    print(f"  seed={args.seed}")
    print(f"  device={device}")
    print(f"  hidden_states_dtype={hidden_states_dtype}")
    print(f"  vllm_endpoint={args.vllm_endpoint}")
    print(f"  verifier_name_or_path={args.verifier_name_or_path}")
    print(f"  from_pretrained={args.from_pretrained or '<random/init from args>'}")
    print(f"  draft_config={args.draft_config or '<none>'}")

    registry = SpeculatorModel.registry
    if registry is None:
        raise RuntimeError("SpeculatorModel registry is empty")
    model_class = registry["dspark"]
    d2t, t2d, draft_vocab_size = train_script.parse_vocab_mappings(train_args)
    model = train_script.build_draft_model(
        train_args, model_class, t2d, d2t, draft_vocab_size
    )
    if not isinstance(model, DSparkDraftModel):
        raise TypeError(f"Expected DSparkDraftModel, got {type(model).__name__}")
    _force_npu_supported_attention(model, device)
    model.to(device=device, dtype=hidden_states_dtype)  # type: ignore[call-arg]
    model.train()

    print("TRACE model")
    print(f"  class={type(model).__name__}")
    print(f"  block_size={int(model.block_size)}")
    print(f"  config.max_anchors={int(model.config.max_anchors)}")
    print(f"  draft_vocab_size={draft_vocab_size}")
    print(f"  target_layer_ids={list(model.target_layer_ids)}")
    print(f"  markov_head={None if model.markov_head is None else type(model.markov_head).__name__}")
    print(f"  confidence_head={None if model.confidence_head is None else type(model.confidence_head).__name__}")
    print(f"  d2t_present={model.d2t is not None}")
    print(f"  t2d_present={model.t2d is not None}")

    hidden_size = model.config.transformer_layer_config.hidden_size
    captured_raw_items: list[dict[str, Any]] = []

    def capture_raw_item(item: dict[str, Any]) -> dict[str, Any]:
        source = item.pop("_hidden_state_source", None)
        captured_raw_items.append(_clone_trace_item(item, source))
        return item

    train_loader, _ = create_train_val_loaders(
        data_path=args.data_path,
        total_seq_len=args.total_seq_len,
        hidden_states_dtype=hidden_states_dtype,
        noise_std=args.noise_std,
        legacy_data=args.legacy_data,
        hidden_states_path=args.hidden_states_path,
        vllm_endpoint=args.vllm_endpoint,
        on_missing=args.on_missing,
        on_generate=args.on_generate,
        verifier_name_or_path=args.verifier_name_or_path,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        hidden_size=hidden_size,
        num_target_layers=len(model.target_layer_ids),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        preprocess=capture_raw_item,
    )
    train_call_kwargs, _ = model_class.get_trainer_kwargs(**vars(train_args))
    print("TRACE train_call_kwargs")
    for key, value in train_call_kwargs.items():
        print(f"  {key}={value}")

    augmentor = SampledAcceptanceAugmentor(
        SampledAcceptanceConfig(
            vllm_endpoint=args.vllm_endpoint,
            model=args.verifier_name_or_path,
            request_timeout=args.request_timeout,
            temperature=args.temperature,
        )
    )
    print("TRACE augmentor")
    print(f"  model_id={augmentor.model_id}")
    print(f"  temperature={augmentor.config.temperature}")
    print(f"  sampling_max_anchors={augmentor.config.max_anchors}")
    print(f"  prompt_logprobs={augmentor.config.prompt_logprobs}")

    score_calls, original_score = _install_score_trace()
    original_select_anchors = _install_anchor_selection_trace(args.anchor_selection)
    model_trace = _install_model_traces(model, topk=args.topk)
    sample_trace = _install_sampling_trace(augmentor, topk=args.topk)

    try:
        batch = None
        for idx, candidate in enumerate(train_loader):
            if idx == args.batch_index:
                batch = candidate
                break
        if batch is None:
            raise RuntimeError(f"Could not read batch index {args.batch_index}")

        gpu_batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        _ACTIVE_TRACE_BATCH.clear()
        _ACTIVE_TRACE_BATCH.update(gpu_batch)
        _print_batch(gpu_batch, limit=args.print_anchor_limit)

        before_keys = set(gpu_batch)
        gpu_batch = augmentor(model, gpu_batch)
        _print_added_batch_keys(gpu_batch, before_keys, limit=args.print_anchor_limit)

        if "sampled_draft_ids" not in gpu_batch:
            print("TRACE stop: augmentor skipped sampled acceptance for this batch")
            return

        if model.d2t is not None:
            sampled_draft_for_map = gpu_batch["sampled_draft_ids"].to(model.d2t.device)
            mapped = (
                sampled_draft_for_map + model.d2t[sampled_draft_for_map]
            ).to(
                gpu_batch["sampled_target_ids"].device
            )
            equal = bool(torch.equal(mapped, gpu_batch["sampled_target_ids"]))
            print("TRACE d2t_check")
            print(f"  mapped_target_ids={mapped.detach().cpu().flatten().tolist()}")
            print(f"  equals_sampled_target_ids={equal}")

        sampled_draft_id_list = (
            gpu_batch["sampled_draft_ids"][0].detach().cpu().tolist()
        )
        sampled_target_id_list = (
            gpu_batch["sampled_target_ids"][0].detach().cpu().tolist()
        )
        _print_vocab_mapping_invariants(
            model=model,
            sampled_draft_ids=sampled_draft_id_list,
        )
        _print_tokenizer_mapping_check(
            args=args,
            model=model,
            sampled_draft_ids=sampled_draft_id_list,
            sampled_target_ids=sampled_target_id_list,
        )
        anchor_pos = int(gpu_batch["sampled_anchor_pos"][0].detach().cpu().item())
        _print_packed_raw_alignment(
            train_loader=train_loader,
            captured_items=captured_raw_items,
            batch=gpu_batch,
            batch_index=args.batch_index,
            anchor_pos=anchor_pos,
        )
        batch_indices = _batch_indices_for_loader(train_loader, args.batch_index)
        raw_item_for_anchor = None
        local_pos_for_anchor = 0
        if batch_indices is not None and "document_ids" in gpu_batch:
            doc_id = int(gpu_batch["document_ids"][0, anchor_pos].detach().cpu().item())
            raw_item_for_anchor = _captured_item_for_doc(
                train_loader=train_loader,
                captured_items=captured_raw_items,
                batch_index=args.batch_index,
                doc_id=doc_id,
            )
            doc_start, _ = _document_prefix_bounds(gpu_batch, anchor_pos)
            local_pos_for_anchor = anchor_pos - doc_start
        _print_raw_doc_regen_compare(
            augmentor=augmentor,
            raw_item=raw_item_for_anchor,
            local_pos=local_pos_for_anchor,
            label="gt_doc_prefix",
        )
        _compare_gt_verifier_paths(
            args=args,
            model=model,
            batch=gpu_batch,
            augmentor=augmentor,
        )
        _compare_sampled_verifier_paths(
            batch=gpu_batch,
            augmentor=augmentor,
        )

        draft_tokens, loss, metrics = model(**gpu_batch, **train_call_kwargs)
        print("TRACE forward.done")
        print(f"  loss_for_backward={_fmt(_scalar(loss))}")
        print(f"  draft_tokens_shape={tuple(draft_tokens.shape)}")
        print(f"  score_calls={len(score_calls)}")

        raw = {k: _scalar(v) for k, v in metrics.items()}
        normalized = normalize_counted_metrics(dict(raw), world_size=1)
        _print_metrics(raw, normalized, int(args.block_size))
        if "loss" in normalized:
            sampled_loss = normalized.get("sampled_acceptance_loss", 0.0)
            total_logged = normalized.get("total_loss")
            print("TRACE loss_decomposition")
            print(f"  base_train_loss={_fmt(normalized['loss'])}")
            print(f"  sampled_acceptance_loss={_fmt(sampled_loss)}")
            print(f"  sampled_alpha={_fmt(args.sampled_acceptance_loss_alpha)}")
            print(
                "  base_plus_weighted_sampled="
                f"{_fmt(normalized['loss'] + args.sampled_acceptance_loss_alpha * sampled_loss)}"
            )
            print(f"  loss_for_backward={_fmt(_scalar(loss))}")
            if total_logged is not None:
                print(f"  train/total_loss={_fmt(total_logged)}")
        _compare_sampling_and_replay(sample_trace, model_trace)

        if not args.skip_backward:
            print("TRACE backward_step.begin")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            print("TRACE backward_step.done")
            print(f"  grad_norm_before_clip={_fmt(_scalar(grad_norm))}")
            print(f"  lr={_fmt(args.lr)}")
        else:
            print("TRACE backward_step skipped")
    finally:
        sample_trace["restore"]()
        model_trace["restore"]()
        _restore_score_trace(original_score)
        _restore_anchor_selection(original_select_anchors)
        _ACTIVE_TRACE_BATCH.clear()


def _trace_section(text: str, name: str) -> str:
    marker = f"TRACE {name}"
    start = text.find(marker)
    if start < 0:
        return ""
    next_start = text.find("\nTRACE ", start + len(marker))
    return text[start:] if next_start < 0 else text[start:next_start]


def _summarize_trace(text: str, *, tol: float = 1e-5) -> int:
    failures: list[str] = []

    packed = _trace_section(text, "packed_raw_alignment")
    if not packed:
        failures.append("packed_raw_alignment missing")
    elif (
        "token_match=True" not in packed
        or "hidden_max_abs_diff=0.000000" not in packed
        or "verifier_last_max_abs_diff=0.000000" not in packed
    ):
        failures.append(packed.strip())

    sampled = _trace_section(text, "sampled_verifier_prefix_compare.summary")
    if not sampled:
        failures.append("sampled verifier prefix summary missing")
    else:
        bad_sampled = []
        for line in sampled.splitlines():
            parts = line.strip().split(",")
            if len(parts) == 7 and parts[0].isdigit() and abs(float(parts[-1])) > tol:
                bad_sampled.append(line.strip())
        if bad_sampled:
            failures.append("sampled doc-prefix mismatch:\n" + "\n".join(bad_sampled))

    gt = _trace_section(text, "gt_verifier_compare.summary")
    bad_gt = []
    for line in gt.splitlines():
        parts = line.strip().split(",")
        if len(parts) == 9 and parts[0].isdigit():
            diff_hidden = abs(float(parts[-1]))
            if diff_hidden > tol:
                bad_gt.append(line.strip())
    if bad_gt:
        if packed:
            for line in packed.splitlines():
                if "hidden_state_source=" in line:
                    failures.append(line.strip())
                    break
        direct_hidden = _trace_section(text, "gt_doc_prefix.cached_vs_fresh_hidden")
        if direct_hidden:
            failures.append(direct_hidden.strip())
        raw_regen = _trace_section(text, "gt_doc_prefix.raw_doc_regen_compare")
        if raw_regen:
            failures.append(raw_regen.strip())
        failures.append("gt cached-hidden mismatch:\n" + "\n".join(bad_gt))

    if failures:
        print("TRACE CHECK FAIL")
        for failure in failures:
            print(failure)
            print()
        return 1

    print("TRACE CHECK OK")
    return 0


def main() -> None:
    args = _parser().parse_args()
    if args.verbose:
        trace_real_step(args)
        return

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        trace_real_step(args)
    raise SystemExit(_summarize_trace(buf.getvalue()))


if __name__ == "__main__":
    main()
