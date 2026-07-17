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
import math
from pathlib import Path
import sys
from types import MethodType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    parser.add_argument("--skip-backward", action="store_true")
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

        prefix_token_ids = input_ids[0, : anchor_pos + 1].tolist()
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
                "prefix_len": len(prefix_token_ids),
                "sampled_target_ids": list(sampled_target_ids),
                "sampled_draft_ids": list(sampled_draft_ids),
                "draft_logprobs": result["draft_logprobs"].detach().cpu(),
                "slot_trace": slot_trace,
            }
        )
        print("TRACE draft_sampling.end")
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
        preprocess=None,
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
        _print_batch(gpu_batch, limit=args.print_anchor_limit)

        before_keys = set(gpu_batch)
        gpu_batch = augmentor(model, gpu_batch)
        _print_added_batch_keys(gpu_batch, before_keys, limit=args.print_anchor_limit)

        if "sampled_draft_ids" not in gpu_batch:
            print("TRACE stop: augmentor skipped sampled acceptance for this batch")
            return

        if model.d2t is not None:
            mapped = model.d2t[gpu_batch["sampled_draft_ids"].to(model.d2t.device)].to(
                gpu_batch["sampled_target_ids"].device
            )
            equal = bool(torch.equal(mapped, gpu_batch["sampled_target_ids"]))
            print("TRACE d2t_check")
            print(f"  mapped_target_ids={mapped.detach().cpu().flatten().tolist()}")
            print(f"  equals_sampled_target_ids={equal}")

        draft_tokens, loss, metrics = model(**gpu_batch, **train_call_kwargs)
        print("TRACE forward.done")
        print(f"  loss_for_backward={_fmt(_scalar(loss))}")
        print(f"  draft_tokens_shape={tuple(draft_tokens.shape)}")
        print(f"  score_calls={len(score_calls)}")

        raw = {k: _scalar(v) for k, v in metrics.items()}
        normalized = normalize_counted_metrics(dict(raw), world_size=1)
        _print_metrics(raw, normalized, int(args.block_size))
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


def main() -> None:
    args = _parser().parse_args()
    trace_real_step(args)


if __name__ == "__main__":
    main()
