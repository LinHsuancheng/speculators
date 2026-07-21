#!/usr/bin/env python3
"""Batch DSpark train-vs-infer anchor checks with online vLLM hidden states."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

from compare_dspark_train_infer_anchor import (
    capture_final_norm_input,
    infer_replay,
    load_eval_impl,
    target_full_to_draft,
    train_replay,
)
from replay_dspark_anchor_step import (
    dtype_of,
    load_vllm_sample,
    load_vocab_maps,
    sim,
)

log = logging.getLogger("batch_compare_dspark_train_infer")


def valid_anchors(torch, loss_mask, block_size: int):
    valid = loss_mask.bool().clone()
    valid[-block_size:] = False
    return torch.nonzero(valid, as_tuple=False).view(-1).tolist()


def pick_anchors(torch, loss_mask, block_size: int, count: int, explicit: str | None):
    anchors = valid_anchors(torch, loss_mask, block_size)
    if explicit:
        requested = [int(x) for x in explicit.split(",") if x.strip()]
        missing = [x for x in requested if x not in set(anchors)]
        if missing:
            raise ValueError(f"invalid requested anchors for this sample: {missing}")
        return requested
    if len(anchors) <= count:
        return anchors
    if count <= 1:
        return [anchors[0]]
    step = (len(anchors) - 1) / float(count - 1)
    return [anchors[round(i * step)] for i in range(count)]


def run_hf_prefix(torch, target, input_ids, anchor):
    box = {}
    hook = capture_final_norm_input(target, box)
    with torch.inference_mode():
        out = target(
            input_ids=input_ids[:, : anchor + 1],
            position_ids=torch.arange(anchor + 1, device=input_ids.device).unsqueeze(0),
            use_cache=False,
            output_hidden_states=True,
        )
    hook.remove()
    return out, box["final_norm_input"]


def compare_case(torch, eval_impl, runner, draft, tokenizer, target, sample, anchor):
    device = draft.lm_head.weight.device
    token_ids = sample.token_ids.to(device)
    input_ids = token_ids.unsqueeze(0)

    train, train_raw = train_replay(torch, draft, tokenizer, sample, anchor)
    target_from_vllm = draft.verifier_lm_head(
        draft.verifier_norm(train_raw.verifier_last[:, anchor, :])
    )[0]

    vllm_prefix = sample.train_hidden[:anchor].to(device).unsqueeze(0)
    infer_vllm = infer_replay(
        torch,
        eval_impl,
        runner,
        draft,
        tokenizer,
        input_ids[:, : anchor + 1],
        vllm_prefix,
        target_from_vllm,
    )

    out, norm_input = run_hf_prefix(torch, target, input_ids, anchor)
    hf_aux_full = runner._extract_context_feature(out.hidden_states)
    hf_aux = hf_aux_full[:, :anchor, :].to(train_raw.features.dtype)
    hf_target = target_full_to_draft(torch, draft, out.logits[:, -1, :])[0]
    infer_hf = infer_replay(
        torch,
        eval_impl,
        runner,
        draft,
        tokenizer,
        input_ids[:, : anchor + 1],
        hf_aux,
        hf_target,
    )

    vllm_last = sample.train_last.to(device).unsqueeze(0)[:, anchor, :]
    hf_norm_pre = norm_input[:, anchor, :].to(train_raw.verifier_last.dtype)
    hf_norm_post = out.hidden_states[-1][:, anchor, :].to(train_raw.verifier_last.dtype)
    aux_cos, aux_max, aux_mean = sim(torch, train_raw.features[:, :anchor, :], hf_aux)
    norm_pre_cos, norm_pre_max, norm_pre_mean = sim(torch, vllm_last, hf_norm_pre)
    norm_post_cos, _, _ = sim(torch, vllm_last, hf_norm_post)
    t1_final_cos, t1_final_max, _ = sim(torch, train.final_logits, infer_vllm.final_logits)
    t2_final_cos, t2_final_max, _ = sim(torch, train.final_logits, infer_hf.final_logits)
    t2_target_cos, t2_target_max, _ = sim(torch, train.target_logits, infer_hf.target_logits)

    return {
        "sample": None,
        "anchor": anchor,
        "anchor_id": int(token_ids[anchor].item()),
        "seq_len": int(token_ids.numel()),
        "t1_draft_ok": train.draft_top1 == infer_vllm.draft_top1,
        "t1_target_ok": train.target_top1 == infer_vllm.target_top1,
        "t1_greedy_ok": train.greedy_match == infer_vllm.greedy_match,
        "t1_final_cos": t1_final_cos,
        "t1_final_max": t1_final_max,
        "t2_draft_ok": train.draft_top1 == infer_hf.draft_top1,
        "t2_target_ok": train.target_top1 == infer_hf.target_top1,
        "t2_greedy_ok": train.greedy_match == infer_hf.greedy_match,
        "t2_final_cos": t2_final_cos,
        "t2_final_max": t2_final_max,
        "t2_target_cos": t2_target_cos,
        "t2_target_max": t2_target_max,
        "aux_cos": aux_cos,
        "aux_max": aux_max,
        "aux_mean": aux_mean,
        "norm_pre_cos": norm_pre_cos,
        "norm_pre_max": norm_pre_max,
        "norm_pre_mean": norm_pre_mean,
        "norm_post_cos": norm_post_cos,
        "train_draft": train.draft_top1,
        "infer_draft": infer_hf.draft_top1,
        "train_target": train.target_top1,
        "infer_target": infer_hf.target_top1,
        "train_greedy": train.greedy_match,
        "infer_greedy": infer_hf.greedy_match,
        "soft_diff": abs(train.soft_overlap - infer_hf.soft_overlap),
    }


def print_header():
    log.info(
        "%-6s %-7s %-7s %-3s %-3s %-3s %-10s %-10s %-10s %-10s %-10s %-10s %-12s %-12s %-12s %-9s",
        "sample",
        "anchor",
        "tok",
        "t1D",
        "t2D",
        "t2T",
        "aux_cos",
        "aux_max",
        "norm_pre",
        "norm_max",
        "final_cos",
        "target_cos",
        "draft",
        "target",
        "greedy",
        "softdiff",
    )


def print_result(row):
    log.info(
        "%-6d %-7d %-7d %-3s %-3s %-3s %-10.6f %-10.4g %-10.6f %-10.4g %-10.6f %-10.6f %-12s %-12s %-12s %-9.4g",
        row["sample"],
        row["anchor"],
        row["anchor_id"],
        "OK" if row["t1_draft_ok"] else "BAD",
        "OK" if row["t2_draft_ok"] else "BAD",
        "OK" if row["t2_target_ok"] else "BAD",
        row["aux_cos"],
        row["aux_max"],
        row["norm_pre_cos"],
        row["norm_pre_max"],
        row["t2_final_cos"],
        row["t2_target_cos"],
        f'{row["train_draft"]}->{row["infer_draft"]}',
        f'{row["train_target"]}->{row["infer_target"]}',
        f'{row["train_greedy"]}->{row["infer_greedy"]}',
        row["soft_diff"],
    )


def summarize(rows):
    total = len(rows)
    if total == 0:
        log.info("no cases ran")
        return
    counters = {
        "test1_draft_ok": sum(r["t1_draft_ok"] for r in rows),
        "test1_target_ok": sum(r["t1_target_ok"] for r in rows),
        "test1_greedy_ok": sum(r["t1_greedy_ok"] for r in rows),
        "test2_draft_ok": sum(r["t2_draft_ok"] for r in rows),
        "test2_target_ok": sum(r["t2_target_ok"] for r in rows),
        "test2_greedy_ok": sum(r["t2_greedy_ok"] for r in rows),
    }
    log.info("")
    log.info("=== Summary ===")
    log.info("cases=%d", total)
    for key, value in counters.items():
        log.info("%s=%d/%d %.2f%%", key, value, total, 100.0 * value / total)
    log.info(
        "min_aux_cos=%.6f max_aux_max=%.4g min_norm_pre_cos=%.6f max_soft_diff=%.6g",
        min(r["aux_cos"] for r in rows),
        max(r["aux_max"] for r in rows),
        min(r["norm_pre_cos"] for r in rows),
        max(r["soft_diff"] for r in rows),
    )


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from speculators.models.dspark.core import DSparkDraftModel

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    eval_impl = load_eval_impl(torch)
    tokenizer = AutoTokenizer.from_pretrained(
        args.verifier_model, trust_remote_code=args.trust_remote_code
    )

    cfg = DSparkDraftModel.config_class.from_pretrained(args.draft_model)
    if getattr(cfg, "sample_from_anchor", False) or args.sample_from_anchor:
        raise ValueError("batch debug only supports slot1-as-first-draft mode")
    if args.draft_attn_impl != "auto":
        cfg.transformer_layer_config._attn_implementation = args.draft_attn_impl
    d2t, t2d = load_vocab_maps(torch, args)
    draft = DSparkDraftModel.from_pretrained(
        args.draft_model, config=cfg, d2t=d2t, t2d=t2d
    ).to(device).eval()
    target = AutoModelForCausalLM.from_pretrained(
        args.verifier_model,
        torch_dtype=dtype_of(torch, args.dtype),
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()
    runner = eval_impl.DSparkOfflineRunner(
        target, draft, tokenizer, SimpleNamespace(temperature=0.0)
    )

    rows = []
    print_header()
    for sample_index in range(args.sample_start, args.sample_start + args.num_samples):
        args.sample_index = sample_index
        try:
            sample = load_vllm_sample(torch, args)
            anchors = pick_anchors(
                torch,
                sample.loss_mask,
                int(draft.block_size),
                args.anchors_per_sample,
                args.anchor_positions,
            )
            for anchor in anchors:
                row = compare_case(
                    torch, eval_impl, runner, draft, tokenizer, target, sample, anchor
                )
                row["sample"] = sample_index
                rows.append(row)
                print_result(row)
        except Exception:
            log.exception("failed sample_index=%d", sample_index)
            if args.stop_on_error:
                raise
    summarize(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--verifier-model", required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--vllm-endpoint", default="http://localhost:8000/v1")
    p.add_argument("--vllm-model", default=None)
    p.add_argument("--request-timeout", type=float, default=120)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--delete-vllm-hidden-state", action="store_true")
    p.add_argument("--sample-start", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--anchors-per-sample", type=int, default=3)
    p.add_argument("--anchor-positions", default=None)
    p.add_argument("--total-seq-len", type=int, default=3072)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--hidden-states-dtype", default="bfloat16")
    p.add_argument(
        "--draft-attn-impl",
        choices=["auto", "simple_flex_attention", "sdpa", "eager"],
        default="auto",
    )
    p.add_argument("--d2t-path", type=Path, default=None)
    p.add_argument("--t2d-path", type=Path, default=None)
    p.add_argument("--sample-from-anchor", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stop-on-error", action="store_true")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
