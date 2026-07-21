#!/usr/bin/env python3
"""Compare DSpark training replay vs evaluation single-anchor replay."""

from __future__ import annotations

import argparse
import importlib.util
import logging
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from replay_dspark_anchor_step import (
    choose_anchor,
    dtype_of,
    load_vllm_sample,
    load_vocab_maps,
    map_draft_to_target,
    sim,
    text,
)

log = logging.getLogger("compare_dspark_train_infer_anchor")


def load_eval_impl(torch):
    path = Path(__file__).parents[2] / "scripts" / "evaluate" / "dspark_offline_eval.py"
    spec = importlib.util.spec_from_file_location("dspark_offline_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.torch = torch
    return module


@contextmanager
def fixed_anchor_select(torch, anchor: int):
    import speculators.models.dflash.core as dflash_core

    old_select_anchors = dflash_core.select_anchors

    def select_anchors(loss_mask, num_anchors, block_size):
        del block_size
        anchors = torch.zeros(num_anchors, dtype=torch.long, device=loss_mask.device)
        valid = torch.zeros(num_anchors, dtype=torch.bool, device=loss_mask.device)
        anchors[0] = int(anchor)
        valid[0] = True
        return anchors, valid

    dflash_core.select_anchors = select_anchors
    try:
        yield
    finally:
        dflash_core.select_anchors = old_select_anchors


def draft_target_ids(torch, draft):
    draft_ids = torch.arange(draft.draft_vocab_size, device=draft.lm_head.weight.device)
    if draft.use_draft_vocab and draft.d2t is not None:
        return draft_ids + draft.d2t.to(draft_ids.device).long()
    return draft_ids


def target_full_to_draft(torch, draft, full_logits):
    return full_logits.index_select(dim=-1, index=draft_target_ids(torch, draft))


def probs(torch, logits):
    return torch.softmax(logits.float(), dim=-1)


def soft_overlap(torch, draft_logits, target_logits):
    return float(torch.minimum(probs(torch, draft_logits), probs(torch, target_logits)).sum().item())


def slot_summary(torch, tokenizer, draft, name, backbone_hidden, base, bias, final, target):
    draft_id = int(torch.argmax(final).item())
    target_draft_id = int(torch.argmax(target).item())
    draft_target_id = map_draft_to_target(torch, draft, draft_id)
    target_target_id = map_draft_to_target(torch, draft, target_draft_id)
    return SimpleNamespace(
        name=name,
        backbone_hidden=backbone_hidden,
        base_logits=base,
        markov_bias=bias,
        final_logits=final,
        target_logits=target,
        draft_draft_id=draft_id,
        draft_top1=draft_target_id,
        draft_text=text(tokenizer, draft_target_id),
        target_draft_id=target_draft_id,
        target_top1=target_target_id,
        target_text=text(tokenizer, target_target_id),
        greedy_match=draft_target_id == target_target_id,
        soft_overlap=soft_overlap(torch, final, target),
    )


def train_replay(torch, draft, tokenizer, sample, anchor):
    block = int(draft.block_size)
    input_ids = sample.token_ids.to(draft.lm_head.weight.device).unsqueeze(0)
    features = sample.train_hidden.to(draft.lm_head.weight.device).unsqueeze(0)
    verifier_last = sample.train_last.to(draft.lm_head.weight.device).unsqueeze(0)
    loss_mask = sample.loss_mask.to(draft.lm_head.weight.device).unsqueeze(0)
    document_ids = torch.zeros_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)

    with fixed_anchor_select(torch, anchor):
        hidden, base_logits, targets, aligned_mask, anchored_idx = draft._backbone_forward(
            features, input_ids, loss_mask, verifier_last, document_ids, position_ids
        )

    n_blocks = int(draft.config.max_anchors)
    block_tokens = input_ids[0, anchored_idx].view(n_blocks, block)
    prev_ids = draft._build_markov_prev_token_ids(
        block_tokens, getattr(draft.config, "sample_from_anchor", False)
    )
    hidden_blocks = hidden.view(n_blocks, block, -1)
    base = base_logits.view(n_blocks, block, -1)
    if draft.markov_head is None:
        bias = torch.zeros_like(base)
    else:
        bias = draft.markov_head.block_bias(prev_token_ids=prev_ids, hidden_states=hidden_blocks)

    slot = 1
    final = base[0, slot] + bias[0, slot]
    target = targets.view(n_blocks, block, -1)[0, slot]
    return slot_summary(
        torch,
        tokenizer,
        draft,
        "train",
        hidden_blocks[0, slot],
        base[0, slot],
        bias[0, slot],
        final,
        target,
    ), SimpleNamespace(
        features=features,
        verifier_last=verifier_last,
        position_ids=position_ids,
        anchored_idx=anchored_idx,
        aligned_mask=aligned_mask.view(n_blocks, block)[0, slot],
        prev_id=int(prev_ids[0, slot].item()),
    )


def infer_replay(torch, eval_impl, runner, draft, tokenizer, input_ids, hidden_prefix, target_logits):
    slot = 1
    hidden, base_logits = runner._single_anchor_backbone(
        hidden_prefix,
        input_ids,
        input_ids.shape[1] - 1,
    )
    anchor_id = input_ids[:, -1:].long()
    if draft.markov_head is None:
        bias = torch.zeros_like(base_logits[:, slot : slot + 1, :])
    else:
        bias = draft.markov_head.block_bias(
            prev_token_ids=anchor_id,
            hidden_states=hidden[:, slot : slot + 1, :],
        )
    base = base_logits[0, slot]
    bias = bias[0, 0]
    final = base + bias
    del eval_impl
    return slot_summary(
        torch,
        tokenizer,
        draft,
        "infer",
        hidden[0, slot],
        base,
        bias,
        final,
        target_logits,
    )


def print_row(quantity, train, infer, diff):
    log.info("%-26s %-24s %-24s %s", quantity, str(train), str(infer), str(diff))


def position_sig(values):
    if not values:
        return ()
    return (values[0], values[-1], len(values))


def print_compare(torch, title, train, infer, train_meta, infer_meta):
    log.info("")
    log.info("=== %s ===", title)
    log.info("%-26s %-24s %-24s %s", "quantity", "train", "inference", "diff")
    print_row("anchor_token", train_meta.anchor_id, infer_meta.anchor_id, "OK" if train_meta.anchor_id == infer_meta.anchor_id else "DIFF")
    print_row(
        "position_ids(first,last,n)",
        position_sig(train_meta.position_ids),
        position_sig(infer_meta.position_ids),
        "OK" if train_meta.position_ids == infer_meta.position_ids else "DIFF",
    )
    print_row("anchored_index0", train_meta.anchored_index0, infer_meta.anchored_index0, "OK" if train_meta.anchored_index0 == infer_meta.anchored_index0 else "DIFF")
    for name, a, b in (
        ("aux_hidden", train_meta.aux_hidden, infer_meta.aux_hidden),
        ("verifier_last_hidden", train_meta.verifier_last, infer_meta.verifier_last),
        ("backbone_hidden", train.backbone_hidden, infer.backbone_hidden),
        ("base_logits", train.base_logits, infer.base_logits),
        ("markov_bias", train.markov_bias, infer.markov_bias),
        ("final_logits", train.final_logits, infer.final_logits),
        ("target_logits", train.target_logits, infer.target_logits),
    ):
        cos, max_abs, mean_abs = sim(torch, a, b)
        print_row(name, "-", "-", f"cos={cos:.8f} max={max_abs:.8g} mean={mean_abs:.8g}")
    print_row("base_top1", int(torch.argmax(train.base_logits).item()), int(torch.argmax(infer.base_logits).item()), "OK" if int(torch.argmax(train.base_logits).item()) == int(torch.argmax(infer.base_logits).item()) else "DIFF")
    print_row("markov_top1_bias", int(torch.argmax(train.markov_bias).item()), int(torch.argmax(infer.markov_bias).item()), "OK" if int(torch.argmax(train.markov_bias).item()) == int(torch.argmax(infer.markov_bias).item()) else "DIFF")
    print_row("final_draft_top1", train.draft_top1, infer.draft_top1, "OK" if train.draft_top1 == infer.draft_top1 else "DIFF")
    print_row("target_top1", train.target_top1, infer.target_top1, "OK" if train.target_top1 == infer.target_top1 else "DIFF")
    print_row("greedy_accept", train.greedy_match, infer.greedy_match, "OK" if train.greedy_match == infer.greedy_match else "DIFF")
    print_row("soft_overlap", f"{train.soft_overlap:.8f}", f"{infer.soft_overlap:.8f}", f"{abs(train.soft_overlap - infer.soft_overlap):.8g}")
    log.info("train draft=%s %s | target=%s %s", train.draft_top1, train.draft_text, train.target_top1, train.target_text)
    log.info("infer draft=%s %s | target=%s %s", infer.draft_top1, infer.draft_text, infer.target_top1, infer.target_text)


def capture_final_norm_input(target_model, box):
    norm = getattr(getattr(target_model, "model", None), "norm", None)
    if norm is None:
        raise ValueError("target model has no model.norm module to hook")

    def hook(_module, inputs):
        box["final_norm_input"] = inputs[0].detach()

    return norm.register_forward_pre_hook(hook)


def print_norm_status(torch, draft, target_vocab_logits, sample, norm_input, norm_output, anchor):
    vllm_last = sample.train_last.to(norm_input.device).unsqueeze(0)
    pre = norm_input[:, anchor, :]
    post = norm_output[:, anchor, :]
    last = vllm_last[:, anchor, :]
    pre_cos, pre_max, pre_mean = sim(torch, last, pre)
    post_cos, post_max, post_mean = sim(torch, last, post)
    log.info("")
    log.info("=== vLLM final hidden norm status ===")
    log.info(
        "vllm_last vs HF model.norm input : cos=%.8f max_abs=%.8g mean_abs=%.8g",
        pre_cos,
        pre_max,
        pre_mean,
    )
    log.info(
        "vllm_last vs HF model.norm output: cos=%.8f max_abs=%.8g mean_abs=%.8g",
        post_cos,
        post_max,
        post_mean,
    )
    vllm_projected = draft.verifier_lm_head(draft.verifier_norm(last))[0]
    hf_projected = target_full_to_draft(torch, draft, target_vocab_logits[:, anchor, :])[0]
    cos, max_abs, mean_abs = sim(torch, vllm_projected, hf_projected)
    log.info(
        "lm_head(verifier_norm(vllm_last)) vs HF logits: cos=%.8f max_abs=%.8g mean_abs=%.8g",
        cos,
        max_abs,
        mean_abs,
    )


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from speculators.models.dspark.core import DSparkDraftModel

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    eval_impl = load_eval_impl(torch)
    sample = load_vllm_sample(torch, args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.verifier_model, trust_remote_code=args.trust_remote_code
    )

    cfg = DSparkDraftModel.config_class.from_pretrained(args.draft_model)
    if args.sample_from_anchor:
        cfg.sample_from_anchor = True
    if getattr(cfg, "sample_from_anchor", False):
        raise ValueError("This debug script only supports slot1-as-first-draft mode.")
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

    anchor = choose_anchor(torch, sample.loss_mask, int(draft.block_size), args.anchor_position)
    token_ids = sample.token_ids.to(device)
    full_input_ids = token_ids.unsqueeze(0)
    prefix_input_ids = token_ids[: anchor + 1].unsqueeze(0)
    anchor_id = int(token_ids[anchor].item())
    log.info("sample_index=%d vllm_hidden_file=%s", args.sample_index, sample.hs_path)
    log.info("anchor=%d anchor_id=%d text=%s", anchor, anchor_id, text(tokenizer, anchor_id))

    train, train_raw = train_replay(torch, draft, tokenizer, sample, anchor)
    train_target_from_cache = draft.verifier_lm_head(
        draft.verifier_norm(train_raw.verifier_last[:, anchor, :])
    )[0]

    cached_infer_hidden = sample.train_hidden[:anchor].to(device).unsqueeze(0)
    infer_cached = infer_replay(
        torch,
        eval_impl,
        runner,
        draft,
        tokenizer,
        prefix_input_ids,
        cached_infer_hidden,
        train_target_from_cache,
    )
    common_train_meta = SimpleNamespace(
        anchor_id=anchor_id,
        position_ids=tuple(train_raw.position_ids[0, : anchor + 1].tolist()),
        anchored_index0=int(train_raw.anchored_idx[0].item()),
        aux_hidden=train_raw.features[:, :anchor, :],
        verifier_last=train_raw.verifier_last[:, anchor, :],
    )
    infer_cached_meta = SimpleNamespace(
        anchor_id=anchor_id,
        position_ids=tuple(range(anchor + 1)),
        anchored_index0=anchor,
        aux_hidden=cached_infer_hidden,
        verifier_last=train_raw.verifier_last[:, anchor, :],
    )
    print_compare(torch, "Test 1: same vLLM hidden, training block vs eval single-anchor", train, infer_cached, common_train_meta, infer_cached_meta)

    norm_box = {}
    norm_hook = capture_final_norm_input(target, norm_box)
    with torch.inference_mode():
        out = target(
            input_ids=prefix_input_ids,
            position_ids=torch.arange(anchor + 1, device=device).unsqueeze(0),
            use_cache=False,
            output_hidden_states=True,
        )
    norm_hook.remove()
    print_norm_status(
        torch,
        draft,
        out.logits,
        sample,
        norm_box["final_norm_input"],
        out.hidden_states[-1],
        anchor,
    )
    infer_hf_hidden_full = runner._extract_context_feature(out.hidden_states)
    infer_hf_hidden = infer_hf_hidden_full[:, :anchor, :].to(train_raw.features.dtype)
    infer_hf_target = target_full_to_draft(torch, draft, out.logits[:, -1, :])[0]
    infer_hf = infer_replay(
        torch,
        eval_impl,
        runner,
        draft,
        tokenizer,
        prefix_input_ids,
        infer_hf_hidden,
        infer_hf_target,
    )
    infer_hf_meta = SimpleNamespace(
        anchor_id=anchor_id,
        position_ids=tuple(range(anchor + 1)),
        anchored_index0=anchor,
        aux_hidden=infer_hf_hidden,
        verifier_last=out.hidden_states[-1][:, anchor, :].to(train_raw.verifier_last.dtype),
    )
    print_compare(torch, "Test 2: vLLM train hidden vs HF/eval extracted hidden", train, infer_hf, common_train_meta, infer_hf_meta)

    accepted = infer_hf.draft_top1 == infer_hf.target_top1
    expected = infer_hf.greedy_match
    log.info("")
    log.info("=== Test 3: greedy one-token verify ===")
    log.info("draft_top1=%d target_top1=%d accepted=%s expected=%s diff=%s", infer_hf.draft_top1, infer_hf.target_top1, accepted, expected, "OK" if accepted == expected else "DIFF")


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
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--anchor-position", type=int, required=True)
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
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
