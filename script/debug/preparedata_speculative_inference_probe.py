#!/usr/bin/env python3
"""Run speculative inference from DATA_PATH preparedata prompts and report slot acc."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

from block_slot_acc_probe import SlotStats, generate_one, load_eval_impl
from replay_dspark_anchor_step import dtype_of, load_vocab_maps

log = logging.getLogger("preparedata_speculative_inference_probe")


def sample_indices(torch, dataset_len, start, count, randomize):
    available = list(range(start, dataset_len))
    if len(available) <= count:
        return available
    if not randomize:
        return available[:count]
    perm = torch.randperm(len(available))[:count].tolist()
    return [available[i] for i in perm]


def prompt_prefix_from_preparedata(torch, item, min_prompt_tokens):
    input_ids = torch.as_tensor(item["input_ids"], dtype=torch.long)
    loss_mask = torch.as_tensor(item["loss_mask"], dtype=torch.bool)
    valid = torch.nonzero(loss_mask, as_tuple=False).view(-1)
    if valid.numel() == 0:
        return None, "no_loss_tokens"
    first_loss = int(valid[0].item())
    if first_loss < min_prompt_tokens:
        return None, f"prompt_too_short:{first_loss}"
    return input_ids[:first_loss].unsqueeze(0), None


def run(args):
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from speculators.models.dspark.core import DSparkDraftModel

    torch.manual_seed(args.seed)
    eval_impl = load_eval_impl(torch)
    tokenizer = AutoTokenizer.from_pretrained(
        args.verifier_model,
        trust_remote_code=args.trust_remote_code,
    )
    cfg = DSparkDraftModel.config_class.from_pretrained(args.draft_model)
    if getattr(cfg, "sample_from_anchor", False):
        raise ValueError("script only supports slot1-as-first-draft mode")
    if args.draft_attn_impl != "auto":
        cfg.transformer_layer_config._attn_implementation = args.draft_attn_impl
    d2t, t2d = load_vocab_maps(torch, args)
    draft = DSparkDraftModel.from_pretrained(
        args.draft_model,
        config=cfg,
        d2t=d2t,
        t2d=t2d,
    ).to(args.device).eval()
    target = AutoModelForCausalLM.from_pretrained(
        args.verifier_model,
        torch_dtype=dtype_of(torch, args.dtype),
        trust_remote_code=args.trust_remote_code,
    ).to(args.device).eval()
    runner = eval_impl.DSparkOfflineRunner(
        target,
        draft,
        tokenizer,
        SimpleNamespace(temperature=0.0),
    )

    data = load_from_disk(args.data_path)
    indices = sample_indices(
        torch,
        len(data),
        args.sample_start,
        args.num_samples,
        args.random_samples,
    )
    stats = SlotStats("preparedata_prompt_speculative_inference_rounds", draft.block_size)
    skipped = 0
    for sample_index in indices:
        item = data[int(sample_index)]
        prompt_ids, reason = prompt_prefix_from_preparedata(
            torch,
            item,
            args.min_prompt_tokens,
        )
        if prompt_ids is None:
            skipped += 1
            log.info("sample=%d skipped=%s", sample_index, reason)
            continue
        prompt_ids = prompt_ids.to(args.device)
        output_ids, rounds, avg_accepted = generate_one(
            torch,
            eval_impl,
            runner,
            prompt_ids,
            args.max_new_tokens,
            stats,
        )
        log.info(
            "sample=%d prompt_tokens=%d output_tokens=%d rounds=%d avg_accepted=%.3f",
            sample_index,
            prompt_ids.shape[1],
            output_ids.shape[1] - prompt_ids.shape[1],
            rounds,
            avg_accepted,
        )

    log.info("skipped=%d/%d", skipped, len(indices))
    stats.print()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--verifier-model", required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--sample-start", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--random-samples", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--min-prompt-tokens", type=int, default=1)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument(
        "--draft-attn-impl",
        choices=["auto", "simple_flex_attention", "sdpa", "eager"],
        default="auto",
    )
    p.add_argument("--d2t-path", type=Path, default=None)
    p.add_argument("--t2d-path", type=Path, default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
