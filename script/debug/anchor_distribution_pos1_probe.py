#!/usr/bin/env python3
"""Probe token-uniform vs proposal-round weighted DSpark pos1 accuracy."""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

from compare_dspark_train_infer_anchor import (
    infer_replay,
    load_eval_impl,
    target_full_to_draft,
    train_replay,
)
from replay_dspark_anchor_step import (
    dtype_of,
    load_vllm_sample,
    load_vocab_maps,
    text,
)

log = logging.getLogger("anchor_distribution_pos1_probe")


def load_eval_helpers(torch):
    module = load_eval_impl(torch)
    from transformers import DynamicCache

    module.DynamicCache = DynamicCache
    return module


def valid_anchors(torch, loss_mask, block_size):
    valid = loss_mask.bool().clone()
    valid[-block_size:] = False
    return torch.nonzero(valid, as_tuple=False).view(-1).tolist()


def pick_even(items, count):
    if len(items) <= count:
        return items
    if count <= 1:
        return [items[0]]
    step = (len(items) - 1) / float(count - 1)
    return [items[round(i * step)] for i in range(count)]


class Acc:
    def __init__(self, name):
        self.name = name
        self.total = 0
        self.correct = 0
        self.soft_sum = 0.0

    def add(self, correct, soft):
        self.total += 1
        self.correct += int(bool(correct))
        self.soft_sum += float(soft)

    def line(self):
        acc = self.correct / self.total if self.total else 0.0
        soft = self.soft_sum / self.total if self.total else 0.0
        return f"{self.name}: acc={acc:.6f} ({self.correct}/{self.total}) soft={soft:.6f}"


def draft_logits_for_infer(torch, runner, draft, output_ids, anchor):
    prefix_before = output_ids[:, :anchor]
    with torch.inference_mode():
        out_before = runner.target_model(
            input_ids=prefix_before,
            position_ids=torch.arange(anchor, device=output_ids.device).unsqueeze(0),
            use_cache=False,
            output_hidden_states=True,
        )
        hidden = runner._extract_context_feature(out_before.hidden_states)
        hidden, base_logits = runner._single_anchor_backbone(hidden, output_ids, anchor)
        slot = 1
        anchor_id = output_ids[:, anchor : anchor + 1].long()
        if draft.markov_head is None:
            bias = torch.zeros_like(base_logits[:, slot : slot + 1, :])
        else:
            bias = draft.markov_head.block_bias(
                prev_token_ids=anchor_id,
                hidden_states=hidden[:, slot : slot + 1, :],
            )
        final = base_logits[0, slot] + bias[0, 0]
        out_anchor = runner.target_model(
            input_ids=output_ids[:, : anchor + 1],
            position_ids=torch.arange(anchor + 1, device=output_ids.device).unsqueeze(0),
            use_cache=False,
        )
        target = target_full_to_draft(torch, draft, out_anchor.logits[:, -1, :])[0]
    return final, target


def soft_overlap(torch, draft_logits, target_logits):
    q = torch.softmax(draft_logits.float(), dim=-1)
    p = torch.softmax(target_logits.float(), dim=-1)
    return float(torch.minimum(q, p).sum().item())


def top1_match(torch, draft, draft_logits, target_logits):
    draft_id = int(torch.argmax(draft_logits).item())
    target_id = int(torch.argmax(target_logits).item())
    draft_target = draft_id
    target_target = target_id
    if draft.use_draft_vocab and draft.d2t is not None:
        draft_target = int((draft_id + draft.d2t[draft_id].item()))
        target_target = int((target_id + draft.d2t[target_id].item()))
    return draft_target == target_target, draft_target, target_target


def run_training_uniform(torch, args, draft, tokenizer):
    stats = Acc("train_token_uniform")
    for sample_index in range(args.train_sample_start, args.train_sample_start + args.train_samples):
        args.sample_index = sample_index
        sample = load_vllm_sample(torch, args)
        anchors = pick_even(
            valid_anchors(torch, sample.loss_mask, int(draft.block_size)),
            args.train_anchors_per_sample,
        )
        for anchor in anchors:
            train, _ = train_replay(torch, draft, tokenizer, sample, anchor)
            stats.add(train.greedy_match, train.soft_overlap)
    return stats


def record_round(torch, draft, proposal, verification):
    draft_token = int(proposal.verify_input_ids[0, 1].item())
    target_token = int(torch.argmax(verification.target_output.logits[0, 0]).item())
    correct = draft_token == target_token
    if verification.support_accept_rates is not None:
        soft = float(verification.support_accept_rates[0, 0].float().item())
    else:
        soft = float(correct)
    return correct, soft, draft_token, target_token


def generate_with_round_records(torch, eval_impl, runner, input_ids, max_new_tokens):
    device = input_ids.device
    target = runner.target_model
    past = eval_impl.DynamicCache()
    max_proposal = runner.max_proposal_tokens
    output_ids = torch.empty(
        (1, input_ids.shape[1] + max_new_tokens + max_proposal + 2),
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    with torch.inference_mode():
        out = target(
            input_ids=input_ids,
            position_ids=position_ids[:, : input_ids.shape[1]],
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
        )
    output_ids[:, : input_ids.shape[1]] = input_ids
    output_ids[:, input_ids.shape[1]] = torch.argmax(out.logits[:, -1, :], dim=-1)
    context = runner._init_context(initial_output=out)
    start = input_ids.shape[1]
    max_length = input_ids.shape[1] + max_new_tokens
    records = []

    while start < max_length:
        proposal = runner._propose(
            context=context,
            output_ids=output_ids,
            position_ids=position_ids,
            start=start,
        )
        verification = eval_impl.verify_draft_tokens(
            target_model=target,
            proposal=proposal,
            position_ids=position_ids,
            start=start,
            past_key_values_target=past,
            temperature=0.0,
            max_proposal_tokens=max_proposal,
            current_token_ids=output_ids[:, start : start + 1],
        )
        correct, soft, draft_token, target_token = record_round(
            torch, runner.draft_model, proposal, verification
        )
        records.append(
            {
                "anchor": start,
                "correct": correct,
                "soft": soft,
                "accepted": int(verification.accepted_draft_tokens),
                "draft": draft_token,
                "target": target_token,
            }
        )
        accepted = int(verification.accepted_draft_tokens)
        output_ids[:, start : start + accepted + 1] = proposal.verify_input_ids[
            :, : accepted + 1
        ]
        output_ids[:, start + accepted + 1] = verification.next_token
        start += accepted + 1
        past.crop(start)
        runner._update(context, verification)

    return output_ids[:, : min(start + 1, max_length)], records


def prompt_from_dataset(eval_impl, tokenizer, dataset_path, sample_index):
    records = eval_impl._load_jsonl(Path(dataset_path))
    return eval_impl._prompt_from_record(
        records[sample_index],
        tokenizer,
        source=f"{dataset_path}:{sample_index}",
    )


def run_inference_metrics(torch, args, eval_impl, runner, draft, tokenizer):
    round_stats = Acc("infer_round_weighted")
    token_stats = Acc("infer_generated_token_uniform")
    for prompt_index in range(args.prompt_start, args.prompt_start + args.num_prompts):
        if args.prompt:
            prompt = args.prompt
        else:
            prompt = prompt_from_dataset(eval_impl, tokenizer, args.prompt_dataset, prompt_index)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        output_ids, rounds = generate_with_round_records(
            torch, eval_impl, runner, input_ids, args.max_new_tokens
        )
        for item in rounds:
            round_stats.add(item["correct"], item["soft"])
        start = input_ids.shape[1]
        anchors = list(range(start, max(output_ids.shape[1] - 1, start)))
        anchors = pick_even(anchors, args.generated_token_anchors)
        for anchor in anchors:
            draft_logits, target_logits = draft_logits_for_infer(
                torch, runner, draft, output_ids, anchor
            )
            correct, _, _ = top1_match(torch, draft, draft_logits, target_logits)
            token_stats.add(correct, soft_overlap(torch, draft_logits, target_logits))
        log.info(
            "prompt=%d rounds=%d generated_token_anchors=%d avg_accepted=%.3f",
            prompt_index,
            len(rounds),
            len(anchors),
            sum(x["accepted"] for x in rounds) / max(len(rounds), 1),
        )
    return round_stats, token_stats


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from speculators.models.dspark.core import DSparkDraftModel

    torch.manual_seed(args.seed)
    if args.prompt is None and args.prompt_dataset is None:
        raise ValueError("provide --prompt or --prompt-dataset")
    eval_impl = load_eval_helpers(torch)
    tokenizer = AutoTokenizer.from_pretrained(
        args.verifier_model, trust_remote_code=args.trust_remote_code
    )
    cfg = DSparkDraftModel.config_class.from_pretrained(args.draft_model)
    if getattr(cfg, "sample_from_anchor", False) or args.sample_from_anchor:
        raise ValueError("script only supports slot1-as-first-draft mode")
    if args.draft_attn_impl != "auto":
        cfg.transformer_layer_config._attn_implementation = args.draft_attn_impl
    d2t, t2d = load_vocab_maps(torch, args)
    draft = DSparkDraftModel.from_pretrained(
        args.draft_model, config=cfg, d2t=d2t, t2d=t2d
    ).to(args.device).eval()
    target = AutoModelForCausalLM.from_pretrained(
        args.verifier_model,
        torch_dtype=dtype_of(torch, args.dtype),
        trust_remote_code=args.trust_remote_code,
    ).to(args.device).eval()
    runner = eval_impl.DSparkOfflineRunner(
        target, draft, tokenizer, SimpleNamespace(temperature=0.0)
    )

    train_stats = run_training_uniform(torch, args, draft, tokenizer)
    round_stats, token_stats = run_inference_metrics(
        torch, args, eval_impl, runner, draft, tokenizer
    )
    log.info("")
    log.info("=== Anchor Distribution Pos1 Metrics ===")
    log.info(train_stats.line())
    log.info(round_stats.line())
    log.info(token_stats.line())
    if token_stats.total:
        log.info(
            "round_minus_generated_token_acc=%.6f",
            round_stats.correct / round_stats.total
            - token_stats.correct / token_stats.total,
        )


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
    p.add_argument("--train-sample-start", type=int, default=0)
    p.add_argument("--train-samples", type=int, default=8)
    p.add_argument("--train-anchors-per-sample", type=int, default=8)
    p.add_argument("--prompt-dataset", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--prompt-start", type=int, default=0)
    p.add_argument("--num-prompts", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--generated-token-anchors", type=int, default=64)
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
