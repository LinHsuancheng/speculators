#!/usr/bin/env python3
"""Per-slot DSpark accuracy for inference rounds vs random generated anchors."""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

from replay_dspark_anchor_step import dtype_of, load_vocab_maps, text

log = logging.getLogger("block_slot_acc_probe")


def load_eval_impl(torch):
    path = Path(__file__).parents[2] / "scripts" / "evaluate" / "dspark_offline_eval.py"
    spec = importlib.util.spec_from_file_location("dspark_offline_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.torch = torch
    from transformers import DynamicCache

    module.DynamicCache = DynamicCache
    return module


class SlotStats:
    def __init__(self, name, block_size):
        self.name = name
        self.correct = [0] * block_size
        self.total = [0] * block_size
        self.soft = [0.0] * block_size

    def add(self, slot, correct, soft):
        self.correct[slot] += int(bool(correct))
        self.total[slot] += 1
        self.soft[slot] += float(soft)

    def print(self):
        log.info("")
        log.info("=== %s ===", self.name)
        log.info("%-6s %-10s %-14s %s", "slot", "acc", "count", "soft")
        for slot in range(1, len(self.total)):
            total = self.total[slot]
            if total == 0:
                log.info("%-6d %-10s %-14s %s", slot, "n/a", "0", "n/a")
                continue
            acc = self.correct[slot] / total
            soft = self.soft[slot] / total
            log.info(
                "%-6d %-10.6f %-14s %.6f",
                slot,
                acc,
                f"{self.correct[slot]}/{total}",
                soft,
            )


def prompt_from_dataset(eval_impl, tokenizer, path, sample_index):
    records = eval_impl._load_jsonl(Path(path))
    return eval_impl._prompt_from_record(
        records[sample_index],
        tokenizer,
        source=f"{path}:{sample_index}",
    )


def slot_records(torch, proposal, verification):
    draft_count = int(proposal.draft_token_count)
    target_logits = verification.target_output.logits[:, :draft_count, :]
    target_probs = verification.target_probs[:, :draft_count, :]
    draft_probs = proposal.draft_probs[:, :draft_count, :]
    draft_top1 = draft_probs.argmax(dim=-1)
    target_top1 = target_logits.argmax(dim=-1)
    overlap = torch.minimum(draft_probs.float(), target_probs.float()).sum(dim=-1)
    rows = []
    for pos in range(draft_count):
        slot = pos + 1
        rows.append(
            {
                "slot": slot,
                "draft": int(draft_top1[0, pos].item()),
                "target": int(target_top1[0, pos].item()),
                "correct": bool(draft_top1[0, pos].item() == target_top1[0, pos].item()),
                "soft": float(overlap[0, pos].item()),
            }
        )
    return rows


def generate_one(torch, eval_impl, runner, input_ids, max_new_tokens, round_stats):
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
    rounds = 0
    accepted_sum = 0

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
        for row in slot_records(torch, proposal, verification):
            round_stats.add(row["slot"], row["correct"], row["soft"])

        accepted = int(verification.accepted_draft_tokens)
        accepted_sum += accepted
        rounds += 1
        output_ids[:, start : start + accepted + 1] = proposal.verify_input_ids[
            :, : accepted + 1
        ]
        output_ids[:, start + accepted + 1] = verification.next_token
        start += accepted + 1
        past.crop(start)
        runner._update(context, verification)

    output_ids = output_ids[:, : min(start + 1, max_length)]
    return output_ids, rounds, accepted_sum / max(rounds, 1)


def replay_anchor(torch, eval_impl, runner, output_ids, anchor, random_stats):
    device = output_ids.device
    target = runner.target_model
    past = eval_impl.DynamicCache()
    position_ids = torch.arange(output_ids.shape[1] + runner.max_proposal_tokens + 2, device=device).unsqueeze(0)
    with torch.inference_mode():
        out = target(
            input_ids=output_ids[:, :anchor],
            position_ids=position_ids[:, :anchor],
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
        )
    context = runner._init_context(initial_output=out)
    proposal = runner._propose(
        context=context,
        output_ids=output_ids,
        position_ids=position_ids,
        start=anchor,
    )
    verification = eval_impl.verify_draft_tokens(
        target_model=target,
        proposal=proposal,
        position_ids=position_ids,
        start=anchor,
        past_key_values_target=past,
        temperature=0.0,
        max_proposal_tokens=runner.max_proposal_tokens,
        current_token_ids=output_ids[:, anchor : anchor + 1],
    )
    for row in slot_records(torch, proposal, verification):
        random_stats.add(row["slot"], row["correct"], row["soft"])


def random_generated_anchors(torch, output_ids, prompt_len, block_size, count):
    end = output_ids.shape[1] - block_size
    if end < prompt_len:
        return []
    anchors = list(range(prompt_len, end + 1))
    if len(anchors) <= count:
        return anchors
    perm = torch.randperm(len(anchors))[:count].tolist()
    return [anchors[i] for i in perm]


def random_preparedata_indices(torch, dataset_len, start, count):
    available = list(range(start, dataset_len))
    if len(available) <= count:
        return available
    perm = torch.randperm(len(available))[:count].tolist()
    return [available[i] for i in perm]


def random_preparedata_anchors(torch, input_ids, loss_mask, block_size, count):
    end = input_ids.shape[1] - block_size
    if end <= 0:
        return []
    valid = loss_mask[: end + 1].bool()
    anchors = torch.nonzero(valid, as_tuple=False).view(-1).tolist()
    if len(anchors) <= count:
        return anchors
    perm = torch.randperm(len(anchors))[:count].tolist()
    return [anchors[i] for i in perm]


def run_preparedata_random(torch, args, eval_impl, runner, dataset_stats):
    from datasets import load_from_disk

    data = load_from_disk(args.data_path)
    indices = random_preparedata_indices(
        torch,
        len(data),
        args.preparedata_sample_start,
        args.preparedata_num_samples,
    )
    for sample_index in indices:
        item = data[int(sample_index)]
        input_ids = torch.as_tensor(item["input_ids"], dtype=torch.long).to(
            args.device
        ).unsqueeze(0)
        loss_mask = torch.as_tensor(item["loss_mask"], dtype=torch.bool)
        anchors = random_preparedata_anchors(
            torch,
            input_ids,
            loss_mask,
            runner.draft_model.block_size,
            args.preparedata_anchors_per_sample,
        )
        for anchor in anchors:
            replay_anchor(torch, eval_impl, runner, input_ids, anchor, dataset_stats)
        log.info(
            "preparedata_sample=%d tokens=%d random_anchors=%d",
            sample_index,
            input_ids.shape[1],
            len(anchors),
        )


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from speculators.models.dspark.core import DSparkDraftModel

    torch.manual_seed(args.seed)
    if args.prompt is None and args.prompt_dataset is None:
        raise ValueError("provide --prompt or --prompt-dataset")
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
    round_stats = SlotStats("complete_inference_proposal_rounds", draft.block_size)
    random_stats = SlotStats("random_anchors_in_generated_samples", draft.block_size)
    preparedata_stats = SlotStats("random_anchors_in_preparedata_samples", draft.block_size)

    for sample_index in range(args.sample_start, args.sample_start + args.num_samples):
        if args.prompt:
            prompt = args.prompt
        else:
            prompt = prompt_from_dataset(eval_impl, tokenizer, args.prompt_dataset, sample_index)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(args.device)
        output_ids, rounds, avg_accepted = generate_one(
            torch,
            eval_impl,
            runner,
            input_ids,
            args.max_new_tokens,
            round_stats,
        )
        anchors = random_generated_anchors(
            torch,
            output_ids,
            input_ids.shape[1],
            draft.block_size,
            args.anchors_per_sample,
        )
        for anchor in anchors:
            replay_anchor(torch, eval_impl, runner, output_ids, anchor, random_stats)
        log.info(
            "sample=%d prompt_tokens=%d output_tokens=%d rounds=%d random_anchors=%d avg_accepted=%.3f",
            sample_index,
            input_ids.shape[1],
            output_ids.shape[1] - input_ids.shape[1],
            rounds,
            len(anchors),
            avg_accepted,
        )

    round_stats.print()
    random_stats.print()
    if args.preparedata_num_samples > 0:
        run_preparedata_random(torch, args, eval_impl, runner, preparedata_stats)
        preparedata_stats.print()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--verifier-model", required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--prompt-dataset", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--sample-start", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--anchors-per-sample", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--preparedata-sample-start", type=int, default=0)
    p.add_argument("--preparedata-num-samples", type=int, default=0)
    p.add_argument("--preparedata-anchors-per-sample", type=int, default=32)
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
