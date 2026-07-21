#!/usr/bin/env python3
"""Resample DATA_PATH next tokens with the verifier and compare stored tokens."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from replay_dspark_anchor_step import dtype_of, text

log = logging.getLogger("preparedata_resample_next_token_probe")


@dataclass
class Totals:
    correct: int = 0
    total: int = 0

    def add(self, matched: bool) -> None:
        self.correct += int(matched)
        self.total += 1

    @property
    def acc(self) -> float:
        if self.total == 0:
            return 0.0
        return self.correct / self.total


def sample_indices(torch, dataset_len: int, start: int, count: int, randomize: bool):
    available = list(range(start, dataset_len))
    if len(available) <= count:
        return available
    if not randomize:
        return available[:count]
    perm = torch.randperm(len(available))[:count].tolist()
    return [available[i] for i in perm]


def target_positions(torch, loss_mask, include_unmasked: bool):
    if include_unmasked:
        return torch.arange(1, loss_mask.numel(), dtype=torch.long)
    valid = loss_mask.bool().clone()
    valid[0] = False
    return torch.nonzero(valid, as_tuple=False).view(-1).long()


def sample_positions(torch, positions, count: int, randomize: bool):
    if positions.numel() <= count:
        return positions.tolist()
    if not randomize:
        return positions[:count].tolist()
    perm = torch.randperm(positions.numel())[:count]
    return positions[perm].tolist()


def greedy_full_sequence(torch, model, input_ids, positions):
    if input_ids.shape[1] < 2:
        return {}
    with torch.inference_mode():
        out = model(input_ids=input_ids[:, :-1], use_cache=False)
    logits = out.logits[0]
    preds = {}
    for target_pos in positions:
        preds[int(target_pos)] = int(torch.argmax(logits[int(target_pos) - 1]).item())
    return preds


def greedy_single_prefix(torch, model, input_ids, positions):
    preds = {}
    with torch.inference_mode():
        for target_pos in positions:
            out = model(input_ids=input_ids[:, : int(target_pos)], use_cache=False)
            preds[int(target_pos)] = int(torch.argmax(out.logits[0, -1]).item())
    return preds


def print_examples(tokenizer, examples):
    if not examples:
        return
    log.info("")
    log.info("=== mismatch_examples ===")
    for row in examples:
        pred_text = text(tokenizer, row["pred_id"]) if tokenizer is not None else "n/a"
        true_text = text(tokenizer, row["true_id"]) if tokenizer is not None else "n/a"
        anchor_text = text(tokenizer, row["anchor_id"]) if tokenizer is not None else "n/a"
        log.info(
            "sample=%d target_pos=%d anchor_pos=%d anchor_id=%d %s "
            "pred=%d %s true=%d %s",
            row["sample"],
            row["target_pos"],
            row["anchor_pos"],
            row["anchor_id"],
            anchor_text,
            row["pred_id"],
            pred_text,
            row["true_id"],
            true_text,
        )


def run(args):
    import torch
    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    data = load_from_disk(args.data_path)
    indices = sample_indices(
        torch,
        len(data),
        args.sample_start,
        args.num_samples,
        args.random_samples,
    )
    tokenizer = None
    if args.show_examples > 0:
        tokenizer = AutoTokenizer.from_pretrained(
            args.verifier_model,
            trust_remote_code=args.trust_remote_code,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.verifier_model,
        dtype=dtype_of(torch, args.dtype),
        trust_remote_code=args.trust_remote_code,
    ).to(args.device).eval()

    totals = Totals()
    skipped = 0
    examples = []

    for sample_index in indices:
        item = data[int(sample_index)]
        ids = torch.as_tensor(item["input_ids"], dtype=torch.long)
        loss_mask = torch.as_tensor(item["loss_mask"], dtype=torch.bool)
        positions = target_positions(torch, loss_mask, args.include_unmasked)
        positions = sample_positions(
            torch,
            positions,
            args.tokens_per_sample,
            args.random_tokens,
        )
        if not positions:
            skipped += 1
            log.info("sample=%d skipped=no_target_positions", sample_index)
            continue

        input_ids = ids.to(args.device).unsqueeze(0)
        if args.mode == "single-prefix":
            preds = greedy_single_prefix(torch, model, input_ids, positions)
        else:
            preds = greedy_full_sequence(torch, model, input_ids, positions)

        sample_totals = Totals()
        for target_pos in positions:
            pred_id = int(preds[int(target_pos)])
            true_id = int(ids[int(target_pos)].item())
            matched = pred_id == true_id
            totals.add(matched)
            sample_totals.add(matched)
            if not matched and len(examples) < args.show_examples:
                anchor_pos = int(target_pos) - 1
                examples.append(
                    {
                        "sample": int(sample_index),
                        "target_pos": int(target_pos),
                        "anchor_pos": anchor_pos,
                        "anchor_id": int(ids[anchor_pos].item()),
                        "pred_id": pred_id,
                        "true_id": true_id,
                    }
                )

        log.info(
            "sample=%d tokens=%d checked=%d acc=%.6f (%d/%d)",
            sample_index,
            ids.numel(),
            sample_totals.total,
            sample_totals.acc,
            sample_totals.correct,
            sample_totals.total,
        )

    log.info("")
    log.info("=== preparedata_target_greedy_resample ===")
    log.info("mode=%s include_unmasked=%s", args.mode, args.include_unmasked)
    log.info("samples=%d skipped=%d", len(indices), skipped)
    log.info("acc=%.6f (%d/%d)", totals.acc, totals.correct, totals.total)
    print_examples(tokenizer, examples)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--verifier-model", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--sample-start", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--tokens-per-sample", type=int, default=64)
    p.add_argument("--random-samples", action="store_true")
    p.add_argument("--random-tokens", action="store_true")
    p.add_argument(
        "--mode",
        choices=["full-sequence", "single-prefix"],
        default="full-sequence",
        help=(
            "full-sequence is vectorized teacher-forced scoring; single-prefix "
            "literally reruns one prefix per checked token."
        ),
    )
    p.add_argument(
        "--include-unmasked",
        action="store_true",
        help="check all token positions instead of only loss_mask==1 result tokens",
    )
    p.add_argument("--show-examples", type=int, default=8)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
