#!/usr/bin/env python3
"""Compare DSpark slot accuracy on stored vs target-greedy continuations."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from block_slot_acc_probe import SlotStats, load_eval_impl, slot_records
from replay_dspark_anchor_step import dtype_of, load_vocab_maps, text

log = logging.getLogger("stored_vs_greedy_continuation_slot_probe")


@dataclass
class AnchorPair:
    offset: int
    anchor_a: int
    anchor_b: int


def load_jsonl_record(path: Path, index: int) -> dict:
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no == index:
                return json.loads(line)
    raise IndexError(f"{path} has no record at index {index}")


def normalized_prompt_messages(record: dict, assistant_turn: int):
    raw_messages = record.get("messages", record.get("conversations"))
    if not isinstance(raw_messages, list):
        return None

    role_map = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
        "tool": "tool",
    }
    messages = []
    seen_assistants = 0
    for turn in raw_messages:
        if not isinstance(turn, dict):
            return None
        raw_role = turn.get("from", turn.get("role"))
        raw_content = turn.get("value", turn.get("content"))
        if not isinstance(raw_role, str) or not isinstance(raw_content, str):
            return None
        role = role_map.get(raw_role)
        if role is None:
            continue
        if role == "assistant":
            if seen_assistants == assistant_turn:
                return messages
            seen_assistants += 1
        messages.append({"role": role, "content": raw_content})
    return None


def apply_chat_template(tokenizer, messages, enable_thinking: str):
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    if enable_thinking != "default":
        kwargs["enable_thinking"] = enable_thinking == "true"
    return tokenizer.apply_chat_template(messages, **kwargs)


def prompt_ids_from_raw(torch, tokenizer, args):
    if args.raw_jsonl is None:
        return None
    raw_index = args.raw_sample_index
    if raw_index is None:
        raw_index = args.sample_index
    record = load_jsonl_record(args.raw_jsonl, raw_index)
    messages = normalized_prompt_messages(record, args.assistant_turn)
    if messages is None:
        raise ValueError(
            f"cannot build prompt messages from {args.raw_jsonl}:{raw_index}; "
            "expected messages or conversations"
        )
    prompt_ids = apply_chat_template(tokenizer, messages, args.enable_thinking)
    prompt_ids = torch.as_tensor(prompt_ids, dtype=torch.long)
    if prompt_ids.ndim == 2:
        prompt_ids = prompt_ids[0]
    log.info(
        "raw_jsonl=%s raw_sample_index=%d assistant_turn=%d prompt_messages=%d "
        "enable_thinking=%s",
        args.raw_jsonl,
        raw_index,
        args.assistant_turn,
        len(messages),
        args.enable_thinking,
    )
    return prompt_ids


def first_loss_position(torch, loss_mask) -> int:
    positions = torch.nonzero(loss_mask.bool(), as_tuple=False).view(-1)
    if positions.numel() == 0:
        raise ValueError("sample has no loss_mask==1 positions")
    return int(positions[0].item())


def prompt_ids_for_sample(torch, tokenizer, stored_ids, loss_mask, args):
    prompt_ids = prompt_ids_from_raw(torch, tokenizer, args)
    if prompt_ids is None:
        response_start = first_loss_position(torch, loss_mask)
        prompt_ids = stored_ids[:response_start].clone()
        log.warning(
            "no --raw-jsonl supplied; falling back to response_start=first loss_mask "
            "position (%d). This checks token-prefix behavior but does not prove the "
            "raw chat-template contract.",
            response_start,
        )
        return prompt_ids, response_start, "loss_mask_fallback"

    response_start = int(prompt_ids.numel())
    stored_prefix = stored_ids[:response_start].cpu()
    if not torch.equal(stored_prefix, prompt_ids.cpu()):
        diff = first_token_diff(stored_prefix.tolist(), prompt_ids.cpu().tolist())
        log.error("raw prompt_ids do not match stored prefix; first_diff=%s", diff)
        print_token_window(
            tokenizer,
            "stored_prefix_around_diff",
            stored_ids,
            max(0, int(diff or 0) - 8),
            min(stored_ids.numel(), int(diff or 0) + 16),
            loss_mask,
        )
        if args.strict_prompt_prefix:
            raise AssertionError("stored_ids prefix != raw reconstructed prompt_ids")
    else:
        log.info("stored prompt prefix matches raw reconstructed prompt_ids")
    return prompt_ids, response_start, "raw_chat_template"


def first_token_diff(a: list[int], b: list[int]) -> int | None:
    for idx, (x, y) in enumerate(zip(a, b, strict=False)):
        if int(x) != int(y):
            return idx
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def generate_greedy(torch, model, prompt_ids, max_new_tokens: int, device: str):
    input_ids = prompt_ids.to(device).unsqueeze(0)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=False,
        )
    return generated[0].detach().long().cpu()


def replay_anchor_rows(torch, eval_impl, runner, output_ids, anchor: int):
    device = output_ids.device
    target = runner.target_model
    past = eval_impl.DynamicCache()
    position_ids = torch.arange(
        output_ids.shape[1] + runner.max_proposal_tokens + 2,
        device=device,
    ).unsqueeze(0)
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
    return slot_records(torch, proposal, verification)


def add_rows(stats: SlotStats, rows):
    for row in rows:
        stats.add(row["slot"], row["correct"], row["soft"])


def make_offsets(torch, common_len: int, block_size: int, count: int):
    if common_len <= block_size:
        return []
    valid_offsets = torch.arange(0, common_len - block_size + 1, dtype=torch.long)
    if valid_offsets.numel() <= count:
        return valid_offsets.tolist()
    perm = torch.randperm(valid_offsets.numel())[:count]
    return valid_offsets[perm].tolist()


def first_divergence(stored_cont, free_cont):
    limit = min(stored_cont.numel(), free_cont.numel())
    for offset in range(limit):
        if int(stored_cont[offset].item()) != int(free_cont[offset].item()):
            return offset
    if stored_cont.numel() != free_cont.numel():
        return limit
    return None


def print_token_window(tokenizer, title, ids, start, end, loss_mask=None):
    log.info("")
    log.info("=== %s [%d:%d] ===", title, start, end)
    for idx in range(start, end):
        mask = "-" if loss_mask is None else int(loss_mask[idx].item())
        log.info(
            "%-6d id=%-8d mask=%s text=%s",
            idx,
            int(ids[idx].item()),
            mask,
            text(tokenizer, int(ids[idx].item())),
        )


def print_continuation(tokenizer, title, ids, response_start, max_tokens):
    end = min(ids.numel(), response_start + max_tokens)
    log.info("")
    log.info("=== %s decoded [%d:%d] ===", title, response_start, end)
    log.info("%s", repr(tokenizer.decode(ids[response_start:end].tolist())))


def print_stats_table(stats_list):
    block_size = len(stats_list[0][1].total)
    header = ["metric"] + [f"slot{i}" for i in range(1, block_size)]
    log.info("")
    log.info("=== summary ===")
    log.info(" ".join(f"{x:>18}" for x in header))
    for name, stats in stats_list:
        values = [name]
        for slot in range(1, block_size):
            total = stats.total[slot]
            values.append("n/a" if total == 0 else f"{stats.correct[slot] / total:.6f}")
        log.info(" ".join(f"{x:>18}" for x in values))


def print_anchor_examples(torch, tokenizer, examples):
    if not examples:
        return
    log.info("")
    log.info("=== paired_anchor_examples ===")
    for ex in examples:
        offset = ex["offset"]
        log.info(
            "offset=%d A_anchor=%d %s B_anchor=%d %s",
            offset,
            ex["anchor_a"],
            text(tokenizer, ex["anchor_token_a"]),
            ex["anchor_b"],
            text(tokenizer, ex["anchor_token_b"]),
        )
        for label, rows in (("A", ex["rows_a"]), ("B", ex["rows_b"])):
            parts = []
            for row in rows:
                parts.append(
                    "slot%d draft=%d %s target=%d %s ok=%s"
                    % (
                        row["slot"],
                        row["draft"],
                        text(tokenizer, row["draft"]),
                        row["target"],
                        text(tokenizer, row["target"]),
                        row["correct"],
                    )
                )
            log.info("%s %s", label, " | ".join(parts))


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
        dtype=dtype_of(torch, args.dtype),
        trust_remote_code=args.trust_remote_code,
    ).to(args.device).eval()
    runner = eval_impl.DSparkOfflineRunner(
        target,
        draft,
        tokenizer,
        SimpleNamespace(temperature=0.0),
    )

    data = load_from_disk(args.data_path)
    item = data[int(args.sample_index)]
    stored_ids = torch.as_tensor(item["input_ids"], dtype=torch.long)
    loss_mask = torch.as_tensor(item["loss_mask"], dtype=torch.bool)
    prompt_ids, response_start_a, prompt_source = prompt_ids_for_sample(
        torch,
        tokenizer,
        stored_ids,
        loss_mask,
        args,
    )
    stored_cont = stored_ids[response_start_a:]
    max_new_tokens = args.max_new_tokens or int(stored_cont.numel())
    free_ids = generate_greedy(
        torch,
        target,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        device=args.device,
    )
    response_start_b = int(prompt_ids.numel())
    free_cont = free_ids[response_start_b:]
    common_len = min(int(stored_cont.numel()), int(free_cont.numel()))
    first_diff = first_divergence(stored_cont[:common_len], free_cont[:common_len])
    offsets = make_offsets(torch, common_len, draft.block_size, args.num_anchors)
    pairs = [
        AnchorPair(
            offset=int(offset),
            anchor_a=response_start_a + int(offset),
            anchor_b=response_start_b + int(offset),
        )
        for offset in offsets
    ]

    log.info("")
    log.info("=== sample ===")
    log.info("sample_id=%d prompt_source=%s", args.sample_index, prompt_source)
    log.info("prompt_tokens=%d", int(prompt_ids.numel()))
    log.info("stored_tokens=%d", int(stored_ids.numel()))
    log.info("stored_continuation_tokens=%d", int(stored_cont.numel()))
    log.info("free_generated_tokens=%d", int(free_cont.numel()))
    log.info("common_continuation_tokens=%d", common_len)
    log.info("first_divergence_offset=%s", first_diff)
    log.info("anchors=%d block_size=%d", len(pairs), draft.block_size)

    print_token_window(
        tokenizer,
        "stored_loss_mask_boundary",
        stored_ids,
        max(0, response_start_a - args.boundary_tokens),
        min(stored_ids.numel(), response_start_a + args.boundary_tokens),
        loss_mask,
    )
    print_continuation(
        tokenizer,
        "stored_continuation",
        stored_ids,
        response_start_a,
        args.decode_tokens,
    )
    print_continuation(
        tokenizer,
        "free_greedy_continuation",
        free_ids,
        response_start_b,
        args.decode_tokens,
    )

    output_a = stored_ids.to(args.device).unsqueeze(0)
    output_b = free_ids.to(args.device).unsqueeze(0)
    stats_a = SlotStats("A_stored", draft.block_size)
    stats_b = SlotStats("B_free_greedy", draft.block_size)
    stats_a_before = SlotStats("A_before_divergence", draft.block_size)
    stats_b_before = SlotStats("B_before_divergence", draft.block_size)
    stats_a_after = SlotStats("A_after_divergence", draft.block_size)
    stats_b_after = SlotStats("B_after_divergence", draft.block_size)
    examples = []

    for pair in pairs:
        rows_a = replay_anchor_rows(torch, eval_impl, runner, output_a, pair.anchor_a)
        rows_b = replay_anchor_rows(torch, eval_impl, runner, output_b, pair.anchor_b)
        add_rows(stats_a, rows_a)
        add_rows(stats_b, rows_b)
        if first_diff is None or pair.offset < first_diff:
            add_rows(stats_a_before, rows_a)
            add_rows(stats_b_before, rows_b)
            for ra, rb in zip(rows_a, rows_b, strict=True):
                if ra["draft"] != rb["draft"] or ra["target"] != rb["target"]:
                    raise AssertionError(
                        "A/B rows differ before first divergence: "
                        f"offset={pair.offset} slot={ra['slot']} A={ra} B={rb}"
                    )
        else:
            add_rows(stats_a_after, rows_a)
            add_rows(stats_b_after, rows_b)
        if len(examples) < args.show_anchors:
            examples.append(
                {
                    "offset": pair.offset,
                    "anchor_a": pair.anchor_a,
                    "anchor_b": pair.anchor_b,
                    "anchor_token_a": int(output_a[0, pair.anchor_a].item()),
                    "anchor_token_b": int(output_b[0, pair.anchor_b].item()),
                    "rows_a": rows_a,
                    "rows_b": rows_b,
                }
            )

    print_stats_table(
        [
            ("A_stored", stats_a),
            ("B_free_greedy", stats_b),
            ("A_before_div", stats_a_before),
            ("B_before_div", stats_b_before),
            ("A_after_div", stats_a_after),
            ("B_after_div", stats_b_after),
        ]
    )
    stats_a.print()
    stats_b.print()
    stats_a_before.print()
    stats_b_before.print()
    stats_a_after.print()
    stats_b_after.print()
    print_anchor_examples(torch, tokenizer, examples)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--verifier-model", required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--sample-index", type=int, required=True)
    p.add_argument("--raw-jsonl", type=Path, default=None)
    p.add_argument("--raw-sample-index", type=int, default=None)
    p.add_argument("--assistant-turn", type=int, default=0)
    p.add_argument(
        "--enable-thinking",
        choices=["default", "true", "false"],
        default="false",
    )
    p.add_argument(
        "--strict-prompt-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--num-anchors", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=0)
    p.add_argument("--boundary-tokens", type=int, default=20)
    p.add_argument("--decode-tokens", type=int, default=160)
    p.add_argument("--show-anchors", type=int, default=5)
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
