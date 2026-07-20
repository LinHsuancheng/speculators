#!/usr/bin/env python3
"""Print one DSpark draft proposal for a single sample.

This is intentionally narrow: it runs one target prefill, builds one DSpark
draft block with the offline evaluator path, and prints the sampled draft tokens
plus top-k alternatives for each speculative slot.
"""

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import dspark_offline_eval as eval_impl

logger = logging.getLogger("dspark_debug_one_step")


def _token_text(tokenizer, token_id: int) -> str:
    return repr(tokenizer.decode([int(token_id)], skip_special_tokens=False))


def _load_prompt(args: argparse.Namespace, tokenizer) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.dataset is None:
        raise ValueError("Provide either --prompt or --dataset.")

    records = eval_impl._load_jsonl(args.dataset)
    if args.sample_index < 1 or args.sample_index > len(records):
        raise ValueError(
            f"--sample-index must be in [1, {len(records)}], got {args.sample_index}"
        )
    return eval_impl._prompt_from_record(
        records[args.sample_index - 1],
        tokenizer,
        source=f"{args.dataset}:{args.sample_index}",
    )


def _row(
    *,
    tokenizer,
    slot: int,
    rank: int,
    draft_id: int,
    target_id: int,
    prob: float,
    logit: float,
) -> dict[str, Any]:
    return {
        "slot": slot,
        "rank": rank,
        "draft_id": draft_id,
        "target_id": target_id,
        "text": _token_text(tokenizer, target_id),
        "prob": prob,
        "logit": logit,
    }


def _print_topk(tokenizer, *, slot: int, logits, probs, draft_model, top_k: int) -> None:
    k = min(int(top_k), probs.shape[-1])
    top_probs, top_draft_ids = probs.float().topk(k, dim=-1)
    rows = []
    for rank in range(k):
        draft_id = int(top_draft_ids[0, 0, rank].item())
        target_id = eval_impl._draft_ids_to_target_ids(draft_model, [draft_id])[0]
        rows.append(
            _row(
                tokenizer=tokenizer,
                slot=slot,
                rank=rank + 1,
                draft_id=draft_id,
                target_id=target_id,
                prob=float(top_probs[0, 0, rank].item()),
                logit=float(logits[0, 0, draft_id].float().item()),
            )
        )

    for item in rows:
        logger.info(
            (
                "slot=%02d rank=%02d draft_id=%d target_id=%d "
                "prob=%.6g logit=%.6g text=%s"
            ),
            item["slot"],
            item["rank"],
            item["draft_id"],
            item["target_id"],
            item["prob"],
            item["logit"],
            item["text"],
        )


def run(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from speculators.models.dspark.core import DSparkDraftModel

    eval_impl.torch = torch
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype) if args.dtype != "auto" else "auto"

    tokenizer = AutoTokenizer.from_pretrained(
        args.verifier_model,
        trust_remote_code=args.trust_remote_code,
    )
    prompt = _load_prompt(args, tokenizer)
    logger.info("prompt_chars=%d", len(prompt))
    logger.info("prompt_preview=%r", prompt[:300])

    target_model = AutoModelForCausalLM.from_pretrained(
        args.verifier_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()

    draft_config = DSparkDraftModel.config_class.from_pretrained(args.draft_model)
    if args.sample_from_anchor:
        draft_config.sample_from_anchor = True
    draft_attn_impl = eval_impl._resolve_draft_attn_impl(args.device, args.draft_attn_impl)
    if draft_attn_impl is not None:
        draft_config.transformer_layer_config._attn_implementation = draft_attn_impl
    d2t, t2d = eval_impl._load_vocab_mapping_tensors(
        draft_model_path=args.draft_model,
        d2t_path=args.d2t_path,
        t2d_path=args.t2d_path,
    )
    draft_model = DSparkDraftModel.from_pretrained(
        args.draft_model,
        config=draft_config,
        d2t=d2t,
        t2d=t2d,
    ).to(device).eval()
    eval_impl._ensure_loaded_vocab_mappings(draft_model, args)

    runner_args = SimpleNamespace(temperature=args.temperature)
    runner = eval_impl.DSparkOfflineRunner(
        target_model,
        draft_model,
        tokenizer,
        runner_args,
    )
    if args.max_proposal_tokens is not None:
        runner.max_proposal_tokens = int(args.max_proposal_tokens)

    logger.info(
        (
            "draft block_size=%s max_proposal_tokens=%s use_draft_vocab=%s "
            "draft_vocab=%s verifier_vocab=%s markov_head=%s sample_from_anchor=%s"
        ),
        draft_model.block_size,
        runner.max_proposal_tokens,
        draft_model.use_draft_vocab,
        draft_model.draft_vocab_size,
        draft_model.verifier_vocab_size,
        draft_model.markov_head is not None,
        runner.sample_from_anchor,
    )
    logger.info("loaded | mem=%s", eval_impl._format_device_memory())

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output_ids = torch.empty(
        (1, input_ids.shape[1] + runner.max_proposal_tokens + 2),
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)

    with torch.inference_mode():
        target_output = target_model(
            input_ids=input_ids,
            position_ids=position_ids[:, : input_ids.shape[1]],
            use_cache=True,
            output_hidden_states=True,
        )
        target_probs = eval_impl.logits_to_probs(
            target_output.logits[:, -1:, :],
            float(args.temperature),
        )
        anchor_token = eval_impl.sample_from_probs(target_probs)
        output_ids[:, : input_ids.shape[1]] = input_ids
        output_ids[:, input_ids.shape[1] : input_ids.shape[1] + 1] = anchor_token

        start = input_ids.shape[1]
        context = runner._init_context(
            initial_output=target_output,
            output_ids=output_ids,
            position_ids=position_ids,
            num_input_tokens=input_ids.shape[1],
        )
        hidden, base_logits = runner._single_anchor_backbone(
            context.target_hidden_states,
            output_ids,
            start,
        )

        anchor_id = int(anchor_token[0, 0].item())
        logger.info(
            "input_tokens=%d start=%d anchor_target_id=%d anchor_text=%s",
            input_ids.shape[1],
            start,
            anchor_id,
            _token_text(tokenizer, anchor_id),
        )
        logger.info(
            "target_hidden_shape=%s draft_hidden_shape=%s base_logits_shape=%s",
            tuple(context.target_hidden_states.shape),
            tuple(hidden.shape),
            tuple(base_logits.shape),
        )

        sampled_target_ids = []
        prev_token = anchor_token.reshape(1, 1).long()
        first_slot = 0 if runner.sample_from_anchor else 1
        for token_idx in range(runner.max_proposal_tokens):
            slot = first_slot + token_idx
            logits = base_logits[:, slot : slot + 1, :]
            if draft_model.markov_head is not None:
                logits = logits + draft_model.markov_head.block_bias(
                    prev_token_ids=prev_token,
                    hidden_states=hidden[:, slot : slot + 1, :],
                )
            probs = eval_impl.logits_to_probs(logits, float(args.temperature))
            draft_id = int(eval_impl.sample_from_probs(probs)[0, 0].item())
            target_id = eval_impl._draft_ids_to_target_ids(draft_model, [draft_id])[0]
            sampled_target_ids.append(target_id)
            logger.info(
                "slot=%02d sampled draft_id=%d target_id=%d prev_target_id=%d text=%s",
                slot,
                draft_id,
                target_id,
                int(prev_token[0, 0].item()),
                _token_text(tokenizer, target_id),
            )
            _print_topk(
                tokenizer,
                slot=slot,
                logits=logits,
                probs=probs,
                draft_model=draft_model,
                top_k=args.top_k,
            )
            prev_token = torch.tensor([[target_id]], dtype=torch.long, device=device)

        logger.info("proposal_target_ids=%s", sampled_target_ids)
        logger.info(
            "proposal_text=%s",
            repr(tokenizer.decode(sampled_target_ids, skip_special_tokens=False)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug one DSpark draft step.")
    parser.add_argument("--verifier-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-proposal-tokens", type=int, default=None)
    parser.add_argument("--device", default="npu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--d2t-path", type=Path, default=None)
    parser.add_argument("--t2d-path", type=Path, default=None)
    parser.add_argument(
        "--sample-from-anchor",
        action="store_true",
        help="Enable PR 806 DSpark Markov previous-token alignment.",
    )
    parser.add_argument(
        "--draft-attn-impl",
        choices=["auto", "simple_flex_attention", "sdpa", "eager"],
        default="auto",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
