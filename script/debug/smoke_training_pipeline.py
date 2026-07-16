#!/usr/bin/env python3
"""Smoke-test the two-stage online training pipeline against a live vLLM server.

Stage 1 mirrors training hidden-state extraction.  If vLLM prefix cache makes the
connector emit suffix-only hidden states, the sample is reported as skipped.

Stage 2 sends a separate prompt-logprob scoring request and intentionally does
not read any hidden-state file returned by the connector.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

DEFAULT_REQUEST_TIMEOUT = 120.0


def _parse_indices(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(part) for part in value.replace(",", " ").split()]


def _choose_indices(args: argparse.Namespace, dataset_len: int) -> list[int]:
    indices = _parse_indices(args.indices)
    if not indices:
        indices = list(range(dataset_len))
        if args.shuffle:
            rng = random.Random(args.seed)
            rng.shuffle(indices)
        indices = indices[: args.num_samples]
    for index in indices:
        if index < 0 or index >= dataset_len:
            raise IndexError(f"Index {index} outside dataset length {dataset_len}")
    return indices


def _wait_for_hidden_state_file(path: Path, timeout: float) -> None:
    from speculators.data_generation.vllm_client import wait_for_lock  # noqa: PLC0415

    lock_path = Path(str(path) + ".lock")
    deadline = time.monotonic() + timeout
    if lock_path.exists():
        wait_for_lock(str(lock_path), timeout=timeout)
    while not path.exists():
        if time.monotonic() >= deadline:
            raise FileNotFoundError(f"Timed out waiting for hidden-state file: {path}")
        if lock_path.exists():
            wait_for_lock(str(lock_path), timeout=max(deadline - time.monotonic(), 0.1))
            continue
        time.sleep(0.05)


def _get_prompt_token_ids(response: Any) -> list[int] | None:
    choices = getattr(response, "choices", None)
    if choices:
        prompt_ids = getattr(choices[0], "prompt_token_ids", None)
        if prompt_ids is not None:
            return list(prompt_ids)
    prompt_ids = getattr(response, "prompt_token_ids", None)
    if prompt_ids is not None:
        return list(prompt_ids)
    return None


def _get_kv_hidden_path(response: Any) -> str | None:
    params = getattr(response, "kv_transfer_params", None)
    if isinstance(params, dict):
        return params.get("hidden_states_path")
    return None


def _request_hidden_states(
    *,
    client,
    model: str,
    item: dict[str, Any],
    request_timeout: float,
) -> Any:
    from speculators.train.data import build_client_item  # noqa: PLC0415

    client_item = build_client_item(item)
    messages = client_item.get("messages")
    if messages is None:
        return client.completions.create(
            model=model,
            prompt=client_item["input_ids"],
            max_tokens=1,
            extra_body={"return_token_ids": True},
            timeout=request_timeout,
        )
    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1,
        extra_body={"add_generation_prompt": False, "return_token_ids": True},
        timeout=request_timeout,
    )


def _hidden_extract_stage(
    *,
    client,
    model: str,
    item: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    from safetensors.torch import load_file  # noqa: PLC0415

    expected_ids = item["input_ids"].tolist()
    response = _request_hidden_states(
        client=client,
        model=model,
        item=item,
        request_timeout=args.request_timeout,
    )
    response_prompt_ids = _get_prompt_token_ids(response)
    hs_path = _get_kv_hidden_path(response)
    if hs_path is None:
        return {
            "ok": False,
            "status": "skipped",
            "reason": "response missing kv_transfer_params.hidden_states_path",
        }, None

    _wait_for_hidden_state_file(Path(hs_path), args.file_timeout)
    loaded = load_file(hs_path)
    token_ids = loaded["token_ids"]
    hidden = loaded["hidden_states"]

    token_ids_match = token_ids.tolist() == expected_ids
    prompt_ids_match = response_prompt_ids == expected_ids
    hidden_len_match = hidden.shape[0] == len(expected_ids)
    hidden_has_nan = bool(hidden.isnan().any().item())
    ok = token_ids_match and prompt_ids_match and hidden_len_match and not hidden_has_nan

    if not args.keep_generated:
        path = Path(hs_path)
        path.unlink(missing_ok=True)
        Path(str(path) + ".lock").unlink(missing_ok=True)

    return {
        "ok": ok,
        "status": "usable" if ok else "skipped",
        "hidden_states_path": hs_path,
        "prompt_token_ids_len": None
        if response_prompt_ids is None
        else len(response_prompt_ids),
        "token_ids_len": int(token_ids.shape[0]),
        "expected_tokens": len(expected_ids),
        "hidden_states_shape": list(hidden.shape),
        "checks": {
            "token_ids_match": token_ids_match,
            "prompt_token_ids_match": prompt_ids_match,
            "hidden_seq_len_match": hidden_len_match,
            "hidden_has_nan": hidden_has_nan,
            "hidden_minus_expected_len": int(hidden.shape[0] - len(expected_ids)),
        },
    }, {"token_ids": token_ids, "hidden_states": hidden} if ok else None


def _auto_device() -> str:
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        return "cuda"
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        return "npu"
    return "cpu"


def _load_draft_model(args: argparse.Namespace):
    if not args.draft_checkpoint:
        return None

    import torch  # noqa: PLC0415

    from speculators.model import SpeculatorModel  # noqa: PLC0415
    from speculators.models.attention import create_float_mask  # noqa: PLC0415

    device = _auto_device() if args.draft_device == "auto" else args.draft_device
    dtype = getattr(torch, args.draft_dtype)
    model = SpeculatorModel.from_pretrained(args.draft_checkpoint)
    if args.draft_attn_impl:
        model.config.transformer_layer_config._attn_implementation = (  # noqa: SLF001
            args.draft_attn_impl
        )
        if hasattr(model, "_attn_impl"):
            model._attn_impl = args.draft_attn_impl  # noqa: SLF001
        if hasattr(model, "_create_mask_fn"):
            model._create_mask_fn = create_float_mask  # noqa: SLF001
        for module in model.modules():
            config = getattr(module, "config", None)
            if config is not None and hasattr(config, "_attn_implementation"):
                config._attn_implementation = args.draft_attn_impl  # noqa: SLF001
    model.to(dtype=dtype)
    model.to(device)
    model.eval()
    return model


def _target_token_id(draft_model, draft_token_id: int) -> int:
    d2t = getattr(draft_model, "d2t", None)
    if d2t is None:
        return draft_token_id
    return int(d2t[draft_token_id].item())


def _apply_markov_bias(draft_model, logits, hidden_blocks, prev_token_ids):
    if getattr(draft_model, "markov_head", None) is None:
        return logits
    prev_emb = draft_model.markov_head.prev_embeddings(prev_token_ids)
    markov_bias = draft_model.markov_head.block_bias(
        prev_token_ids=prev_token_ids,
        hidden_states=hidden_blocks,
        prev_emb=prev_emb,
    )
    return logits + markov_bias


def _sample_from_draft(
    *,
    draft_model,
    item: dict[str, Any],
    hidden_artifacts: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch  # noqa: PLC0415

    token_ids = hidden_artifacts["token_ids"].to(next(draft_model.parameters()).device)
    hidden = hidden_artifacts["hidden_states"].to(
        device=token_ids.device,
        dtype=next(draft_model.parameters()).dtype,
    )
    total_len = token_ids.shape[0]
    block_size = int(draft_model.block_size)
    sampled_len = min(args.score_tokens, block_size - 1)
    prefix_len = min(max(args.score_prefix_tokens, 1), total_len - sampled_len)
    anchor_pos = prefix_len - 1
    if anchor_pos < 0 or anchor_pos + sampled_len >= total_len:
        raise ValueError(
            f"Invalid draft sampling window: prefix_len={prefix_len}, "
            f"sampled_len={sampled_len}, total_len={total_len}"
        )

    input_ids = token_ids.unsqueeze(0)
    loss_mask = torch.zeros((1, total_len), dtype=torch.bool, device=token_ids.device)
    loss_mask[:, anchor_pos] = True
    document_ids = torch.zeros((1, total_len), dtype=torch.long, device=token_ids.device)
    position_ids = torch.arange(total_len, dtype=torch.long, device=token_ids.device)
    position_ids = position_ids.unsqueeze(0)
    hidden_states = hidden[:, :-1].flatten(1).unsqueeze(0)
    verifier_last_hidden_states = hidden[:, -1].unsqueeze(0)

    old_max_anchors = draft_model.config.max_anchors
    draft_model.config.max_anchors = 1
    try:
        with torch.no_grad():
            hidden_out, logits, _, _, _ = draft_model._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
            )
    finally:
        draft_model.config.max_anchors = old_max_anchors

    hidden_blocks = hidden_out.view(1, block_size, -1)
    logits_blocks = logits.view(1, block_size, -1)
    prev_token_ids = torch.full(
        (1, block_size),
        int(input_ids[0, anchor_pos].item()),
        dtype=torch.long,
        device=token_ids.device,
    )

    sampled_target_ids: list[int] = []
    sampled_draft_ids: list[int] = []
    draft_logprobs: list[float] = []
    for slot in range(1, sampled_len + 1):
        if slot > 1:
            prev_token_ids[0, slot] = sampled_target_ids[-1]
        biased_logits = _apply_markov_bias(
            draft_model, logits_blocks, hidden_blocks, prev_token_ids
        )
        slot_logits = biased_logits[0, slot].float()
        if args.draft_temperature <= 0:
            draft_token_id = int(torch.argmax(slot_logits).item())
            log_probs = torch.log_softmax(slot_logits, dim=-1)
        else:
            log_probs = torch.log_softmax(slot_logits / args.draft_temperature, dim=-1)
            probs = torch.softmax(slot_logits / args.draft_temperature, dim=-1)
            draft_token_id = int(torch.multinomial(probs, num_samples=1).item())
        target_token_id = _target_token_id(draft_model, draft_token_id)
        sampled_draft_ids.append(draft_token_id)
        sampled_target_ids.append(target_token_id)
        draft_logprobs.append(float(log_probs[draft_token_id].item()))

    return {
        "source": "draft",
        "prefix_ids": token_ids[:prefix_len].tolist(),
        "sampled_ids": sampled_target_ids,
        "draft_sampled_token_ids": sampled_draft_ids,
        "draft_logprobs": draft_logprobs,
        "anchor_position": anchor_pos,
        "block_size": block_size,
    }


def _build_scoring_prompt(
    input_ids: list[int],
    prefix_tokens: int,
    score_tokens: int,
) -> tuple[list[int], list[int]]:
    if len(input_ids) < 2:
        raise ValueError("Need at least two tokens for prompt scoring")
    score_tokens = min(score_tokens, len(input_ids) - 1)
    prefix_tokens = min(max(prefix_tokens, 1), len(input_ids) - score_tokens)
    return (
        input_ids[:prefix_tokens],
        input_ids[prefix_tokens : prefix_tokens + score_tokens],
    )


def _score_prompt_logprobs(
    *,
    client,
    model: str,
    input_ids: list[int],
    draft_model,
    item: dict[str, Any],
    hidden_artifacts: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from speculators.data_generation.vllm_client import (  # noqa: PLC0415
        score_sampled_tokens,
    )

    if draft_model is not None:
        if hidden_artifacts is None:
            return {
                "ok": None,
                "status": "not_run_hidden_unusable_for_draft_sampling",
            }
        sample = _sample_from_draft(
            draft_model=draft_model,
            item=item,
            hidden_artifacts=hidden_artifacts,
            args=args,
        )
        prefix_ids = sample["prefix_ids"]
        sampled_ids = sample["sampled_ids"]
    else:
        prefix_ids, sampled_ids = _build_scoring_prompt(
            input_ids,
            prefix_tokens=args.score_prefix_tokens,
            score_tokens=args.score_tokens,
        )
        sample = {"source": "dataset", "prefix_ids": prefix_ids, "sampled_ids": sampled_ids}

    scored = score_sampled_tokens(
        client=client,
        model=model,
        prefix_token_ids=prefix_ids,
        sampled_token_ids=sampled_ids,
        prompt_logprobs=args.prompt_logprobs,
        timeout=args.request_timeout,
        cleanup_hidden_states=not args.keep_generated,
        hidden_states_file_timeout=args.file_timeout,
    )

    return {
        "ok": True,
        "score_prompt_len": len(scored["prompt_token_ids"]),
        "prefix_len": len(prefix_ids),
        "sampled_len": len(sampled_ids),
        "sample": {
            key: value
            for key, value in sample.items()
            if key not in ("prefix_ids", "sampled_ids")
        },
        "sampled_token_ids": sampled_ids,
        "score_positions": scored["score_positions"],
        "target_token_logprobs": scored["token_logprobs"],
        "ignored_hidden_states_path": scored["hidden_states_path"],
        "ignored_hidden_states_deleted": scored["hidden_states_deleted"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a two-sample smoke test: hidden-state extraction may skip, "
            "prompt-logprob scoring ignores hidden states."
        )
    )
    parser.add_argument("--data-path", required=True, help="Preprocessed Arrow dataset")
    parser.add_argument("--vllm-endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--indices", default=None)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--file-timeout", type=float, default=30.0)
    parser.add_argument("--score-prefix-tokens", type=int, default=128)
    parser.add_argument("--score-tokens", type=int, default=8)
    parser.add_argument("--prompt-logprobs", type=int, default=1)
    parser.add_argument(
        "--draft-checkpoint",
        default=None,
        help=(
            "Optional speculator checkpoint. If provided, sampled tokens come from "
            "the draft model instead of dataset continuation tokens."
        ),
    )
    parser.add_argument("--draft-device", default="auto")
    parser.add_argument("--draft-dtype", default="bfloat16")
    parser.add_argument(
        "--draft-attn-impl",
        default="sdpa",
        help="Attention implementation used when running the draft model.",
    )
    parser.add_argument(
        "--draft-temperature",
        type=float,
        default=0.0,
        help="<=0 uses greedy decoding for the draft smoke test.",
    )
    parser.add_argument(
        "--score-only-usable-hidden",
        action="store_true",
        help="Skip scoring when hidden-state extraction skipped the sample.",
    )
    parser.add_argument(
        "--keep-generated",
        action="store_true",
        help="Keep hidden-state files generated during stage 1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import openai  # noqa: PLC0415
    from datasets import load_from_disk  # noqa: PLC0415

    dataset = load_from_disk(args.data_path)
    indices = _choose_indices(args, len(dataset))
    draft_model = _load_draft_model(args)
    client = openai.OpenAI(
        base_url=args.vllm_endpoint,
        api_key="EMPTY",
        max_retries=0,
    )
    model_id = client.models.list().data[0].id
    if args.model is not None and args.model != model_id:
        raise ValueError(
            f"--model {args.model!r} does not match vLLM model id {model_id!r}"
        )

    print(
        f"dataset={args.data_path} | len={len(dataset)} | "
        f"endpoint={args.vllm_endpoint} | model={model_id} | indices={indices}"
    )

    failures = 0
    for index in indices:
        item = dataset[index]
        expected_ids = item["input_ids"].tolist()
        print(f"[sample {index}] start | tokens={len(expected_ids)}")

        sample_result: dict[str, Any] = {"sample": index}
        try:
            hidden_result, hidden_artifacts = _hidden_extract_stage(
                client=client,
                model=model_id,
                item=item,
                args=args,
            )
            sample_result["hidden_extract"] = hidden_result

            if args.score_only_usable_hidden and not hidden_result["ok"]:
                sample_result["scoring"] = {
                    "ok": None,
                    "status": "not_run_hidden_skipped",
                }
            else:
                sample_result["scoring"] = _score_prompt_logprobs(
                    client=client,
                    model=model_id,
                    input_ids=expected_ids,
                    draft_model=draft_model,
                    item=item,
                    hidden_artifacts=hidden_artifacts,
                    args=args,
                )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            sample_result["error"] = f"{type(exc).__name__}: {exc}"

        print(json.dumps(sample_result, indent=2, ensure_ascii=False))
        scoring = sample_result.get("scoring")
        if isinstance(scoring, dict) and scoring.get("ok") is False:
            failures += 1

    if failures:
        raise SystemExit(f"{failures}/{len(indices)} sample(s) failed smoke test")


if __name__ == "__main__":
    main()
