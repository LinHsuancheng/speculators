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


def _get_prompt_logprobs(response: Any) -> Any:
    choices = getattr(response, "choices", None)
    if choices:
        prompt_logprobs = getattr(choices[0], "prompt_logprobs", None)
        if prompt_logprobs is not None:
            return prompt_logprobs
    prompt_logprobs = getattr(response, "prompt_logprobs", None)
    if prompt_logprobs is not None:
        return prompt_logprobs
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if dumped.get("choices"):
            return dumped["choices"][0].get("prompt_logprobs")
        return dumped.get("prompt_logprobs")
    return None


def _summarize_prompt_logprobs(prompt_logprobs: Any, score_positions: range) -> dict:
    if prompt_logprobs is None:
        return {"present": False}
    try:
        total_len = len(prompt_logprobs)
    except TypeError:
        return {
            "present": True,
            "type": type(prompt_logprobs).__name__,
            "repr": repr(prompt_logprobs)[:300],
        }

    non_null = 0
    scored_non_null = 0
    first_non_null = None
    for idx, item in enumerate(prompt_logprobs):
        if item is None:
            continue
        non_null += 1
        if idx in score_positions:
            scored_non_null += 1
        if first_non_null is None:
            first_non_null = repr(item)[:300]
    return {
        "present": True,
        "len": total_len,
        "non_null": non_null,
        "scored_positions": [score_positions.start, score_positions.stop],
        "scored_non_null": scored_non_null,
        "first_non_null": first_non_null,
    }


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
) -> dict[str, Any]:
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
        }

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
    }


def _build_scoring_prompt(
    input_ids: list[int],
    prefix_tokens: int,
    score_tokens: int,
) -> tuple[list[int], range]:
    if len(input_ids) < 2:
        raise ValueError("Need at least two tokens for prompt scoring")
    score_tokens = min(score_tokens, len(input_ids) - 1)
    prefix_tokens = min(max(prefix_tokens, 1), len(input_ids) - score_tokens)
    score_ids = input_ids[: prefix_tokens + score_tokens]
    return score_ids, range(prefix_tokens, len(score_ids))


def _score_prompt_logprobs(
    *,
    client,
    model: str,
    input_ids: list[int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    score_ids, score_positions = _build_scoring_prompt(
        input_ids,
        prefix_tokens=args.score_prefix_tokens,
        score_tokens=args.score_tokens,
    )
    extra_body = {
        "return_token_ids": True,
        "prompt_logprobs": args.prompt_logprobs,
    }

    max_tokens_used = args.score_max_tokens
    try:
        response = client.completions.create(
            model=model,
            prompt=score_ids,
            max_tokens=max_tokens_used,
            extra_body=extra_body,
            timeout=args.request_timeout,
        )
    except Exception as exc:
        if not args.retry_score_max_tokens_one or max_tokens_used != 0:
            raise
        max_tokens_used = 1
        response = client.completions.create(
            model=model,
            prompt=score_ids,
            max_tokens=max_tokens_used,
            extra_body=extra_body,
            timeout=args.request_timeout,
        )
        retry_note = f"max_tokens=0 failed with {type(exc).__name__}: {exc}"
    else:
        retry_note = None

    response_prompt_ids = _get_prompt_token_ids(response)
    prompt_logprobs = _get_prompt_logprobs(response)
    summary = _summarize_prompt_logprobs(prompt_logprobs, score_positions)
    ignored_hs_path = _get_kv_hidden_path(response)

    return {
        "ok": bool(summary.get("present")),
        "score_prompt_len": len(score_ids),
        "score_positions": [score_positions.start, score_positions.stop],
        "max_tokens_used": max_tokens_used,
        "retry_note": retry_note,
        "prompt_token_ids_len": None
        if response_prompt_ids is None
        else len(response_prompt_ids),
        "prompt_token_ids_match": response_prompt_ids == score_ids,
        "prompt_logprobs": summary,
        "ignored_hidden_states_path": ignored_hs_path,
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
        "--score-max-tokens",
        type=int,
        default=0,
        help="Use 0 for pure prompt scoring; retry with 1 by default if unsupported.",
    )
    parser.add_argument(
        "--no-retry-score-max-tokens-one",
        dest="retry_score_max_tokens_one",
        action="store_false",
        help="Do not retry scoring with max_tokens=1 if max_tokens=0 fails.",
    )
    parser.set_defaults(retry_score_max_tokens_one=True)
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
            hidden_result = _hidden_extract_stage(
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
