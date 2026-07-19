#!/usr/bin/env python3
"""Check DSpark dataloader hidden generation binding for one dataset item.

This script intentionally avoids multipack batching and collate. It focuses on
dataset index 45760 and answers two questions:

1. Does an existing generated safetensors file contain the same token_ids as the
   dataset item?
2. Does a single call through ArrowDataset's original generate helper produce
   the same hidden states as direct vLLM requests for the same prompt?
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check single-item DSpark hidden generation binding."
    )
    parser.add_argument("--model-path", default="/models/Qwen3-4B")
    parser.add_argument("--data-path", default="/data/open_perfectblend_qwen3_4b_100k")
    parser.add_argument("--hidden-states-path", default=None)
    parser.add_argument("--dataset-index", type=int, default=45760)
    parser.add_argument(
        "--generated-path",
        default="",
        help="Optional existing cmpl-*.safetensors file to inspect first.",
    )
    parser.add_argument("--vllm-endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--device", default="npu:15")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--hidden-states-dtype", default="bfloat16")
    parser.add_argument("--target-layer-ids", type=int, nargs="+", default=[1, 9, 17, 25, 33])
    parser.add_argument("--final-layer-id", type=int, default=None)
    parser.add_argument("--total-seq-len", type=int, default=3072)
    parser.add_argument("--local-start", type=int, default=67)
    parser.add_argument("--gt-len", type=int, default=7)
    parser.add_argument("--prompt-logprobs", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--hidden-file-timeout", type=float, default=30.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--keep-hidden-files", action="store_true")
    return parser


def _fmt(x: float) -> str:
    if math.isnan(x) or math.isinf(x):
        return str(x)
    if x == 0 or (1e-3 <= abs(x) < 1e4):
        return f"{x:.6f}"
    return f"{x:.6e}"


def _resolve_dtype(torch: Any, value: str) -> Any:
    if value == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def _diff_stats(a: Any, b: Any) -> tuple[float, float]:
    diff = (a.detach().float().cpu() - b.detach().float().cpu()).abs()
    return float(diff.max().item()), float(diff.mean().item())


def _max_abs_delta(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b, strict=True)) if a else float("nan")


def _load_raw_item(data_path: str, dataset_index: int) -> dict[str, Any]:
    from datasets import load_from_disk

    dataset = load_from_disk(data_path)
    item = dataset[int(dataset_index)]
    input_ids = item["input_ids"]
    if hasattr(input_ids, "tolist"):
        token_ids = [int(x) for x in input_ids.tolist()]
    else:
        token_ids = [int(x) for x in input_ids]
    return {"item": item, "token_ids": token_ids}


def _first_mismatch_positions(a: Any, b: Any, limit: int = 20) -> list[int]:
    mismatch = (a.detach().cpu() != b.detach().cpu()).nonzero(as_tuple=False).flatten()
    return [int(x) for x in mismatch[:limit].tolist()]


def _inspect_generated_file(path_value: str, raw_ids: Any) -> dict[str, Any] | None:
    if not path_value:
        print("TRACE generated_file_inspect skipped=no_generated_path")
        return None

    path = Path(path_value)
    print("TRACE generated_file_inspect")
    print(f"  path={path}")
    print(f"  exists={path.exists()}")
    if not path.exists():
        return None

    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        saved_ids = f.get_tensor("token_ids")
        saved_hidden = f.get_tensor("hidden_states")

    token_ids_match = bool(saved_ids.equal(raw_ids.cpu()))
    print(f"  token_ids_match_raw_item={token_ids_match}")
    print(f"  first_mismatch_positions={_first_mismatch_positions(saved_ids, raw_ids)}")
    print(f"  saved_token_ids_shape={tuple(saved_ids.shape)}")
    print(f"  saved_hidden_shape={tuple(saved_hidden.shape)}")
    return {"token_ids": saved_ids, "hidden": saved_hidden}


def _wait_for_hidden_file(path_value: str, timeout: float) -> None:
    from speculators.data_generation.vllm_client import wait_for_lock

    path = Path(path_value)
    lock_path = Path(path_value + ".lock")
    deadline = time.monotonic() + timeout
    while lock_path.exists() or not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for hidden states file: {path}")
        if lock_path.exists():
            wait_for_lock(str(lock_path), timeout=max(deadline - time.monotonic(), 0.1))
            continue
        time.sleep(0.05)


def _delete_hidden_file(path_value: str | None) -> None:
    if path_value is None:
        return
    path = Path(path_value)
    path.unlink(missing_ok=True)
    Path(str(path) + ".lock").unlink(missing_ok=True)


def _make_openai_client(endpoint: str) -> tuple[Any, str]:
    import openai

    client = openai.OpenAI(base_url=endpoint, api_key="EMPTY", max_retries=0)
    model_id = client.models.list().data[0].id
    return client, model_id


def _request_direct_hidden_only(
    *,
    client: Any,
    model_id: str,
    prompt: list[int],
    request_timeout: float,
    hidden_file_timeout: float,
) -> dict[str, Any]:
    from safetensors.torch import load_file
    from speculators.data_generation.vllm_client import (
        _kv_hidden_states_path,
        _prompt_token_ids,
    )

    response = client.completions.create(
        model=model_id,
        prompt=prompt,
        max_tokens=1,
        extra_body={"return_token_ids": True},
        timeout=request_timeout,
    )
    prompt_ids = _prompt_token_ids(response)
    hidden_path = _kv_hidden_states_path(response)
    if prompt_ids is None:
        raise RuntimeError("direct hidden-only response missing prompt_token_ids")
    if hidden_path is None:
        raise RuntimeError("direct hidden-only response missing hidden_states_path")
    _wait_for_hidden_file(hidden_path, hidden_file_timeout)
    loaded = load_file(hidden_path)
    return {
        "prompt_ids": list(prompt_ids),
        "hidden_path": hidden_path,
        "file_token_ids": loaded["token_ids"].detach().cpu().clone(),
        "hidden": loaded["hidden_states"].detach().cpu().clone(),
    }


def _request_direct_scored(
    *,
    client: Any,
    model_id: str,
    prompt: list[int],
    score_positions: list[int],
    prompt_logprobs: int,
    request_timeout: float,
    hidden_file_timeout: float,
) -> dict[str, Any]:
    from safetensors.torch import load_file
    from speculators.data_generation.vllm_client import (
        _extract_token_logprob,
        _kv_hidden_states_path,
        _prompt_logprobs,
        _prompt_token_ids,
    )

    response = client.completions.create(
        model=model_id,
        prompt=prompt,
        max_tokens=1,
        extra_body={
            "return_token_ids": True,
            "prompt_logprobs": prompt_logprobs,
        },
        timeout=request_timeout,
    )
    prompt_ids = _prompt_token_ids(response)
    response_prompt_logprobs = _prompt_logprobs(response)
    hidden_path = _kv_hidden_states_path(response)
    if prompt_ids is None:
        raise RuntimeError("direct scored response missing prompt_token_ids")
    if response_prompt_logprobs is None:
        raise RuntimeError("direct scored response missing prompt_logprobs")
    if hidden_path is None:
        raise RuntimeError("direct scored response missing hidden_states_path")
    _wait_for_hidden_file(hidden_path, hidden_file_timeout)
    loaded = load_file(hidden_path)
    token_logprobs = [
        _extract_token_logprob(response_prompt_logprobs, pos, prompt[pos])
        for pos in score_positions
    ]
    return {
        "prompt_ids": list(prompt_ids),
        "hidden_path": hidden_path,
        "file_token_ids": loaded["token_ids"].detach().cpu().clone(),
        "hidden": loaded["hidden_states"].detach().cpu().clone(),
        "token_logprobs": token_logprobs,
    }


def _helper_generate_once(args: argparse.Namespace, torch: Any) -> dict[str, Any]:
    from speculators.train.data import ArrowDataset

    dataset = ArrowDataset(
        datapath=args.data_path,
        max_len=args.total_seq_len,
        hidden_states_path=args.hidden_states_path,
        vllm_endpoint=args.vllm_endpoint,
        on_missing="generate",
        on_generate="delete",
        split_ratio=1.0,
        transform=None,
        hidden_states_dtype=getattr(torch, args.hidden_states_dtype),
        model=args.model_path,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
    )
    loaded = dataset._maybe_generate_hs(args.dataset_index)  # noqa: SLF001
    if loaded is None:
        raise RuntimeError("ArrowDataset._maybe_generate_hs returned None")
    return {
        "source": loaded.get("_hidden_state_source"),
        "token_ids": loaded["token_ids"].detach().cpu().clone(),
        "hidden": loaded["hidden_states"].detach().cpu().clone(),
    }


def _load_hf_model(args: argparse.Namespace, torch: Any) -> Any:
    from transformers import AutoModelForCausalLM

    dtype = _resolve_dtype(torch, args.dtype)
    kwargs = {
        "trust_remote_code": args.trust_remote_code,
    }
    if dtype != "auto":
        kwargs["dtype"] = dtype
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    except TypeError:
        if dtype != "auto":
            kwargs.pop("dtype", None)
            kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    model.to(torch.device(args.device))
    model.eval()
    return model


def _project_full_lm_head(
    *,
    torch: Any,
    hf_model: Any,
    final_hidden: Any,
    prompt: list[int],
    score_positions: list[int],
) -> list[float]:
    device = next(hf_model.parameters()).device
    hidden = hf_model.model.norm(final_hidden.to(device=device))
    logits = hf_model.lm_head(hidden).float()
    hidden_positions = torch.tensor(
        [pos - 1 for pos in score_positions],
        dtype=torch.long,
        device=device,
    )
    target_ids = torch.tensor(
        [prompt[pos] for pos in score_positions],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        logprobs = torch.log_softmax(logits[hidden_positions], dim=-1)
        gathered = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return [float(x) for x in gathered.detach().cpu().tolist()]


def _print_run_header(label: str, run: dict[str, Any], raw_ids: Any) -> None:
    token_ids = run["file_token_ids"] if "file_token_ids" in run else run["token_ids"]
    if not hasattr(token_ids, "detach"):
        import torch

        token_ids = torch.tensor(token_ids, dtype=torch.long)
    print(f"TRACE {label}")
    if "source" in run:
        print(f"  source={run['source']}")
    if "hidden_path" in run:
        print(f"  hidden_path={run['hidden_path']}")
    print(f"  token_ids_match_raw_item={bool(token_ids.equal(raw_ids.cpu()))}")
    print(f"  first_mismatch_positions={_first_mismatch_positions(token_ids, raw_ids)}")
    print(f"  hidden_shape={tuple(run['hidden'].shape)}")


def _print_hidden_compares(
    *,
    helper: dict[str, Any],
    direct_hidden: dict[str, Any],
    direct_scored: dict[str, Any],
    local_start: int,
    gt_len: int,
) -> None:
    local_slice = slice(local_start, local_start + gt_len)
    final_slot = helper["hidden"].shape[1] - 1
    print("TRACE hidden_compare")
    for label, a, b in (
        ("helper_vs_direct_hidden_all_slots", helper["hidden"][local_slice], direct_hidden["hidden"][local_slice]),
        ("helper_vs_direct_hidden_final", helper["hidden"][local_slice, final_slot], direct_hidden["hidden"][local_slice, final_slot]),
        ("helper_vs_direct_scored_final", helper["hidden"][local_slice, final_slot], direct_scored["hidden"][local_slice, final_slot]),
        ("direct_hidden_vs_direct_scored_final", direct_hidden["hidden"][local_slice, final_slot], direct_scored["hidden"][local_slice, final_slot]),
    ):
        max_abs, mean_abs = _diff_stats(a, b)
        print(f"  {label}_max_abs={_fmt(max_abs)}")
        print(f"  {label}_mean_abs={_fmt(mean_abs)}")


def main() -> None:
    args = _parser().parse_args()

    import torch
    from transformers import AutoConfig

    raw = _load_raw_item(args.data_path, args.dataset_index)
    raw_ids = torch.tensor(raw["token_ids"], dtype=torch.long)
    score_positions = list(
        range(args.local_start + 1, args.local_start + args.gt_len + 1)
    )

    config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if hasattr(config, "text_config"):
        config = config.text_config
    final_layer_id = (
        int(args.final_layer_id)
        if args.final_layer_id is not None
        else int(config.num_hidden_layers)
    )
    connector_layer_ids = list(args.target_layer_ids)
    if final_layer_id not in connector_layer_ids:
        connector_layer_ids.append(final_layer_id)

    print("TRACE config")
    print(f"  repo={ROOT}")
    print(f"  model_path={args.model_path}")
    print(f"  data_path={args.data_path}")
    print(f"  dataset_index={args.dataset_index}")
    print(f"  item_len={len(raw_ids)}")
    print(f"  generated_path={args.generated_path or '<none>'}")
    print(f"  vllm_endpoint={args.vllm_endpoint}")
    print(f"  device={args.device}")
    print(f"  dtype={args.dtype}")
    print(f"  target_layer_ids={args.target_layer_ids}")
    print(f"  final_layer_id={final_layer_id}")
    print(f"  assumed_connector_layer_ids={connector_layer_ids}")
    print(f"  local_hidden_window={args.local_start}:{args.local_start + args.gt_len}")
    print(f"  local_score_positions={score_positions}")

    _inspect_generated_file(args.generated_path, raw_ids)

    client, model_id = _make_openai_client(args.vllm_endpoint)
    print("TRACE vllm")
    print(f"  model_id={model_id}")

    helper = _helper_generate_once(args, torch)
    direct_hidden = _request_direct_hidden_only(
        client=client,
        model_id=model_id,
        prompt=raw["token_ids"],
        request_timeout=args.request_timeout,
        hidden_file_timeout=args.hidden_file_timeout,
    )
    direct_scored = _request_direct_scored(
        client=client,
        model_id=model_id,
        prompt=raw["token_ids"],
        score_positions=score_positions,
        prompt_logprobs=args.prompt_logprobs,
        request_timeout=args.request_timeout,
        hidden_file_timeout=args.hidden_file_timeout,
    )

    _print_run_header("helper_once", helper, raw_ids)
    _print_run_header("direct_hidden_only", direct_hidden, raw_ids)
    _print_run_header("direct_scored", direct_scored, raw_ids)
    print(f"  prompt_logprobs={[_fmt(x) for x in direct_scored['token_logprobs']]}")

    _print_hidden_compares(
        helper=helper,
        direct_hidden=direct_hidden,
        direct_scored=direct_scored,
        local_start=args.local_start,
        gt_len=args.gt_len,
    )

    hf_model = _load_hf_model(args, torch)
    final_slot = direct_scored["hidden"].shape[1] - 1
    helper_plog = _project_full_lm_head(
        torch=torch,
        hf_model=hf_model,
        final_hidden=helper["hidden"][:, final_slot],
        prompt=raw["token_ids"],
        score_positions=score_positions,
    )
    direct_hidden_plog = _project_full_lm_head(
        torch=torch,
        hf_model=hf_model,
        final_hidden=direct_hidden["hidden"][:, final_slot],
        prompt=raw["token_ids"],
        score_positions=score_positions,
    )
    direct_scored_plog = _project_full_lm_head(
        torch=torch,
        hf_model=hf_model,
        final_hidden=direct_scored["hidden"][:, final_slot],
        prompt=raw["token_ids"],
        score_positions=score_positions,
    )

    print("TRACE projection_compare")
    print("  local_score_pos,target_id,helper,direct_hidden,direct_scored,D_prompt")
    for i, pos in enumerate(score_positions):
        print(
            f"  {pos},"
            f"{raw['token_ids'][pos]},"
            f"{_fmt(helper_plog[i])},"
            f"{_fmt(direct_hidden_plog[i])},"
            f"{_fmt(direct_scored_plog[i])},"
            f"{_fmt(direct_scored['token_logprobs'][i])}"
        )
    print(
        "  direct_scored_projection_vs_prompt_max_abs="
        f"{_fmt(_max_abs_delta(direct_scored_plog, direct_scored['token_logprobs']))}"
    )
    print(
        "  helper_projection_vs_direct_scored_projection_max_abs="
        f"{_fmt(_max_abs_delta(helper_plog, direct_scored_plog))}"
    )
    print(
        "  direct_hidden_projection_vs_direct_scored_projection_max_abs="
        f"{_fmt(_max_abs_delta(direct_hidden_plog, direct_scored_plog))}"
    )

    helper_vs_direct_hidden, _ = _diff_stats(
        helper["hidden"][
            args.local_start : args.local_start + args.gt_len,
            final_slot,
        ],
        direct_hidden["hidden"][
            args.local_start : args.local_start + args.gt_len,
            final_slot,
        ],
    )
    direct_scored_vs_prompt = _max_abs_delta(
        direct_scored_plog,
        direct_scored["token_logprobs"],
    )
    print("TRACE interpretation")
    if helper_vs_direct_hidden <= 1e-2 and direct_scored_vs_prompt <= 0.5:
        print("  conclusion=single helper matches direct hidden-only; batch/concurrent scheduling remains implicated")
    elif helper_vs_direct_hidden > 1e-2:
        print("  conclusion=single helper differs from direct hidden-only; inspect request/file binding or helper request path")
    elif direct_scored_vs_prompt > 0.5:
        print("  conclusion=direct scored hidden projection does not match prompt_logprobs; target forward/context issue is implicated")
    else:
        print("  conclusion=mixed; inspect TRACE hidden_compare and TRACE projection_compare")

    if not args.keep_hidden_files:
        _delete_hidden_file(direct_hidden.get("hidden_path"))
        _delete_hidden_file(direct_scored.get("hidden_path"))


if __name__ == "__main__":
    main()
