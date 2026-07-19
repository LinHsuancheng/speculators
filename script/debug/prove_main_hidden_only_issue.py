#!/usr/bin/env python3
"""Prove the main branch hidden-only generation path is inconsistent.

This script does not modify training code. It uses the same helper that main
training calls for on-missing generation, then compares that hidden file with:

1. a direct hidden-only vLLM request
2. a direct prompt_logprobs-scored vLLM request

The goal is to show that the main branch's hidden-only generation path can
produce hidden states that do not match the scored request for the same prompt.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove main branch hidden-only generation is inconsistent."
    )
    parser.add_argument("--verifier-name-or-path", default="/models/Qwen3-4B")
    parser.add_argument("--data-path", default="/data/open_perfectblend_qwen3_4b_100k")
    parser.add_argument("--hidden-states-path", default=None)
    parser.add_argument("--vllm-endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--dataset-index", type=int, default=45760)
    parser.add_argument("--hidden-states-dtype", default="bfloat16")
    parser.add_argument("--local-start", type=int, default=67)
    parser.add_argument("--gt-len", type=int, default=7)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--hidden-file-timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--device", default="npu:15")
    parser.add_argument("--prompt-logprobs", type=int, default=1)
    parser.add_argument("--position-diff-print", choices=["none", "summary", "all"], default="summary")
    parser.add_argument("--position-equal-tol", type=float, default=1e-6)
    parser.add_argument("--position-diff-chunk-size", type=int, default=128)
    return parser.parse_args()


def fmt(value: float) -> str:
    if math.isnan(value) or math.isinf(value):
        return str(value)
    if value == 0 or (1e-3 <= abs(value) < 1e4):
        return f"{value:.6f}"
    return f"{value:.6e}"


def dtype_from_name(torch: Any, name: str) -> Any:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def load_train_split(data_path: str) -> Any:
    from datasets import load_from_disk

    dataset = load_from_disk(data_path)
    return dataset.select(range(int(len(dataset) * 0.9)))


def token_ids_for(dataset: Any, index: int) -> list[int]:
    value = dataset[int(index)]["input_ids"]
    if hasattr(value, "tolist"):
        return [int(x) for x in value.tolist()]
    return [int(x) for x in value]


def wait_for_hidden_file(path_value: str, timeout: float) -> None:
    from speculators.data_generation.vllm_client import wait_for_lock

    path = Path(path_value)
    lock = Path(path_value + ".lock")
    deadline = time.monotonic() + timeout
    while lock.exists() or not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for hidden-state file: {path}")
        if lock.exists():
            wait_for_lock(str(lock), timeout=max(deadline - time.monotonic(), 0.1))
        else:
            time.sleep(0.05)


def delete_hidden_file(path_value: str | None) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.unlink(missing_ok=True)
    Path(str(path) + ".lock").unlink(missing_ok=True)


def _prompt_token_ids(response: Any) -> list[int] | None:
    if hasattr(response, "choices"):
        prompt_token_ids = getattr(response.choices[0], "prompt_token_ids", None)
    else:
        prompt_token_ids = getattr(response, "prompt_token_ids", None)
    return None if prompt_token_ids is None else list(prompt_token_ids)


def _kv_hidden_states_path(response: Any) -> str | None:
    kv_transfer_params = getattr(response, "kv_transfer_params", None)
    if isinstance(kv_transfer_params, dict):
        return kv_transfer_params.get("hidden_states_path")
    return None


def _prompt_logprobs(response: Any) -> Any:
    if hasattr(response, "choices"):
        prompt_logprobs = getattr(response.choices[0], "prompt_logprobs", None)
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


def _logprob_value(value: Any) -> float:
    if isinstance(value, dict):
        return float(value["logprob"])
    return float(getattr(value, "logprob"))


def _extract_token_logprob(prompt_logprobs: Any, position: int, token_id: int) -> float:
    position_logprobs = prompt_logprobs[position]
    if position_logprobs is None:
        raise RuntimeError(f"missing prompt logprobs at position {position}")
    if isinstance(position_logprobs, dict):
        for key in (token_id, str(token_id)):
            if key in position_logprobs:
                return _logprob_value(position_logprobs[key])
    if hasattr(position_logprobs, "get"):
        for key in (token_id, str(token_id)):
            value = position_logprobs.get(key)
            if value is not None:
                return _logprob_value(value)
    raise RuntimeError(
        f"unsupported prompt logprobs entry at position {position}: "
        f"{type(position_logprobs).__name__}"
    )


def request_hidden_only(
    *,
    endpoint: str,
    model_id: str,
    token_ids: list[int],
    request_timeout: float,
    hidden_file_timeout: float,
) -> dict[str, Any]:
    import openai
    from safetensors.torch import load_file

    client = openai.OpenAI(base_url=endpoint, api_key="EMPTY", max_retries=0)
    response = client.completions.create(
        model=model_id,
        prompt=token_ids,
        max_tokens=1,
        extra_body={"return_token_ids": True},
        timeout=request_timeout,
    )
    response_ids = _prompt_token_ids(response)
    hidden_path = _kv_hidden_states_path(response)
    if response_ids is None:
        raise RuntimeError("hidden-only response missing prompt_token_ids")
    if hidden_path is None:
        raise RuntimeError("hidden-only response missing hidden_states_path")
    wait_for_hidden_file(hidden_path, hidden_file_timeout)
    loaded = load_file(hidden_path)
    return {
        "prompt_token_ids": [int(x) for x in response_ids],
        "hidden_path": hidden_path,
        "hidden": loaded["hidden_states"].detach().cpu().clone(),
        "file_token_ids": loaded["token_ids"].detach().cpu().clone(),
    }


def request_scored(
    *,
    endpoint: str,
    model_id: str,
    token_ids: list[int],
    score_positions: list[int],
    prompt_logprobs: int,
    request_timeout: float,
    hidden_file_timeout: float,
) -> dict[str, Any]:
    import openai
    from safetensors.torch import load_file

    client = openai.OpenAI(base_url=endpoint, api_key="EMPTY", max_retries=0)
    response = client.completions.create(
        model=model_id,
        prompt=token_ids,
        max_tokens=1,
        extra_body={"return_token_ids": True, "prompt_logprobs": prompt_logprobs},
        timeout=request_timeout,
    )
    response_ids = _prompt_token_ids(response)
    prompt_logprob_obj = _prompt_logprobs(response)
    hidden_path = _kv_hidden_states_path(response)
    if response_ids is None:
        raise RuntimeError("scored response missing prompt_token_ids")
    if prompt_logprob_obj is None:
        raise RuntimeError("scored response missing prompt_logprobs")
    if hidden_path is None:
        raise RuntimeError("scored response missing hidden_states_path")
    wait_for_hidden_file(hidden_path, hidden_file_timeout)
    loaded = load_file(hidden_path)
    return {
        "prompt_token_ids": [int(x) for x in response_ids],
        "hidden_path": hidden_path,
        "hidden": loaded["hidden_states"].detach().cpu().clone(),
        "file_token_ids": loaded["token_ids"].detach().cpu().clone(),
        "prompt_logprobs": [
            _extract_token_logprob(prompt_logprob_obj, pos, token_ids[pos])
            for pos in score_positions
        ],
    }


class FullVerifierHead:
    def __init__(self, model_path: str, device: Any, dtype: Any) -> None:
        import torch
        from transformers import AutoConfig
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

        from speculators.utils.loading import load_model_layers

        config = AutoConfig.from_pretrained(model_path)
        if hasattr(config, "text_config"):
            config = config.text_config
        weights = load_model_layers(
            ["embed_tokens.weight", "lm_head.weight", "model.norm.weight"],
            model_path,
        )
        lm_head_source = (
            "lm_head.weight" if "lm_head.weight" in weights else "embed_tokens.weight"
        )
        self.lm_head_weight = weights[lm_head_source].to(device=device, dtype=dtype)
        self.lm_head_source = lm_head_source
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(
            device=device,
            dtype=dtype,
        )
        self.norm.load_state_dict(
            {"weight": weights["model.norm.weight"].to(device=device, dtype=dtype)}
        )
        self.device = device
        self.dtype = dtype
        self.torch = torch

    def project(self, hidden_last: Any, prompt: list[int], score_positions: list[int]) -> list[float]:
        torch = self.torch
        hidden_positions = torch.tensor(
            [pos - 1 for pos in score_positions],
            dtype=torch.long,
            device=self.device,
        )
        target_ids = torch.tensor(
            [prompt[pos] for pos in score_positions],
            dtype=torch.long,
            device=self.device,
        )
        with torch.no_grad():
            selected = hidden_last.to(device=self.device, dtype=self.dtype).index_select(
                0,
                hidden_positions,
            )
            normed = self.norm(selected)
            logits = torch.nn.functional.linear(normed, self.lm_head_weight).float()
            logprobs = torch.log_softmax(logits, dim=-1)
            values = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        return [float(x) for x in values.detach().cpu().tolist()]


def max_abs_delta(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return math.nan
    if len(left) != len(right):
        return math.nan
    if not left:
        return 0.0
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def position_scan(
    *,
    left: Any,
    right: Any,
    local_start: int,
    local_end: int,
    equal_tol: float,
    chunk_size: int,
    print_mode: str,
) -> None:
    import torch

    if tuple(left.shape) != tuple(right.shape):
        print("TRACE position_scan")
        print(f"  shape_mismatch left={tuple(left.shape)} right={tuple(right.shape)}")
        return

    left_final = left[:, -1].float()
    right_final = right[:, -1].float()
    delta = (left_final - right_final).abs()
    mean_by_pos = delta.mean(dim=-1)
    max_by_pos = delta.max(dim=-1).values
    equal_positions = (mean_by_pos <= equal_tol).nonzero(as_tuple=False).flatten()

    print("TRACE position_scan")
    print(f"  seq_len={int(left.shape[0])}")
    print(f"  final_slot={int(left.shape[1] - 1)}")
    print(f"  equal_tol={equal_tol}")
    print(f"  mean_diff_min={fmt(float(mean_by_pos.min().item()))}")
    print(f"  mean_diff_max={fmt(float(mean_by_pos.max().item()))}")
    print(f"  mean_diff_mean={fmt(float(mean_by_pos.mean().item()))}")
    print(f"  max_diff_max={fmt(float(max_by_pos.max().item()))}")
    print(f"  equal_position_count={int(equal_positions.numel())}")
    if equal_positions.numel() <= 64:
        print(f"  equal_positions={equal_positions.detach().cpu().tolist()}")
    else:
        print(f"  equal_positions_head={equal_positions[:32].detach().cpu().tolist()}")
        print(f"  equal_positions_tail={equal_positions[-32:].detach().cpu().tolist()}")

    print("TRACE chunk_summary")
    print("  start,end,mean_diff_mean,mean_diff_min,mean_diff_max,max_diff_max,equal_count")
    chunk_size = max(int(chunk_size), 1)
    for start in range(0, mean_by_pos.numel(), chunk_size):
        end = min(start + chunk_size, mean_by_pos.numel())
        chunk_mean = mean_by_pos[start:end]
        chunk_max = max_by_pos[start:end]
        chunk_equal = int((chunk_mean <= equal_tol).sum().item())
        print(
            "  "
            f"{start},{end},"
            f"{fmt(float(chunk_mean.mean().item()))},"
            f"{fmt(float(chunk_mean.min().item()))},"
            f"{fmt(float(chunk_mean.max().item()))},"
            f"{fmt(float(chunk_max.max().item()))},"
            f"{chunk_equal}"
        )

    if print_mode != "none":
        print("TRACE position_diff")
        print("  pos,mean_abs,max_abs")
        positions = range(mean_by_pos.numel()) if print_mode == "all" else range(local_start, local_end)
        for pos in positions:
            print(f"  {pos},{fmt(float(mean_by_pos[pos].item()))},{fmt(float(max_by_pos[pos].item()))}")


def main() -> None:
    args = parse_args()

    import torch
    from speculators.data_generation.vllm_client import generate_hidden_states
    from speculators.train.data import build_client_item

    torch.manual_seed(0)
    hidden_dtype = dtype_from_name(torch, args.hidden_states_dtype)
    dataset = load_train_split(args.data_path)
    dataset_item = dataset[int(args.dataset_index)]
    token_ids = token_ids_for(dataset, args.dataset_index)
    score_positions = list(range(args.local_start + 1, args.local_start + args.gt_len + 1))

    client = __import__("openai").OpenAI(
        base_url=args.vllm_endpoint, api_key="EMPTY", max_retries=0
    )
    model_id = client.models.list().data[0].id
    client_item = build_client_item(dataset_item)
    hidden_path = generate_hidden_states(
        client,
        model_id,
        client_item,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
    )
    from safetensors.torch import load_file

    wait_for_hidden_file(hidden_path, args.hidden_file_timeout)
    generated = load_file(hidden_path)
    generated_hidden = generated["hidden_states"].detach().cpu().clone()
    generated_token_ids = generated["token_ids"].detach().cpu().clone()

    hidden_only = request_hidden_only(
        endpoint=args.vllm_endpoint,
        model_id=model_id,
        token_ids=token_ids,
        request_timeout=args.request_timeout,
        hidden_file_timeout=args.hidden_file_timeout,
    )
    scored = request_scored(
        endpoint=args.vllm_endpoint,
        model_id=model_id,
        token_ids=token_ids,
        score_positions=score_positions,
        prompt_logprobs=args.prompt_logprobs,
        request_timeout=args.request_timeout,
        hidden_file_timeout=args.hidden_file_timeout,
    )

    generated_proj = None
    hidden_only_proj = None
    scored_proj = None
    if True:
        head = FullVerifierHead(args.verifier_name_or_path, args.device, hidden_dtype)
        generated_proj = head.project(generated_hidden[:, -1], token_ids, score_positions)
        hidden_only_proj = head.project(hidden_only["hidden"][:, -1], token_ids, score_positions)
        scored_proj = head.project(scored["hidden"][:, -1], token_ids, score_positions)

        print("TRACE full_verifier_head")
        print(f"  device={args.device}")
        print(f"  dtype={hidden_dtype}")
        print(f"  lm_head_source={head.lm_head_source}")
        print(f"  lm_head_shape={tuple(head.lm_head_weight.shape)}")

    print("TRACE config")
    print(f"  repo={ROOT}")
    print(f"  branch=main")
    print(f"  verifier_name_or_path={args.verifier_name_or_path}")
    print(f"  data_path={args.data_path}")
    print(f"  dataset_index={args.dataset_index}")
    print(f"  vllm_endpoint={args.vllm_endpoint}")
    print(f"  hidden_states_dtype={hidden_dtype}")
    print(f"  request_timeout={args.request_timeout}")
    print(f"  hidden_file_timeout={args.hidden_file_timeout}")
    print(f"  prompt_logprobs={args.prompt_logprobs}")
    print(f"  local_hidden_window={args.local_start}:{args.local_start + args.gt_len}")
    print(f"  score_positions={score_positions}")
    print(f"  target_ids={[token_ids[pos] for pos in score_positions]}")

    print("TRACE generated_request")
    print(f"  hidden_path={hidden_path}")
    print(f"  prompt_ids_match={generated_token_ids.tolist() == token_ids}")
    print(f"  file_token_ids_match={bool(torch.equal(generated_token_ids, torch.tensor(token_ids, dtype=torch.long)))}")
    print(f"  hidden_shape={tuple(generated_hidden.shape)}")

    print("TRACE direct_hidden_only")
    print(f"  hidden_path={hidden_only['hidden_path']}")
    print(f"  prompt_ids_match={hidden_only['prompt_token_ids'] == token_ids}")
    print(f"  file_token_ids_match={bool(torch.equal(hidden_only['file_token_ids'], torch.tensor(token_ids, dtype=torch.long)))}")
    print(f"  hidden_shape={tuple(hidden_only['hidden'].shape)}")

    print("TRACE direct_scored")
    print(f"  hidden_path={scored['hidden_path']}")
    print(f"  prompt_ids_match={scored['prompt_token_ids'] == token_ids}")
    print(f"  file_token_ids_match={bool(torch.equal(scored['file_token_ids'], torch.tensor(token_ids, dtype=torch.long)))}")
    print(f"  hidden_shape={tuple(scored['hidden'].shape)}")
    print(f"  prompt_logprobs={[fmt(x) for x in scored['prompt_logprobs']]}")

    print("TRACE hidden_compare")
    print(f"  generated_vs_hidden_only_max_abs={fmt(float((generated_hidden - hidden_only['hidden']).abs().max().item()))}")
    print(f"  generated_vs_hidden_only_mean_abs={fmt(float((generated_hidden - hidden_only['hidden']).abs().mean().item()))}")
    print(f"  generated_vs_scored_max_abs={fmt(float((generated_hidden - scored['hidden']).abs().max().item()))}")
    print(f"  generated_vs_scored_mean_abs={fmt(float((generated_hidden - scored['hidden']).abs().mean().item()))}")
    print(f"  hidden_only_vs_scored_max_abs={fmt(float((hidden_only['hidden'] - scored['hidden']).abs().max().item()))}")
    print(f"  hidden_only_vs_scored_mean_abs={fmt(float((hidden_only['hidden'] - scored['hidden']).abs().mean().item()))}")

    position_scan(
        left=hidden_only["hidden"],
        right=scored["hidden"],
        local_start=args.local_start,
        local_end=args.local_start + args.gt_len,
        equal_tol=args.position_equal_tol,
        chunk_size=args.position_diff_chunk_size,
        print_mode=args.position_diff_print,
    )

    print("TRACE projection_compare")
    print("  pos,target_id,generated,hidden_only,scored")
    for i, pos in enumerate(score_positions):
        target_id = token_ids[pos]
        print(
            "  "
            f"{pos},{target_id},"
            f"{fmt(float(generated_proj[i]))},"
            f"{fmt(float(hidden_only_proj[i]))},"
            f"{fmt(float(scored_proj[i]))}"
        )
    print(
        "  generated_vs_scored_projection_max_abs="
        f"{fmt(max_abs_delta(generated_proj, scored['prompt_logprobs']))}"
    )
    print(
        "  hidden_only_vs_scored_projection_max_abs="
        f"{fmt(max_abs_delta(hidden_only_proj, scored['prompt_logprobs']))}"
    )
    print(
        "  scored_projection_vs_prompt_max_abs="
        f"{fmt(max_abs_delta(scored_proj, scored['prompt_logprobs']))}"
    )
    print(
        "  conclusion="
        + (
            "main_hidden_only_path_is_inconsistent"
            if max_abs_delta(hidden_only_proj, scored["prompt_logprobs"]) > 0.5
            else "no_inconsistency_detected"
        )
    )

    delete_hidden_file(hidden_path)
    delete_hidden_file(hidden_only["hidden_path"])
    delete_hidden_file(scored["hidden_path"])


if __name__ == "__main__":
    main()
