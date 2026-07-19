#!/usr/bin/env python3
"""Direct HF hidden-state ablation for Qwen verifier models.

This script intentionally bypasses vLLM and Speculators training code. It loads
the verifier with Transformers, feeds one token sequence, and prints enough
alignment detail to inspect token ids, token strings, selected hidden states,
and final-hidden projection logprobs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a direct HF hidden-state ablation without vLLM."
    )
    parser.add_argument("--model-path", default="/models/Qwen3-4B")
    parser.add_argument("--data-path", default="/data/open_perfectblend_qwen3_4b_100k")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--text", default="", help="Use tokenizer(text) as input.")
    parser.add_argument(
        "--token-ids",
        default="",
        help="Comma/space separated token ids. Overrides --text and --data-path.",
    )
    parser.add_argument(
        "--target-layer-ids",
        type=int,
        nargs="+",
        default=[1, 9, 17, 25, 33, 36],
        help=(
            "Layer ids to print. HF output_hidden_states index 0 is embeddings; "
            "model layer N is hidden_states[N]."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--positions", type=int, nargs="*", default=[])
    parser.add_argument("--print-token-limit", type=int, default=80)
    parser.add_argument("--print-hidden-limit", type=int, default=8)
    parser.add_argument("--topk", type=int, default=5)
    return parser


def _fmt(x: float) -> str:
    if math.isnan(x) or math.isinf(x):
        return str(x)
    if x == 0 or (1e-3 <= abs(x) < 1e4):
        return f"{x:.6f}"
    return f"{x:.6e}"


def _parse_token_ids(value: str) -> list[int]:
    chunks = value.replace(",", " ").split()
    if not chunks:
        raise ValueError("--token-ids was passed but no ids were parsed")
    return [int(x) for x in chunks]


def _resolve_torch_dtype(torch: Any, value: str) -> Any:
    if value == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


def _resolve_device(torch: Any, device_arg: str | None) -> Any:
    if device_arg:
        return torch.device(device_arg)
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _load_dataset_token_ids(args: argparse.Namespace) -> list[int]:
    from datasets import load_from_disk

    dataset = load_from_disk(args.data_path)
    item = dataset[int(args.sample_index)]
    input_ids = item["input_ids"]
    if hasattr(input_ids, "tolist"):
        token_ids = input_ids.tolist()
    else:
        token_ids = list(input_ids)
    return [int(x) for x in token_ids]


def _load_token_ids(args: argparse.Namespace, tokenizer: Any) -> list[int]:
    if args.token_ids:
        token_ids = _parse_token_ids(args.token_ids)
    elif args.text:
        encoded = tokenizer(args.text, add_special_tokens=False)
        token_ids = [int(x) for x in encoded["input_ids"]]
    else:
        token_ids = _load_dataset_token_ids(args)

    if args.max_seq_len > 0:
        token_ids = token_ids[: args.max_seq_len]
    if len(token_ids) < 2:
        raise ValueError("Need at least two tokens to inspect next-token logprobs")
    return token_ids


def _default_positions(seq_len: int) -> list[int]:
    candidates = [0, 1, 2, seq_len // 4, seq_len // 2, seq_len - 2, seq_len - 1]
    out = []
    for pos in candidates:
        if 0 <= pos < seq_len and pos not in out:
            out.append(pos)
    return out


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
    except Exception:  # noqa: BLE001
        token = "<convert_error>"
    try:
        text = tokenizer.decode([token_id])
    except Exception:  # noqa: BLE001
        text = "<decode_error>"
    return f"{token!r} text={text!r}"


def _print_token_table(
    *,
    tokenizer: Any,
    token_ids: list[int],
    positions: set[int],
    limit: int,
) -> None:
    print("TRACE tokens")
    print(f"  seq_len={len(token_ids)}")
    print(f"  first_ids={token_ids[: min(limit, len(token_ids))]}")
    print("  pos,token_id,next_token_id,selected,token")
    shown = set(range(min(limit, len(token_ids)))) | positions
    for pos in sorted(x for x in shown if 0 <= x < len(token_ids)):
        next_id = token_ids[pos + 1] if pos + 1 < len(token_ids) else None
        selected = "yes" if pos in positions else "no"
        print(
            f"  {pos},{token_ids[pos]},{next_id},{selected},"
            f"{_decode_token(tokenizer, token_ids[pos])}"
        )


def _safe_layer_ids(layer_ids: list[int], hidden_count: int) -> list[int]:
    out = []
    for layer_id in layer_ids:
        if 0 <= layer_id < hidden_count and layer_id not in out:
            out.append(layer_id)
        else:
            print(
                "TRACE layer_skip "
                f"layer_id={layer_id} hidden_states_count={hidden_count}"
            )
    return out


def _tensor_values(tensor: Any, limit: int) -> list[float]:
    return [float(x) for x in tensor.detach().float().cpu().flatten()[:limit].tolist()]


def _print_hidden_summaries(
    *,
    hidden_states: tuple[Any, ...],
    layer_ids: list[int],
    positions: list[int],
    limit: int,
) -> None:
    print("TRACE hidden_states")
    print(f"  count={len(hidden_states)}")
    for i, tensor in enumerate(hidden_states):
        print(f"  hidden_states[{i}].shape={tuple(tensor.shape)} dtype={tensor.dtype}")
    print("TRACE selected_hidden_values")
    print("  layer,pos,mean,std,min,max,first_values")
    for layer_id in layer_ids:
        layer = hidden_states[layer_id][0]
        for pos in positions:
            vec = layer[pos].detach().float().cpu()
            print(
                f"  {layer_id},{pos},"
                f"{_fmt(float(vec.mean().item()))},"
                f"{_fmt(float(vec.std(unbiased=False).item()))},"
                f"{_fmt(float(vec.min().item()))},"
                f"{_fmt(float(vec.max().item()))},"
                f"{[_fmt(x) for x in _tensor_values(vec, limit)]}"
            )


def _project_final_hidden(model: Any, final_hidden: Any) -> Any:
    norm = getattr(model.model, "norm", None)
    if norm is not None:
        final_hidden = norm(final_hidden)
    return model.lm_head(final_hidden)


def _print_logprob_checks(
    *,
    torch: Any,
    model: Any,
    outputs: Any,
    token_ids: list[int],
    positions: list[int],
    topk: int,
) -> None:
    logits = outputs.logits[0].detach().float()
    projected = (
        _project_final_hidden(model, outputs.hidden_states[-1]).detach().float()[0]
    )
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=logits.device)

    print("TRACE final_projection_check")
    print(f"  logits_shape={tuple(logits.shape)}")
    print(f"  projected_logits_shape={tuple(projected.shape)}")
    print(f"  max_abs_logits_diff={_fmt(float((logits - projected).abs().max().item()))}")
    print("  token_pos,next_token_id,model_logprob,projected_logprob,diff,top_tokens")
    for pos in positions:
        if pos + 1 >= len(token_ids):
            continue
        next_id = input_ids[pos + 1]
        model_lp = torch.log_softmax(logits[pos], dim=-1)
        projected_lp = torch.log_softmax(projected[pos], dim=-1)
        top = torch.topk(model_lp, k=min(topk, model_lp.numel()))
        top_pairs = [
            (int(idx), _fmt(float(val)))
            for idx, val in zip(
                top.indices.detach().cpu().tolist(),
                top.values.detach().cpu().tolist(),
                strict=True,
            )
        ]
        a = float(model_lp[next_id].item())
        b = float(projected_lp[next_id].item())
        print(
            f"  {pos},{int(next_id.item())},{_fmt(a)},{_fmt(b)},"
            f"{_fmt(abs(a - b))},{top_pairs}"
        )


def _print_repeat_diffs(
    *,
    hidden_runs: list[tuple[Any, ...]],
    layer_ids: list[int],
    positions: list[int],
) -> None:
    if len(hidden_runs) < 2:
        return
    base = hidden_runs[0]
    print("TRACE repeat_hidden_diffs")
    print("  compare_to_run0,layer,pos,max_abs_diff,mean_abs_diff")
    for run_idx, run in enumerate(hidden_runs[1:], start=1):
        for layer_id in layer_ids:
            for pos in positions:
                diff = (
                    base[layer_id][0, pos].detach().float().cpu()
                    - run[layer_id][0, pos].detach().float().cpu()
                ).abs()
                print(
                    f"  {run_idx},{layer_id},{pos},"
                    f"{_fmt(float(diff.max().item()))},"
                    f"{_fmt(float(diff.mean().item()))}"
                )


def main() -> None:
    args = _parser().parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _resolve_device(torch, args.device)
    torch_dtype = _resolve_torch_dtype(torch, args.dtype)

    print("TRACE config")
    print(f"  model_path={args.model_path}")
    print(f"  data_path={args.data_path}")
    print(f"  sample_index={args.sample_index}")
    print(f"  device={device}")
    print(f"  dtype={torch_dtype}")
    print(f"  target_layer_ids={args.target_layer_ids}")
    print(f"  repeat={args.repeat}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    token_ids = _load_token_ids(args, tokenizer)
    positions = args.positions or _default_positions(len(token_ids))
    positions = sorted({pos for pos in positions if 0 <= pos < len(token_ids)})
    _print_token_table(
        tokenizer=tokenizer,
        token_ids=token_ids,
        positions=set(positions),
        limit=args.print_token_limit,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    hidden_runs = []
    outputs = None
    with torch.no_grad():
        for run_idx in range(max(args.repeat, 1)):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_runs.append(tuple(outputs.hidden_states))
            print(
                "TRACE run "
                f"idx={run_idx} logits_shape={tuple(outputs.logits.shape)}"
            )

    if outputs is None:
        raise RuntimeError("Model forward did not run")

    layer_ids = _safe_layer_ids(args.target_layer_ids, len(outputs.hidden_states))
    _print_hidden_summaries(
        hidden_states=tuple(outputs.hidden_states),
        layer_ids=layer_ids,
        positions=positions,
        limit=args.print_hidden_limit,
    )
    _print_repeat_diffs(
        hidden_runs=hidden_runs,
        layer_ids=layer_ids,
        positions=positions,
    )
    _print_logprob_checks(
        torch=torch,
        model=model,
        outputs=outputs,
        token_ids=token_ids,
        positions=positions,
        topk=args.topk,
    )

    print("TRACE json_summary")
    print(
        json.dumps(
            {
                "seq_len": len(token_ids),
                "positions": positions,
                "target_layer_ids": layer_ids,
                "hidden_shapes": [list(t.shape) for t in outputs.hidden_states],
                "logits_shape": list(outputs.logits.shape),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
