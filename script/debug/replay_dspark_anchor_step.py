#!/usr/bin/env python3
"""Single-sample DSpark anchor replay for hidden-state layer debugging."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

log = logging.getLogger("replay_dspark_anchor_step")


def text(tokenizer, token_id: int) -> str:
    return repr(tokenizer.decode([int(token_id)], skip_special_tokens=False))


def dtype_of(torch, name: str):
    return "auto" if name == "auto" else getattr(torch, name)


def map_draft_to_target(torch, draft, draft_id: int) -> int:
    if draft.use_draft_vocab and draft.d2t is not None:
        did = torch.tensor(draft_id, device=draft.d2t.device, dtype=torch.long)
        return int((did + draft.d2t[did].long()).item())
    return int(draft_id)


def load_vocab_maps(torch, args):
    paths = []
    if args.d2t_path or args.t2d_path:
        if not (args.d2t_path and args.t2d_path):
            raise ValueError("--d2t-path and --t2d-path must be passed together.")
        paths.append((args.d2t_path, args.t2d_path, "explicit"))
    paths += [
        (Path(args.draft_model) / "d2t.npy", Path(args.draft_model) / "t2d.npy", "draft"),
        (Path(args.data_path) / "d2t.npy", Path(args.data_path) / "t2d.npy", "data"),
    ]
    for d2t_path, t2d_path, source in paths:
        if d2t_path.exists() and t2d_path.exists():
            import numpy as np

            log.info("loading vocab maps from %s: %s %s", source, d2t_path, t2d_path)
            return torch.from_numpy(np.load(d2t_path)), torch.from_numpy(
                np.load(t2d_path)
            )
    return None, None


def load_vllm_sample(torch, args):
    import openai

    from speculators.data_generation.offline import check_hidden_states
    from speculators.data_generation.vllm_client import generate_hidden_states
    from speculators.train.data import ArrowDataset, _maybe_load_hs_file, build_client_item

    dataset = ArrowDataset(
        max_len=args.total_seq_len,
        datapath=args.data_path,
        hidden_states_path=None,
        vllm_endpoint=args.vllm_endpoint,
        on_missing="raise",
        hidden_states_dtype=dtype_of(torch, args.hidden_states_dtype),
        model=args.vllm_model,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
    )
    if not 0 <= args.sample_index < len(dataset):
        raise ValueError(f"--sample-index out of range [0, {len(dataset) - 1}]")

    row = dataset.data[args.sample_index]
    client = openai.OpenAI(base_url=args.vllm_endpoint, api_key="EMPTY", max_retries=0)
    vllm_model = args.vllm_model or client.models.list().data[0].id
    log.info(
        "requesting vLLM hidden states endpoint=%s model=%s sample_index=%d",
        args.vllm_endpoint,
        vllm_model,
        args.sample_index,
    )
    hs_path = Path(
        generate_hidden_states(
            client,
            vllm_model,
            build_client_item(row),
            timeout=args.request_timeout,
            max_retries=args.max_retries,
        )
    )
    hs = _maybe_load_hs_file(hs_path)
    if hs is None:
        raise FileNotFoundError(
            f"vLLM returned hidden state path but file is missing: {hs_path}"
        )

    expected = torch.as_tensor(row["input_ids"], dtype=torch.long)
    check_hidden_states(hs, expected.tolist())
    if not torch.equal(hs["token_ids"].cpu(), expected.cpu()):
        raise ValueError(f"vLLM token_ids do not match dataset row: {hs_path}")

    sample = SimpleNamespace(
        hs_path=hs_path,
        raw_hidden=hs["hidden_states"],
        token_ids=hs["token_ids"].long(),
        loss_mask=torch.as_tensor(row["loss_mask"]),
        train_hidden=hs["hidden_states"][:, :-1]
        .flatten(1)
        .to(dtype_of(torch, args.hidden_states_dtype)),
        train_last=hs["hidden_states"][:, -1].to(
            dtype_of(torch, args.hidden_states_dtype)
        ),
    )
    if args.delete_vllm_hidden_state:
        hs_path.unlink(missing_ok=True)
        log.info("deleted transient vLLM hidden state file: %s", hs_path)
    return sample


def choose_anchor(torch, loss_mask, block_size: int, requested: int | None) -> int:
    valid = loss_mask.bool().clone()
    valid[-block_size:] = False
    if requested is not None:
        if not 0 <= requested < loss_mask.numel() or not bool(valid[requested].item()):
            raise ValueError(
                "--anchor-position must have loss_mask=1 and leave one full block"
            )
        return int(requested)
    candidates = torch.nonzero(valid, as_tuple=False).view(-1)
    if candidates.numel() == 0:
        raise ValueError("sample has no valid anchor")
    return int(candidates[0].item())


def sim(torch, a, b):
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    return (
        float(torch.nn.functional.cosine_similarity(a, b, dim=0).item()),
        float((a - b).abs().max().item()),
        float((a - b).abs().mean().item()),
    )


def hf_feature(torch, hf_hidden_states, layer_ids, offset: int):
    tensors = []
    for layer_id in layer_ids:
        idx = int(layer_id) + offset
        if not 0 <= idx < len(hf_hidden_states):
            raise IndexError(
                f"HF hidden_states[{idx}] invalid for layer_id={layer_id}, "
                f"offset={offset}, len={len(hf_hidden_states)}"
            )
        tensors.append(hf_hidden_states[idx])
    return torch.cat(tensors, dim=-1)


def print_hidden_alignment(torch, cached, direct, offset, layer_ids):
    for name, tensor in (("hf[layer_id]", direct), ("hf[layer_id+1]", offset)):
        cos, max_abs, mean_abs = sim(torch, cached, tensor)
        log.info("%s flat: cosine=%.8f max_abs=%.8f mean_abs=%.8f", name, cos, max_abs, mean_abs)

    hidden = cached.shape[-1] // len(layer_ids)
    direct_scores, offset_scores = [], []
    for pos, layer_id in enumerate(layer_ids):
        lo, hi = pos * hidden, (pos + 1) * hidden
        d = sim(torch, cached[:, :, lo:hi], direct[:, :, lo:hi])
        o = sim(torch, cached[:, :, lo:hi], offset[:, :, lo:hi])
        direct_scores.append(d)
        offset_scores.append(o)
        log.info(
            "layer_id=%s vllm_slot=%d | hf[layer_id] cos=%.8f max_abs=%.8f | "
            "hf[layer_id+1] cos=%.8f max_abs=%.8f",
            layer_id,
            pos,
            d[0],
            d[1],
            o[0],
            o[1],
        )

    d_cos = sum(x[0] for x in direct_scores) / len(direct_scores)
    o_cos = sum(x[0] for x in offset_scores) / len(offset_scores)
    d_abs = max(x[1] for x in direct_scores)
    o_abs = max(x[1] for x in offset_scores)
    winner = "hf[layer_id+1]" if (o_cos, -o_abs) > (d_cos, -d_abs) else "hf[layer_id]"
    log.info(
        "layer_semantics_winner=%s direct_mean_cos=%.8f offset_mean_cos=%.8f "
        "direct_max_abs=%.8f offset_max_abs=%.8f",
        winner,
        d_cos,
        o_cos,
        d_abs,
        o_abs,
    )


def replay_slot1(torch, draft, features, verifier_last, token_ids, real_loss_mask, anchor):
    block = int(draft.block_size)
    input_ids = token_ids.unsqueeze(0)
    seq_len = input_ids.shape[1]
    loss_mask = real_loss_mask.to(device=input_ids.device).bool().unsqueeze(0).clone()
    loss_mask[:, :] = False
    loss_mask[0, anchor : anchor + block] = real_loss_mask[
        anchor : anchor + block
    ].bool()
    loss_mask[0, anchor] = True
    document_ids = torch.zeros((1, seq_len), dtype=torch.long, device=input_ids.device)
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

    hidden, base_logits, targets, aligned_mask, anchored_idx = draft._backbone_forward(
        features, input_ids, loss_mask, verifier_last, document_ids, position_ids
    )
    n_blocks = int(draft.config.max_anchors)
    block_tokens = input_ids[0, anchored_idx].view(n_blocks, block)
    prev_ids = draft._build_markov_prev_token_ids(
        block_tokens, getattr(draft.config, "sample_from_anchor", False)
    )
    hidden_blocks = hidden.view(n_blocks, block, -1)
    base = base_logits.view(n_blocks, block, -1)
    if draft.markov_head is None:
        bias = torch.zeros_like(base)
    else:
        bias = draft.markov_head.block_bias(
            prev_token_ids=prev_ids, hidden_states=hidden_blocks
        )
    slot = 1
    return SimpleNamespace(
        anchor=int(anchored_idx[0].item()),
        prev=int(prev_ids[0, slot].item()),
        base=base[0, slot],
        bias=bias[0, slot],
        final=base[0, slot] + bias[0, slot],
        target=targets.view(n_blocks, block, -1)[0, slot],
        mask=float(aligned_mask.view(n_blocks, block)[0, slot].item()),
    )


def print_replay(torch, tokenizer, draft, name: str, r):
    base_id = int(torch.argmax(r.base).item())
    bias_id = int(torch.argmax(r.bias).item())
    final_id = int(torch.argmax(r.final).item())
    target_id = int(torch.argmax(r.target).item())
    base_tid = map_draft_to_target(torch, draft, base_id)
    final_tid = map_draft_to_target(torch, draft, final_id)
    target_tid = map_draft_to_target(torch, draft, target_id)

    log.info("replay=%s anchor_from_backbone=%d", name, r.anchor)
    log.info(
        "replay=%s slot1 base_logits top1_draft=%d value=%.8f target_id=%d text=%s",
        name,
        base_id,
        float(r.base[base_id].float().item()),
        base_tid,
        text(tokenizer, base_tid),
    )
    log.info(
        "replay=%s slot1 markov_bias prev_id=%d top1_draft=%d top1=%.8f "
        "final_token_bias=%.8f max_abs=%.8f l2=%.8f",
        name,
        r.prev,
        bias_id,
        float(r.bias[bias_id].float().item()),
        float(r.bias[final_id].float().item()),
        float(r.bias.float().abs().max().item()),
        float(torch.linalg.vector_norm(r.bias.float()).item()),
    )
    log.info(
        "replay=%s final slot1 draft top1 draft_id=%d value=%.8f target_id=%d text=%s",
        name,
        final_id,
        float(r.final[final_id].float().item()),
        final_tid,
        text(tokenizer, final_tid),
    )
    log.info(
        "replay=%s training target slot1 top1 draft_id=%d value=%.8f "
        "target_id=%d text=%s aligned_loss_mask=%.1f",
        name,
        target_id,
        float(r.target[target_id].float().item()),
        target_tid,
        text(tokenizer, target_tid),
        r.mask,
    )


def compare_replay(torch, cached, other, name: str):
    for field in ("base", "bias", "final"):
        cos, max_abs, mean_abs = sim(torch, getattr(cached, field), getattr(other, field))
        log.info(
            "compare vllm_request vs %s slot1_%s: cosine=%.8f max_abs=%.8f mean_abs=%.8f",
            name,
            field,
            cos,
            max_abs,
            mean_abs,
        )


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from speculators.models.dspark.core import DSparkDraftModel

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cached = load_vllm_sample(torch, args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.verifier_model, trust_remote_code=args.trust_remote_code
    )

    cfg = DSparkDraftModel.config_class.from_pretrained(args.draft_model)
    if args.sample_from_anchor:
        cfg.sample_from_anchor = True
    if args.draft_attn_impl != "auto":
        cfg.transformer_layer_config._attn_implementation = args.draft_attn_impl
    d2t, t2d = load_vocab_maps(torch, args)
    draft = DSparkDraftModel.from_pretrained(
        args.draft_model, config=cfg, d2t=d2t, t2d=t2d
    ).to(device).eval()

    layer_ids = [int(x) for x in draft.target_layer_ids]
    hidden_size = int(draft.config.transformer_layer_config.hidden_size)
    aux_layers = cached.train_hidden.shape[-1] // hidden_size
    if aux_layers != len(layer_ids):
        raise ValueError(
            f"vLLM aux layers after training layout={aux_layers}, but draft "
            f"target_layer_ids={layer_ids}. vLLM layer list mismatches config."
        )

    anchor = choose_anchor(torch, cached.loss_mask, int(draft.block_size), args.anchor_position)
    token_ids = cached.token_ids.to(device)
    input_ids = token_ids.unsqueeze(0)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)

    log.info("sample_index=%d vllm_hidden_file=%s", args.sample_index, cached.hs_path)
    log.info(
        "seq_len=%d raw_vllm_hidden_shape=%s train_hidden_shape=%s",
        token_ids.numel(),
        tuple(cached.raw_hidden.shape),
        tuple(cached.train_hidden.shape),
    )
    log.info(
        "target_layer_ids=%s block_size=%d max_anchors=%d sample_from_anchor=%s",
        layer_ids,
        int(draft.block_size),
        int(draft.config.max_anchors),
        bool(getattr(draft.config, "sample_from_anchor", False)),
    )
    log.info(
        "fixed_anchor=%d anchor_id=%d anchor_text=%s next_gt_id=%d next_gt_text=%s",
        anchor,
        int(token_ids[anchor].item()),
        text(tokenizer, int(token_ids[anchor].item())),
        int(token_ids[anchor + 1].item()),
        text(tokenizer, int(token_ids[anchor + 1].item())),
    )

    target = AutoModelForCausalLM.from_pretrained(
        args.verifier_model,
        torch_dtype=dtype_of(torch, args.dtype),
        trust_remote_code=args.trust_remote_code,
    ).to(device).eval()

    with torch.inference_mode():
        out = target(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
        )
        cached_aux = cached.train_hidden.unsqueeze(0).to(device)
        cached_last = cached.train_last.unsqueeze(0).to(device)
        hf_direct = hf_feature(torch, out.hidden_states, layer_ids, 0).to(cached_aux.dtype)
        hf_offset = hf_feature(torch, out.hidden_states, layer_ids, 1).to(cached_aux.dtype)

        print_hidden_alignment(torch, cached_aux, hf_direct, hf_offset, layer_ids)

        anchor_logits = out.logits[0, anchor]
        target_top1 = int(torch.argmax(anchor_logits).item())
        log.info(
            "HF anchor target top1 position=%d target_id=%d value=%.8f text=%s",
            anchor,
            target_top1,
            float(anchor_logits[target_top1].float().item()),
            text(tokenizer, target_top1),
        )

        replays = {
            "vllm_request": replay_slot1(
                torch, draft, cached_aux, cached_last, token_ids, cached.loss_mask, anchor
            ),
            "hf_layer_id": replay_slot1(
                torch, draft, hf_direct, cached_last, token_ids, cached.loss_mask, anchor
            ),
            "hf_layer_id_plus_1": replay_slot1(
                torch, draft, hf_offset, cached_last, token_ids, cached.loss_mask, anchor
            ),
        }
        for name, replay in replays.items():
            print_replay(torch, tokenizer, draft, name, replay)
        compare_replay(torch, replays["vllm_request"], replays["hf_layer_id"], "hf_layer_id")
        compare_replay(
            torch,
            replays["vllm_request"],
            replays["hf_layer_id_plus_1"],
            "hf_layer_id_plus_1",
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
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--anchor-position", type=int, default=None)
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
