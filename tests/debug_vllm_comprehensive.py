"""Comprehensive vLLM hidden states debugging - all tests in one file."""
import openai
import torch
from pathlib import Path
import sys

sys.path.insert(0, '/workspace/speculators/src')
sys.path.insert(0, '/workspace/speculators')

from safetensors.torch import load_file

MODEL = "/models/Qwen3-4B"
client = openai.Client(base_url="http://localhost:8000/v1", api_key="dummy")


def get_hs(tokens):
    """Request and return (path, hidden_states tensor, token_ids)."""
    resp = client.completions.create(
        model=MODEL,
        prompt=tokens,
        max_tokens=1,
        extra_body={"return_token_ids": True},
    )
    path = resp.kv_transfer_params.get('hidden_states_path')
    data = load_file(path)
    return path, data


print("=" * 70)
print("TEST 1: Short sequence (5 tokens)")
print("=" * 70)
tokens = [151644, 8948, 198, 151644, 8948]
print(f"Input: {len(tokens)} tokens")
path, data = get_hs(tokens)
print(f"Path: {path}")
print(f"Keys: {list(data.keys())}")
for k, v in data.items():
    print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
hs = data['hidden_states']
print(f"hidden_states length: {hs.shape[0]}, expected: {len(tokens)}")
if hs.shape[0] == len(tokens):
    print("✓ Full length returned")
else:
    print(f"⚠ APC truncated: got {hs.shape[0]}, cached {len(tokens) - hs.shape[0]}")


print("\n" + "=" * 70)
print("TEST 2: extract_output function")
print("=" * 70)
from speculators.data_generation.vllm_client import extract_output
resp = client.completions.create(
    model=MODEL, prompt=tokens, max_tokens=1,
    extra_body={"return_token_ids": True},
)
try:
    p = extract_output(resp, tokens)
    print(f"✓ extract_output: {p}")
except Exception as e:
    print(f"✗ extract_output failed: {e}")


print("\n" + "=" * 70)
print("TEST 3: Long sequence (500 tokens) - trigger APC")
print("=" * 70)
long_tokens = list(range(1000, 1500))
print(f"Input: {len(long_tokens)} tokens")
path, data = get_hs(long_tokens)
hs = data['hidden_states']
print(f"hidden_states length: {hs.shape[0]}, expected: {len(long_tokens)}")
if hs.shape[0] < len(long_tokens):
    cache = len(long_tokens) - hs.shape[0]
    print(f"⚠ APC cached {cache} tokens, returned positions [{cache}:{len(long_tokens)})")
else:
    print("✓ Full length")


print("\n" + "=" * 70)
print("TEST 4: SAME long sequence AGAIN - APC should hit hard")
print("=" * 70)
path, data = get_hs(long_tokens)
hs = data['hidden_states']
print(f"hidden_states length: {hs.shape[0]}, expected: {len(long_tokens)}")
if hs.shape[0] < len(long_tokens):
    cache = len(long_tokens) - hs.shape[0]
    print(f"⚠ APC cached {cache}, returned {hs.shape[0]}")
    if hs.shape[0] == 0:
        print("  ✗✗ COMPLETELY EMPTY - this is the bug!")
else:
    print("✓ Full length")


print("\n" + "=" * 70)
print("TEST 5: On-policy scenario (300 prefix + 7 continuation)")
print("=" * 70)
prefix_len = 300
cont_len = 7
full_seq = list(range(2000, 2000 + prefix_len + cont_len))
print(f"Sequence: {prefix_len} prefix + {cont_len} cont = {len(full_seq)}")
print(f"Need positions: [{prefix_len-1}:{prefix_len-1+cont_len})")
path, data = get_hs(full_seq)
hs = data['hidden_states']
actual = hs.shape[0]
print(f"Returned: {actual} tokens")
if actual < len(full_seq):
    cache = len(full_seq) - actual
    r_start, r_end = cache, len(full_seq)
    n_start, n_end = prefix_len - 1, prefix_len - 1 + cont_len
    print(f"APC cached first {cache}, returned [{r_start}:{r_end})")
    print(f"Needed [{n_start}:{n_end})")
    if n_start >= r_start and n_end <= r_end:
        print(f"✓ Extract from offset {n_start - r_start}")
    else:
        print(f"✗ Needed range NOT available (cached too much)")


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Check: does APC return empty/truncated for repeated sequences?")
print("If TEST 4 returns 0, that confirms the on-policy failure cause.")
