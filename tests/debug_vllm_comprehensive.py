"""Debug vLLM hidden states - focus on 3D shape and file timing."""
import openai
import torch
import time
from pathlib import Path
import sys

sys.path.insert(0, '/workspace/speculators/src')
sys.path.insert(0, '/workspace/speculators')

from safetensors.torch import load_file

MODEL = "/models/Qwen3-4B"
client = openai.Client(base_url="http://localhost:8000/v1", api_key="dummy")


def get_path(tokens):
    resp = client.completions.create(
        model=MODEL, prompt=tokens, max_tokens=1,
        extra_body={"return_token_ids": True},
    )
    return resp.kv_transfer_params.get('hidden_states_path')


def load_with_retry(path, retries=10, delay=0.2):
    """Wait for file to appear (vLLM writes async)."""
    for i in range(retries):
        if Path(path).exists():
            return load_file(path)
        time.sleep(delay)
    return None


print("=" * 70)
print("TEST 1: Confirm 3D shape [seq_len, num_layers, hidden]")
print("=" * 70)
tokens = [151644, 8948, 198, 151644, 8948]
path = get_path(tokens)
data = load_with_retry(path)
hs = data['hidden_states']
print(f"Shape: {tuple(hs.shape)}")
print(f"  dim0 (seq_len?): {hs.shape[0]} == {len(tokens)} tokens")
print(f"  dim1 (layers?): {hs.shape[1]}")
print(f"  dim2 (hidden?): {hs.shape[2]}")
print(f"Target layers config: 5 layers + last = 6 dumped layers")
print(f"→ Layout is [seq_len={hs.shape[0]}, layers={hs.shape[1]}, hidden={hs.shape[2]}]")


print("\n" + "=" * 70)
print("TEST 2: Long sequence WITH file-wait retry")
print("=" * 70)
long_tokens = list(range(1000, 1500))
print(f"Input: {len(long_tokens)} tokens")
path = get_path(long_tokens)
print(f"Path: {path}")
print(f"Exists immediately? {Path(path).exists()}")
data = load_with_retry(path)
if data is None:
    print(f"✗ File never appeared after retries")
else:
    hs = data['hidden_states']
    print(f"✓ Loaded after retry, shape: {tuple(hs.shape)}")
    print(f"  seq_len: {hs.shape[0]}, expected: {len(long_tokens)}")
    if hs.shape[0] < len(long_tokens):
        print(f"  ⚠ APC truncated: cached {len(long_tokens) - hs.shape[0]}")


print("\n" + "=" * 70)
print("TEST 3: Repeat SAME long sequence (APC hard hit)")
print("=" * 70)
path = get_path(long_tokens)
data = load_with_retry(path)
if data is None:
    print(f"✗ File never appeared")
else:
    hs = data['hidden_states']
    print(f"Shape: {tuple(hs.shape)}, seq_len: {hs.shape[0]}")
    if hs.shape[0] == 0:
        print(f"  ✗✗ EMPTY - APC returned nothing!")
    elif hs.shape[0] < len(long_tokens):
        print(f"  ⚠ APC cached {len(long_tokens) - hs.shape[0]}, returned {hs.shape[0]}")
    else:
        print(f"  ✓ Full length (APC did not truncate)")


print("\n" + "=" * 70)
print("TEST 4: On-policy scenario (300 prefix + 7 cont)")
print("=" * 70)
prefix_len, cont_len = 300, 7
full_seq = list(range(2000, 2000 + prefix_len + cont_len))
path = get_path(full_seq)
data = load_with_retry(path)
if data is None:
    print(f"✗ File never appeared")
else:
    hs = data['hidden_states']
    actual = hs.shape[0]
    print(f"Returned {actual} of {len(full_seq)} tokens")
    if actual < len(full_seq):
        cache = len(full_seq) - actual
        n_start, n_end = prefix_len - 1, prefix_len - 1 + cont_len
        print(f"  APC cached {cache}, returned [{cache}:{len(full_seq)})")
        print(f"  Need [{n_start}:{n_end})")
        if n_start >= cache:
            print(f"  ✓ Can extract from offset {n_start - cache}")
        else:
            print(f"  ✗ Needed positions were CACHED (not returned)")


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("1. hidden_states is 3D: [seq_len, num_layers, hidden]")
print("2. Check if file needs retry-wait (async write)")
print("3. Check APC truncation behavior on repeat")
