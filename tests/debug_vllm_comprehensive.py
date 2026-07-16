"""Comprehensive vLLM hidden states debugging - all tests in one file."""
import openai
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, '/workspace/speculators/src')
sys.path.insert(0, '/workspace/speculators')

print("=" * 70)
print("TEST 1: Extract hidden states path from response")
print("=" * 70)

client = openai.Client(base_url="http://localhost:8000/v1", api_key="dummy")
model = "Qwen/Qwen2.5-3B"

test_tokens = [151644, 8948, 198, 151644, 8948]  # 5 tokens
print(f"Input: {len(test_tokens)} tokens = {test_tokens}")

response = client.completions.create(
    model=model,
    prompt=test_tokens,
    max_tokens=1,
    extra_body={"return_token_ids": True},
)

# Extract path
hs_path = None
if hasattr(response, 'kv_transfer_params') and response.kv_transfer_params:
    hs_path = response.kv_transfer_params.get('hidden_states_path')
    print(f"✓ Found in kv_transfer_params: {hs_path}")
else:
    print(f"✗ No kv_transfer_params")
    sys.exit(1)


print("\n" + "=" * 70)
print("TEST 2: Load hidden states file and check shape")
print("=" * 70)

if not Path(hs_path).exists():
    print(f"✗ File not found: {hs_path}")
    sys.exit(1)

print(f"✓ File exists: {hs_path}")

# Try safetensors first
try:
    from safetensors import safe_open
    with safe_open(hs_path, framework="numpy") as f:
        keys = list(f.keys())
        print(f"✓ Loaded with safetensors")
        print(f"  Keys: {keys}")

        for key in keys:
            tensor = f.get_tensor(key)
            print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")

            expected_len = len(test_tokens)
            actual_len = tensor.shape[0]

            if actual_len == expected_len:
                print(f"  ✓ Length matches input ({expected_len} tokens)")
            elif actual_len < expected_len:
                print(f"  ⚠ SHORTER: got {actual_len}, expected {expected_len}")
                print(f"    → APC returned only last {actual_len} tokens")
                print(f"    → Cached first {expected_len - actual_len} tokens")
            else:
                print(f"  ✗ LONGER: got {actual_len}, expected {expected_len}")

except Exception as e:
    print(f"✗ Failed to load: {e}")
    sys.exit(1)


print("\n" + "=" * 70)
print("TEST 3: Check vllm_client.py extract_output function")
print("=" * 70)

from speculators.data_generation.vllm_client import extract_output

try:
    extracted_path = extract_output(response, test_tokens)
    print(f"✓ extract_output returned: {extracted_path}")

    if extracted_path == hs_path:
        print(f"  ✓ Path matches kv_transfer_params")
    else:
        print(f"  ✗ Path mismatch!")
        print(f"    extract_output: {extracted_path}")
        print(f"    kv_transfer_params: {hs_path}")

except Exception as e:
    print(f"✗ extract_output failed: {e}")
    import traceback
    traceback.print_exc()


print("\n" + "=" * 70)
print("TEST 4: Test with longer sequence (check APC behavior)")
print("=" * 70)

# Use a longer sequence to trigger APC
long_tokens = list(range(151644, 151644 + 500))  # 500 tokens
print(f"Input: {len(long_tokens)} tokens")

response2 = client.completions.create(
    model=model,
    prompt=long_tokens,
    max_tokens=1,
    extra_body={"return_token_ids": True},
)

hs_path2 = response2.kv_transfer_params.get('hidden_states_path')
print(f"Response path: {hs_path2}")

try:
    from safetensors import safe_open
    with safe_open(hs_path2, framework="numpy") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            actual_len = tensor.shape[0]
            expected_len = len(long_tokens)

            print(f"  {key}: shape={tensor.shape}")
            print(f"  Expected: {expected_len} tokens")
            print(f"  Got: {actual_len} tokens")

            if actual_len < expected_len:
                cache_hit = expected_len - actual_len
                print(f"  ⚠ APC cached {cache_hit} tokens, returned {actual_len}")
                print(f"  → Returned positions: [{cache_hit}:{expected_len})")
            elif actual_len == expected_len:
                print(f"  ✓ Full sequence returned (no APC)")

except Exception as e:
    print(f"✗ Load failed: {e}")


print("\n" + "=" * 70)
print("TEST 5: Simulate on-policy scenario")
print("=" * 70)

# Simulate: gold prefix (300 tokens) + sampled continuation (7 tokens)
prefix_len = 300
continuation_len = 7
full_seq = list(range(151644, 151644 + prefix_len + continuation_len))

print(f"Sequence: {prefix_len} prefix + {continuation_len} continuation = {len(full_seq)} total")
print(f"Need positions: [{prefix_len-1}:{prefix_len-1+continuation_len})")

response3 = client.completions.create(
    model=model,
    prompt=full_seq,
    max_tokens=1,
    extra_body={"return_token_ids": True},
)

hs_path3 = response3.kv_transfer_params.get('hidden_states_path')

try:
    from safetensors import safe_open
    with safe_open(hs_path3, framework="numpy") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            actual_len = tensor.shape[0]

            print(f"  Returned: {actual_len} tokens")

            if actual_len < len(full_seq):
                cache_hit = len(full_seq) - actual_len
                returned_start = cache_hit
                returned_end = len(full_seq)

                needed_start = prefix_len - 1
                needed_end = prefix_len - 1 + continuation_len

                print(f"  APC cached: first {cache_hit} tokens")
                print(f"  Returned range: [{returned_start}:{returned_end})")
                print(f"  Needed range: [{needed_start}:{needed_end})")

                if needed_start >= returned_start and needed_end <= returned_end:
                    offset = needed_start - returned_start
                    print(f"  ✓ Can extract from offset {offset}")
                    print(f"    Extract: hs[{offset}:{offset + continuation_len}]")
                else:
                    print(f"  ✗ Needed range NOT in returned range")
            else:
                print(f"  ✓ Full sequence, extract normally")

except Exception as e:
    print(f"✗ Load failed: {e}")


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("Key findings:")
print("1. Hidden states path is in response.kv_transfer_params")
print("2. APC causes vLLM to return only non-cached suffix")
print("3. We need to compute offset when extracting from truncated tensor")
print("4. Check if vllm_client.py extracts from the correct location")
