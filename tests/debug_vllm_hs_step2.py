"""Test 2: Extract and verify hidden states from correct location."""
import openai
import numpy as np
from pathlib import Path

print("=" * 60)
print("Test: Extract hidden states from kv_transfer_params")
print("=" * 60)

client = openai.Client(base_url="http://localhost:8000/v1", api_key="dummy")
model = "Qwen/Qwen2.5-3B"

test_tokens = [151644, 8948, 198, 151644, 8948]  # 5 tokens
print(f"Input: {len(test_tokens)} tokens")

response = client.completions.create(
    model=model,
    prompt=test_tokens,
    max_tokens=1,
    extra_body={"return_token_ids": True},
)

print("✓ Request succeeded")

# Extract from kv_transfer_params
if hasattr(response, 'kv_transfer_params') and response.kv_transfer_params:
    hs_path = response.kv_transfer_params.get('hidden_states_path')
    print(f"✓ Found path: {hs_path}")

    # Load and inspect
    if Path(hs_path).exists():
        print(f"✓ File exists")

        # Load with safetensors
        try:
            from safetensors import safe_open
            with safe_open(hs_path, framework="numpy") as f:
                keys = f.keys()
                print(f"  Keys: {list(keys)}")
                for key in keys:
                    tensor = f.get_tensor(key)
                    print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")

                    # Check if shape matches input
                    if tensor.shape[0] == len(test_tokens):
                        print(f"  ✓ Shape matches input length ({len(test_tokens)})")
                    else:
                        print(f"  ✗ Shape mismatch: got {tensor.shape[0]}, expected {len(test_tokens)}")
                        print(f"     This suggests APC is active!")
        except Exception as e:
            print(f"✗ Failed to load with safetensors: {e}")
            # Try numpy
            try:
                data = np.load(hs_path)
                print(f"  Loaded with numpy: keys={list(data.keys())}")
            except:
                print(f"  Also failed with numpy")
    else:
        print(f"✗ File not found: {hs_path}")
else:
    print("✗ No kv_transfer_params in response")
    print(f"  Response attributes: {[a for a in dir(response) if not a.startswith('_')]}")

print("\n" + "=" * 60)
print("Next step: Check if vLLM client code extracts from correct location")
print("=" * 60)
