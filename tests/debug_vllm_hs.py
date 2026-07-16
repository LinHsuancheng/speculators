"""Debug script to test vLLM hidden states extraction step by step."""
import openai
import numpy as np
from pathlib import Path

# Test 1: Can we connect to vLLM?
print("=" * 60)
print("Test 1: Connect to vLLM server")
print("=" * 60)

client = openai.Client(base_url="http://localhost:8000/v1", api_key="dummy")
model = "Qwen/Qwen2.5-3B"

try:
    # Simple completion test
    response = client.completions.create(
        model=model,
        prompt="Hello",
        max_tokens=1,
    )
    print("✓ vLLM server is responding")
    print(f"  Response: {response.choices[0].text}")
except Exception as e:
    print(f"✗ Failed to connect: {e}")
    exit(1)

# Test 2: Does return_token_ids work?
print("\n" + "=" * 60)
print("Test 2: Request with return_token_ids")
print("=" * 60)

test_tokens = [151644, 8948, 198]  # Some token IDs
print(f"Input: {test_tokens}")

try:
    response = client.completions.create(
        model=model,
        prompt=test_tokens,
        max_tokens=1,
        extra_body={"return_token_ids": True},
    )
    print("✓ Request succeeded")
    print(f"  Response type: {type(response)}")
    print(f"  Response: {response}")

    # Check what's in the response
    choice = response.choices[0]
    print(f"\n  Choice attributes: {dir(choice)}")

    # Look for hidden states path
    if hasattr(choice, 'hidden_states_path'):
        hs_path = choice.hidden_states_path
        print(f"\n✓ Found hidden_states_path: {hs_path}")

        # Test 3: Can we load it?
        print("\n" + "=" * 60)
        print("Test 3: Load hidden states file")
        print("=" * 60)

        if Path(hs_path).exists():
            data = np.load(hs_path)
            print(f"✓ File exists and loaded")
            print(f"  Keys: {list(data.keys())}")
            for key in data.keys():
                print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
        else:
            print(f"✗ File not found: {hs_path}")
    else:
        print("✗ No hidden_states_path in response")
        print(f"  Available: {[a for a in dir(choice) if not a.startswith('_')]}")

except Exception as e:
    print(f"✗ Request failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("If all tests pass, vLLM is working correctly.")
print("If Test 2 has no hidden_states_path, the vLLM server may not")
print("be configured to return hidden states.")
