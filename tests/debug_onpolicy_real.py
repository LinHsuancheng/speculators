"""Test real on-policy scorer flow with actual data."""
import sys
sys.path.insert(0, '/workspace/speculators/src')
sys.path.insert(0, '/workspace/speculators')

import torch
import openai
from speculators.models.dspark.onpolicy import VLLMVerifierScorer, build_scored_sequences
from safetensors.torch import load_file


print("=" * 70)
print("TEST: Real on-policy scorer with simulated batch")
print("=" * 70)

client = openai.Client(base_url="http://localhost:8000/v1", api_key="dummy")
model = "/models/Qwen3-4B"
hidden_size = 2560


def load_hs(path):
    """Mimic train.py's _load_last_layer_hs."""
    data = load_file(path)
    hs = data['hidden_states']
    print(f"  Loaded shape: {tuple(hs.shape)}")
    if hs.shape[0] == 0:
        print(f"    ✗ Empty!")
        return torch.zeros(0, hidden_size)
    # 3D: [seq_len, layers, hidden] → take last layer
    if hs.ndim == 3:
        hs = hs[:, -1, :]  # [seq_len, hidden]
    return hs


scorer = VLLMVerifierScorer(
    client=client,
    model=model,
    load_hidden_states=load_hs,
    hidden_size=hidden_size,
    request_timeout=30.0,
)

# Simulate one batch with 2 blocks
# gold_input_ids: [1, total_seq_len] - packed sequence
# document_ids: [1, total_seq_len] - document boundary markers
# anchor_positions: [num_blocks] - where each block starts
# sampled_verifier_ids: [num_blocks, K] - sampled continuation tokens

# Simulate: 2 documents in packed sequence, 1 block per doc
# Doc 0: 300 tokens (positions 0-299), anchor at 293, sample 7 tokens
# Doc 1: 250 tokens (positions 300-549), anchor at 543, sample 7 tokens
gold_input_ids = torch.tensor([list(range(10000, 10300)) + list(range(20000, 20250))]).long()
document_ids = torch.tensor([[0] * 300 + [1] * 250]).long()
anchor_positions = torch.tensor([293, 543]).long()
sampled_verifier_ids = torch.tensor([
    list(range(30000, 30007)),  # block 0 samples
    list(range(40000, 40007)),  # block 1 samples
]).long()

print(f"gold_input_ids: {tuple(gold_input_ids.shape)}")
print(f"document_ids: {tuple(document_ids.shape)}")
print(f"anchor_positions: {anchor_positions.tolist()}")
print(f"sampled_verifier_ids: {tuple(sampled_verifier_ids.shape)}")

# Build sequences (what gets sent to vLLM)
sequences = build_scored_sequences(
    gold_input_ids, document_ids, anchor_positions, sampled_verifier_ids
)
print(f"\nBuilt {len(sequences)} sequences:")
for i, seq in enumerate(sequences):
    print(f"  Seq {i}: {len(seq)} tokens (anchor {anchor_positions[i].item()})")

# Score
print(f"\nCalling scorer.score()...")
try:
    verifier_hidden, valid_mask = scorer.score(
        gold_input_ids=gold_input_ids,
        document_ids=document_ids,
        anchor_positions=anchor_positions,
        sampled_verifier_ids=sampled_verifier_ids,
    )
    print(f"✓ Returned verifier_hidden: {tuple(verifier_hidden.shape)}")
    print(f"  valid_mask: {valid_mask.tolist()}")
    num_valid = valid_mask.sum().item()
    print(f"  {num_valid}/{len(sequences)} samples valid")

    if num_valid == 0:
        print(f"  ✗✗ ALL INVALID - this is the training failure!")
    elif num_valid < len(sequences):
        print(f"  ⚠ Some invalid")
    else:
        print(f"  ✓ All valid")

except Exception as e:
    print(f"✗ scorer.score() failed: {e}")
    import traceback
    traceback.print_exc()


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("This tests the actual on-policy scorer with realistic data.")
print("If all samples are invalid, that confirms the training bug.")
