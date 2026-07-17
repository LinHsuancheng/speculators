#!/usr/bin/env python3
"""Test the enhanced metrics for sample acceptance length training."""

import torch
from src.speculators.models.dspark.metrics import (
    sampled_acceptance_credit,
    exact_acceptance_length_loss,
)

print("=== Testing Enhanced Metrics ===\n")

# 创建测试数据
batch_size = 2
block_size = 5
draft_logp = torch.tensor([
    [-1.0, -1.5, -2.0, -2.5, -3.0],
    [-0.8, -1.2, -1.8, -2.2, -2.8],
])
target_logp = torch.tensor([
    [-0.9, -1.3, -1.8, -2.3, -2.8],  # Target 更确信
    [-1.0, -1.4, -2.0, -2.4, -3.0],  # Target 稍微更确信
])

print("1. Testing sampled_acceptance_credit()")
print(f"   Draft logp shape: {draft_logp.shape}")
print(f"   Target logp shape: {target_logp.shape}")

credit = sampled_acceptance_credit(draft_logp, target_logp)
print(f"   Credit shape: {credit.shape}")
print(f"   Credit:\n{credit}")
print(f"   Credit mean: {credit.mean().item():.4f}")
print(f"   Credit min: {credit.min().item():.4f}")
print(f"   Credit max: {credit.max().item():.4f}")
print(f"   Credit std: {credit.std().item():.4f}")
print()

print("2. Testing exact_acceptance_length_loss()")
loss = exact_acceptance_length_loss(draft_logp, target_logp, credit=credit)
print(f"   Loss: {loss.item():.4f}")
print()

print("3. Testing derived metrics")
log_alpha = torch.minimum(
    torch.zeros_like(draft_logp),
    target_logp - draft_logp,
)
alpha = torch.exp(log_alpha)
print(f"   Alpha (acceptance ratio):\n{alpha}")
print(f"   Alpha mean: {alpha.mean().item():.4f}")
print(f"   Alpha min: {alpha.min().item():.4f}")
print()

survival = torch.exp(torch.cumsum(log_alpha, dim=-1))
print(f"   Survival:\n{survival}")
print(f"   Final survival (per block): {survival[:, -1]}")
print()

undercovered = (draft_logp < target_logp).float()
print(f"   Undercovered ratio: {undercovered.mean().item():.4f}")
print(f"   Overcovered ratio: {(1 - undercovered).mean().item():.4f}")
print()

continuation = torch.flip(
    torch.cumsum(torch.flip(survival, dims=[-1]), dim=-1),
    dims=[-1],
)
estimated_accept_len = continuation[:, 0]
print(f"   Estimated acceptance length per block: {estimated_accept_len}")
print(f"   Mean estimated acceptance length: {estimated_accept_len.mean().item():.4f}")
print()

print("✓ All metrics computed successfully!")
print()
print("Expected behavior:")
print("  - Credit should be non-zero where draft_logp < target_logp")
print("  - Alpha should be in (0, 1] range")
print("  - Survival should decrease along the sequence")
print("  - Loss should be negative (we're maximizing acceptance length)")
