# Loss 配置说明

## 当前 Loss 组成

训练使用的完整 loss 由以下部分组成：

### 1. 原始 Compound Loss (CE + TV)

```python
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
```

**组成**:
- **CE Loss (0.1 权重)**: Cross-entropy loss
  - 标准的 token 预测 loss
  - 使用 gamma decay 位置权重

- **TV Loss (0.9 权重)**: Total Variation loss
  - 优化 draft 和 target 分布的 overlap
  - TV = 1 - sum_v min(p_v, q_v)
  - 等价于优化单步 acceptance rate

**位置权重**: Gamma decay (gamma=4.0)
```python
decay_fn = partial(dflash_loss_decay, gamma=gamma)
weight_k = gamma^(-(k-1))  # k=1,2,3,...
```

### 2. Confidence Head Loss

```python
CONFIDENCE_HEAD_ALPHA=1.0
```

**目标**: 预测每个位置的 acceptance rate
- Binary cross-entropy loss
- Target: c* = sum_v min(p_v, q_v) (analytical acceptance rate)
- 使用相同的 gamma decay 位置权重

### 3. Sampled Acceptance Loss (辅助)

```python
SAMPLED_ACCEPTANCE_LOSS_ALPHA=1.0
```

**目标**: 优化期望接受长度 (trials.md 公式)
- Monte Carlo loss: `-sum_t sg(C_t) * log q_t(Y_t) / K`
- Credit: `C_t = 1[q_t < p_t] * sum_{k>=t} S_k`
- 作为辅助 loss 添加，不替换原始 loss

## 总 Loss 公式

```python
loss = (
    0.1 * CE_loss +                           # Token prediction
    0.9 * TV_loss +                           # Single-step acceptance
    1.0 * confidence_loss +                   # Confidence head
    1.0 * sampled_acceptance_loss             # Multi-step acceptance (辅助)
)
```

所有组件都使用 gamma decay 位置权重（除了 sampled_acceptance_loss 有自己的 credit 权重）。

## 为什么这样设计

### TV Loss 的作用
- 优化单步 acceptance rate β_t = sum_v min(p_v, q_v)
- 等价于 1 - Total Variation distance
- 是 acceptance rate 的精确目标（不是估计）
- 不需要采样，计算高效

### Sampled Acceptance Loss 的作用
- 优化多步期望接受长度 E[L] = sum_k S_k
- 考虑了位置之间的依赖关系（survival 累积）
- 需要 on-policy 采样
- 方差较大，作为辅助信号

### 为什么同时使用两者
1. **TV loss 提供稳定的单步优化**
   - 低方差，每个 batch 都能计算
   - 优化每个位置的 marginal acceptance rate

2. **Sampled acceptance loss 提供序列级引导**
   - 考虑 prefix survival 的累积效应
   - 引导模型优化整体接受长度

3. **互补而不冲突**
   - TV 优化 marginal: β_t = E[α_t]
   - Sampled 优化 joint: E[L] = E[sum_k prod_{i<=k} α_i]
   - 两者都在优化 acceptance，但视角不同

## 与之前错误实现的对比

### 错误实现 (已修复)
```python
if sampled_draft_logprobs is not None:
    # 完全替换为 sampled credit 权重
    position_weights = sampled_credit
    loss = CE_loss  # 只有 CE，没有 TV
```

**问题**:
- TV loss 消失
- CE loss 被极小的 sampled_credit 压制
- 训练死锁

### 正确实现 (当前)
```python
# 始终使用原始 compound loss
loss, term_losses = compound_loss(
    logits, targets, loss_mask, pos_idx,
    loss_config={"ce": 0.1, "tv": 0.9},
    decay_fn=gamma_decay,
    position_weights=cat_weights,
)

# 辅助 loss
if sampled_exact_loss is not None:
    loss = loss + sampled_acceptance_loss_alpha * sampled_exact_loss
```

**好处**:
- TV loss 继续工作
- 所有 loss 使用正常的 gamma decay 权重
- Sampled loss 作为辅助信号

## 训练脚本配置

```bash
# examples/train/dspark_qwen3_4b_trainer.sh

LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0
SAMPLED_ACCEPTANCE_LOSS_ALPHA=1.0

torchrun scripts/train.py \
    --loss-fn "$LOSS_FN" \
    --confidence-head-alpha "$CONFIDENCE_HEAD_ALPHA" \
    --sampled-acceptance-loss-alpha "$SAMPLED_ACCEPTANCE_LOSS_ALPHA" \
    --enable-sampled-acceptance-loss \
    ...
```

## 监控 Loss 组件

```bash
# 查看所有 loss 组件
tail -f train.log | grep -E "ce_loss|tv_loss|confidence_loss|sampled_acceptance_loss|^loss:"
```

预期输出：
```
ce_loss: 2.345
tv_loss: 0.523
confidence_loss: 0.412
sampled_acceptance_loss: -1.234
loss: 1.046  # = 0.1*2.345 + 0.9*0.523 + 1.0*0.412 + 1.0*(-1.234)
```

## 超参数调整建议

### CE vs TV 权重
```python
# 当前: {"ce": 0.1, "tv": 0.9}
# TV 主导，因为 acceptance rate 是主要目标

# 如果 token accuracy 太低，增加 CE:
{"ce": 0.3, "tv": 0.7}

# 如果只关注 acceptance，减少 CE:
{"ce": 0.05, "tv": 0.95}
```

### Sampled Acceptance Loss 权重
```python
# 当前: 1.0

# 训练初期如果太不稳定，减小权重:
SAMPLED_ACCEPTANCE_LOSS_ALPHA=0.1

# 训练后期如果想强化多步优化，增加权重:
SAMPLED_ACCEPTANCE_LOSS_ALPHA=2.0
```

### Confidence Head 权重
```python
# 当前: 1.0

# 如果不需要 confidence head，设为 0 或去掉 --enable-confidence-head
CONFIDENCE_HEAD_ALPHA=0.0

# 如果需要精确的 confidence 校准:
CONFIDENCE_HEAD_ALPHA=2.0
```

## 参考

- TV Loss 实现: `src/speculators/models/metrics.py`
- Sampled Acceptance Loss: `src/speculators/models/dspark/metrics.py`
- Compound Loss: `src/speculators/models/metrics.py:compound_loss()`
- 数学推导: trials.md (sampled acceptance), TV loss paper
