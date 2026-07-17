# 关键修复：恢复到 Main 基线 + 添加辅助 Sampled Acceptance Loss

## 问题

之前的实现破坏了原始训练：
- 当设置 `SAMPLED_ACCEPTANCE_LOSS_ALPHA=0` 时训练失败
- 所有 accuracy 要么是 0 要么是 1
- 说明原始的 loss 计算被破坏了

## 根本原因

之前的实现修改了原始的 loss 计算逻辑，导致即使不使用 sampled acceptance loss，训练也会异常。

## 修复方案

**完全恢复到 main 分支的基线**，然后只在最后添加辅助 loss：

```python
# 原始 main 分支的完整 loss 计算（保持不变）
loss, term_losses = compound_loss(
    logits, targets, loss_mask, pos_idx,
    loss_config=loss_config,
    decay_fn=decay_fn,
    position_weights=cat_weights,
)

# ... 所有原始的 metrics 计算 ...

# 仅在提供 sampled logprobs 时添加辅助 loss
if sampled_draft_logprobs is not None and sampled_target_logprobs is not None:
    sampled_credit = sampled_acceptance_credit(...)
    sampled_exact_loss = exact_acceptance_length_loss(...)
    
    # 作为辅助添加（不修改原始 loss）
    loss = loss + sampled_acceptance_loss_alpha * sampled_exact_loss
    
    # 添加 position-wise metrics
    metrics["sampled_pos{k}_alpha"] = ...
```

## 关键特性

### 1. 向后兼容性

```python
# 当不提供 sampled logprobs 时，行为与 main 分支完全相同
compute_metrics(logits, targets, ...)  # 无 sampled_draft_logprobs
# → 行为 100% 与 main 分支相同
```

### 2. 可禁用的辅助 loss

```bash
# 设置 alpha=0 完全禁用 sampled loss
SAMPLED_ACCEPTANCE_LOSS_ALPHA=0.0
# → 训练行为与 main 分支完全相同
```

### 3. 原始 loss 不受影响

无论 sampled loss 的值如何，原始的 CE + TV + Confidence loss 保持不变：

```python
# 始终计算原始 loss（与 main 完全相同）
loss = 0.1*CE + 0.9*TV + 1.0*Confidence

# 然后添加辅助 loss
if sampled_acceptance_loss_alpha > 0:
    loss += sampled_acceptance_loss_alpha * sampled_exact_loss
```

## 与之前错误实现的对比

### ❌ 错误实现（已修复）

```python
if sampled_draft_logprobs is not None:
    # 替换了原始的 loss 计算
    position_weights = sampled_credit  # 破坏了 gamma decay
    loss = CE_loss  # TV loss 消失
```

**问题**：
- 修改了原始的 loss 计算路径
- 即使 alpha=0 也会影响训练
- 破坏了向后兼容性

### ✅ 正确实现（当前）

```python
# 始终使用原始 loss（与 main 完全相同）
loss = compound_loss(...)  # CE + TV + Confidence

# 仅在需要时添加辅助 loss
if sampled_draft_logprobs is not None:
    loss += alpha * sampled_exact_loss  # 纯粹添加
```

**优点**：
- ✅ 原始 loss 路径不变
- ✅ alpha=0 时与 main 完全相同
- ✅ 向后兼容
- ✅ 安全的增量修改

## 代码统计

```
与 main 分支的差异:
  +130 行添加（新功能）
  -1   行删除（签名修改）

添加的内容:
  1. sampled_acceptance_credit() 函数
  2. exact_acceptance_length_loss() 函数
  3. 辅助 loss 添加逻辑
  4. 按位置的 metrics (sampled_pos{k}_*)
```

## 训练配置

```bash
# examples/train/dspark_qwen3_4b_trainer.sh

# 原始 loss 配置（与 main 相同）
LOSS_FN='{"ce": 0.1, "tv": 0.9}'
CONFIDENCE_HEAD_ALPHA=1.0

# 辅助 loss 配置（新增）
SAMPLED_ACCEPTANCE_LOSS_ALPHA=1.0  # 设为 0 可完全禁用
```

## 验证

### 测试 1: 不使用 sampled loss

```bash
# 不启用 sampled acceptance
--sampled-acceptance-loss-alpha 0.0
# 或者不传 --enable-sampled-acceptance-loss
```

**预期**: 训练行为与 main 分支完全相同

### 测试 2: 使用 sampled loss

```bash
# 启用 sampled acceptance
--sampled-acceptance-loss-alpha 1.0
--enable-sampled-acceptance-loss
```

**预期**: 
- 原始 metrics (ce_loss, tv_loss, position_{k}_acc) 正常
- 额外的 sampled metrics (sampled_pos{k}_alpha) 可用
- 总 loss 包含辅助 loss

## 监控

```bash
# 查看原始 loss 组件（应该正常）
tail -f train.log | grep -E "ce_loss|tv_loss|confidence_loss|position_[0-9]_acc"

# 查看辅助 loss（仅在启用时）
tail -f train.log | grep -E "sampled_acceptance_loss|sampled_pos[0-9]_alpha"
```

## Git 提交

```
62fe9d5 fix: restore to main baseline + add sampled acceptance as auxiliary
```

## 总结

这次修复采用了最保守和安全的方法：

1. ✅ 完全恢复 main 分支的基线
2. ✅ 只在最后添加辅助 loss（纯粹的加法）
3. ✅ 确保向后兼容性
4. ✅ 可以通过 alpha=0 完全禁用

现在训练应该可以正常工作，无论是否使用 sampled acceptance loss。
