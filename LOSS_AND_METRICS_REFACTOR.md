# 日志和 Loss 修改说明

## 修改时间
2026-07-17

## 修改内容

### 1. 恢复原始 Loss 计算

**修改前**:
- 当提供 `sampled_draft_logprobs` 时，完全替换为使用 sampled_credit 作为 position_weights
- CE loss 使用 sampled_credit 权重，禁用 gamma decay

**修改后**:
- **始终使用原始的 loss 计算**（gamma decay + compound loss）
- Sampled exact loss 作为**辅助 loss**添加到总 loss
- CE 和 confidence loss 继续使用原来的 gamma decay 权重

```python
# 原始 loss（gamma decay + compound loss）
loss, term_losses = compound_loss(
    logits, targets, loss_mask, pos_idx,
    loss_config=loss_config,
    decay_fn=partial(dflash_loss_decay, gamma=gamma),
    position_weights=cat_weights,
)

# 辅助 loss
if sampled_exact_loss is not None:
    loss = loss + sampled_acceptance_loss_alpha * sampled_exact_loss
```

**好处**:
- ✅ 避免了 CE/confidence loss 被极小的 sampled_credit 压制
- ✅ 打破了"draft 差 → survival ≈ 0 → loss ≈ 0 → 无梯度 → draft 永远差"的死锁
- ✅ 模型可以通过原始 CE loss warm up
- ✅ Exact loss 作为辅助信号逐渐引导优化

### 2. 修改 Metrics：从 min/max/mean 改为按位置 (position_k)

**修改前**:
- `sampled_credit_mean/min/max/std`
- `sampled_alpha_mean/min/max`
- `sampled_draft_logp_mean/min/max`
- `sampled_target_logp_mean/min/max`
- `sampled_logp_ratio_mean/min/max`
- `sampled_undercovered_ratio`
- `sampled_survival_mean`
- 等等...

**修改后**:
对每个位置 k（1 到 K）单独记录：

```
sampled_pos1_draft_logp       # 位置 1 的 draft log-prob
sampled_pos1_target_logp      # 位置 1 的 target log-prob
sampled_pos1_alpha            # 位置 1 的 acceptance ratio
sampled_pos1_survival         # 位置 1 的 survival S_1
sampled_pos1_credit           # 位置 1 的 credit C_1
sampled_pos1_undercovered     # 位置 1 是否 undercovered
sampled_pos1_logp_ratio       # 位置 1 的 log(p/q)

sampled_pos2_draft_logp       # 位置 2 的 draft log-prob
sampled_pos2_target_logp      # 位置 2 的 target log-prob
...
sampled_posK_*                # 位置 K 的所有统计
```

**保留的聚合指标**:
- `sampled_estimated_accept_len`: 期望接受长度
- `sampled_first_alpha`: 第一个位置的 alpha（最关键）

**好处**:
- ✅ 可以直接看到**第一个位置是否总是最差**
- ✅ 可以看到 survival 在哪个位置塌缩
- ✅ 可以诊断 prefix/shift 错位问题
- ✅ 强相关性（与位置）得到正确反映

### 3. 实时采样日志保持不变

`SampledAcceptanceAugmentor` 中的实时日志保持原样：

```
[SampledAcceptance] anchor_pos=512, sampled_len=7, 
                    draft_logp_mean=-3.21, draft_logp_min=-5.87, 
                    target_logp_mean=-3.02, target_logp_min=-5.23, 
                    alpha_mean=0.82, alpha_min=0.45, 
                    final_survival=0.23, undercovered=5/7
```

## 如何使用新的 Metrics

### 查看所有位置的 alpha

```bash
tail -f train.log | grep -E "sampled_pos[0-9]+_alpha"
```

输出示例：
```
sampled_pos1_alpha: 2.93e-11  ← 第一个位置极小！
sampled_pos2_alpha: 0.8234
sampled_pos3_alpha: 0.9123
sampled_pos4_alpha: 0.8567
```

### 检查第一个位置是否总是最差

```bash
# 提取第一个位置的 alpha
grep "sampled_pos1_alpha:" train.log | tail -20

# 对比其他位置
grep "sampled_pos2_alpha:" train.log | tail -20
```

如果 `pos1_alpha` 总是比其他位置小 10 个数量级，说明**第一个位置的 target scoring 有问题**。

### 查看 survival 在哪个位置塌缩

```bash
tail -f train.log | grep -E "sampled_pos[0-9]+_survival"
```

输出示例：
```
sampled_pos1_survival: 2.93e-11  ← 从这里开始塌缩
sampled_pos2_survival: 2.41e-11  ← 后续无法恢复
sampled_pos3_survival: 2.06e-11
```

### 查看哪些位置是 undercovered

```bash
tail -f train.log | grep -E "sampled_pos[0-9]+_undercovered"
```

输出示例：
```
sampled_pos1_undercovered: 0.0    ← 第一个位置没有 undercovered
sampled_pos2_undercovered: 0.2
sampled_pos3_undercovered: 0.8    ← 大部分 undercovered
```

### 监控原始 CE loss 和辅助 exact loss

```bash
tail -f train.log | grep -E "ce_loss|sampled_acceptance_loss"
```

输出示例：
```
ce_loss: 2.345                    ← 原始 CE loss 正常
sampled_acceptance_loss: -1.5e-30 ← 辅助 loss 极小（符合预期）
loss: 2.346                       ← 总 loss 主要由 CE 主导
```

## 预期行为

### 训练初期（Draft 接近随机）

```
ce_loss: ~3.5                     ← CE loss 主导训练
sampled_acceptance_loss: ~-1e-30  ← Exact loss 极小但不影响训练
confidence_loss: ~0.5             ← Confidence loss 正常

sampled_pos1_alpha: ~1e-11        ← 第一个位置极差
sampled_pos2_alpha: ~1e-10
sampled_estimated_accept_len: ~1e-11

loss: ~4.0                        ← 总 loss 由原始 CE 主导
```

**关键**: 即使 exact loss 接近 0，原始 CE loss 仍然提供梯度，模型可以 warm up。

### 训练中期（Draft 逐渐改善）

```
ce_loss: ~2.0                     ← CE loss 下降
sampled_acceptance_loss: ~-0.5    ← Exact loss 逐渐增大
confidence_loss: ~0.3

sampled_pos1_alpha: ~0.3          ← 第一个位置改善
sampled_pos2_alpha: ~0.5
sampled_estimated_accept_len: ~1.5

loss: ~2.3                        ← 总 loss 继续下降
```

### 训练后期（Draft 对齐良好）

```
ce_loss: ~1.2
sampled_acceptance_loss: ~-2.0    ← Exact loss 显著贡献
confidence_loss: ~0.2

sampled_pos1_alpha: ~0.8          ← 所有位置都改善
sampled_pos2_alpha: ~0.9
sampled_estimated_accept_len: ~3.5

loss: ~1.4
```

## 诊断清单

使用新 metrics 诊断问题：

### 问题 1: 第一个位置的 alpha 总是极小

**症状**:
```
sampled_pos1_alpha: ~1e-11
sampled_pos2_alpha: ~0.8
sampled_pos3_alpha: ~0.9
```

**诊断**: 第一个 target position 的 prefix 或 causal shift 错位

**下一步**: 
- 检查 `score_sampled_tokens()` 中的 prefix 构造
- 确认是否包含/不包含 anchor token
- 确认 logits gather 索引是否正确

### 问题 2: CE loss 正常但 exact loss 始终接近 0

**症状**:
```
ce_loss: 2.5
sampled_acceptance_loss: -1e-30
所有 sampled_pos*_alpha: ~1e-10
```

**诊断**: Draft 和 target 严重不对齐，但原始 CE loss 正在工作

**下一步**:
- 继续训练，让 CE loss 先 warm up
- 检查 draft 和 target 是否使用相同的 tokenizer/processor
- 运行 target 对齐单元测试（gold path）

### 问题 3: 所有 loss 都正常，但 pos1 仍然比其他位置差

**症状**:
```
ce_loss: 1.5
sampled_pos1_alpha: 0.3
sampled_pos2_alpha: 0.8
sampled_pos3_alpha: 0.9
```

**诊断**: 第一个位置确实更难（可能正常），或有轻微错位

**下一步**:
- 检查原始 acceptance rate 的 position_1 acc
- 对比 `sampled_pos1_*` 和 teacher-forced 的 `position_1_acc`

## Git 提交

```bash
git add src/speculators/models/dspark/metrics.py
git commit -m "Refactor: restore original loss and add position-wise metrics

Changes:
1. Restore original loss calculation (gamma decay + compound loss)
   - Sampled exact loss is now an auxiliary loss
   - Fixes the dead-lock where survival ≈ 0 kills all gradients

2. Replace min/max/mean metrics with position-wise metrics
   - sampled_pos{k}_draft_logp/target_logp/alpha/survival/credit
   - Enables direct diagnosis of position-specific issues
   - Can identify if first position alpha is always worst

Benefits:
- Original CE loss continues to provide gradients for warm-up
- Position-wise metrics reveal if first position has prefix/shift issues
- Exact loss serves as auxiliary signal without blocking training"
```

## 总结

这次修改解决了两个关键问题：

1. **避免训练死锁**: 原始 CE loss 继续工作，exact loss 作为辅助
2. **精确诊断能力**: 按位置的 metrics 可以直接看到哪个位置有问题

现在可以安全地训练，同时获得详细的诊断信息。
