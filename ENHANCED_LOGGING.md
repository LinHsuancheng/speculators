# Enhanced Logging for Sample Acceptance Length Training

## 概述

在 `detection` 分支上，我们为 sample acceptance length targeting loss 训练添加了详细的监控指标。这些指标帮助你深入理解训练过程，包括 draft/target 概率、credit 分配、acceptance ratio 等关键信息。

## 修改的文件

1. **`src/speculators/models/dspark/metrics.py`**
   - 在 `compute_metrics()` 函数中添加了 30+ 个新的 metrics
   - 所有新 metrics 都在 `sampled_exact_loss is not None` 条件下计算

2. **`src/speculators/train/sampled_acceptance.py`**
   - 在 `SampledAcceptanceAugmentor.__call__()` 中添加了实时采样统计日志
   - 每个 batch 采样时输出关键统计信息

## 新增的 Metrics

### 1. Credit 统计 (C_t)

根据 trials.md §7-9，credit 是 `C_t = 1[q_t < p_t] * sum_{k>=t} S_k`

```python
sampled_credit_mean      # 平均 credit (应该 > 0)
sampled_credit_min       # 最小 credit
sampled_credit_max       # 最大 credit
sampled_credit_std       # Credit 标准差 (衡量变异性)
```

**如何解读**:
- `credit_mean > 0`: Draft model 在某些位置 undercovered (q < p)
- `credit_mean ≈ 0`: Draft model 完全匹配或 overcovered target
- 训练中 `credit_mean` 应该随时间变化，反映 draft 学习进度

### 2. Alpha (接受概率) 统计

Alpha 是 `α_i = min(1, p_i(Y_i) / q_i(Y_i))`，表示单个 token 的接受概率。

```python
sampled_alpha_mean       # 平均接受概率 (应该逐渐增加)
sampled_alpha_min        # 最小接受概率 (最差位置)
sampled_alpha_max        # 最大接受概率 (应该是 1.0)
```

**如何解读**:
- `alpha_mean` 接近 1.0: Draft 和 target 对齐良好
- `alpha_mean < 0.5`: Draft 对齐很差，需要继续训练
- `alpha_min` 显示最难预测的位置

### 3. Draft Log-Probabilities (log q_t(Y_t))

Draft model 对实际采样 token 的 log 概率。

```python
sampled_draft_logp_mean  # 平均 draft log-prob
sampled_draft_logp_min   # 最低 draft log-prob (最不自信)
sampled_draft_logp_max   # 最高 draft log-prob (最自信)
```

**如何解读**:
- 更高的 log-prob (更接近 0) = 更自信的预测
- `draft_logp_mean` 应该在训练中增加
- 过低的 `draft_logp_min` 表示某些 token 很难预测

### 4. Target Log-Probabilities (log p_t(Y_t))

Target model (verifier) 对相同采样 token 的 log 概率。

```python
sampled_target_logp_mean # 平均 target log-prob
sampled_target_logp_min  # 最低 target log-prob
sampled_target_logp_max  # 最高 target log-prob
```

**如何解读**:
- Target 是冻结的，这些值主要用于对比
- `target_logp_mean` vs `draft_logp_mean` 的差距衡量对齐程度

### 5. Log-Probability Ratio (log(p/q))

比值 `log(p_t/q_t) = log p_t - log q_t`

```python
sampled_logp_ratio_mean  # 平均 log(p/q)
sampled_logp_ratio_min   # 最小 log(p/q) (overcovered: q >> p)
sampled_logp_ratio_max   # 最大 log(p/q) (undercovered: p >> q)
```

**如何解读**:
- `logp_ratio > 0`: Target 比 draft 更确信这个 token (undercovered)
- `logp_ratio < 0`: Draft 比 target 更确信 (overcovered)
- `logp_ratio_mean ≈ 0`: 平均而言两者对齐

### 6. Coverage 统计

```python
sampled_undercovered_ratio   # q < p 的 token 比例
sampled_overcovered_ratio    # q >= p 的 token 比例
```

**如何解读**:
- `undercovered_ratio` 高: Draft 普遍低估概率 (保守)
- `overcovered_ratio` 高: Draft 普遍高估概率 (过度自信)
- 理想情况: 两者平衡

### 7. Survival 概率

Survival `S_k = prod_{i<=k} alpha_i` 是前 k 个 token 全部被接受的概率。

```python
sampled_survival_mean         # 所有位置的平均 survival
sampled_final_survival_mean   # 最后位置的平均 survival (整个 block)
```

**如何解读**:
- `final_survival` 是整个 block 被接受的概率
- 应该随训练增加
- 接近 0: Block 很难完全接受
- 接近 1: Block 大概率完全接受

### 8. 估计的 Acceptance Length

根据 trials.md §3-4，期望接受长度是 `sum_{k=1}^K S_k`

```python
sampled_estimated_accept_len  # 每个 block 的估计接受长度
```

**如何解读**:
- 这是**直接估计**的期望接受长度
- 训练的主要目标就是最大化这个值
- 例如 K=7 时，理想值接近 7，最差值接近 0

### 9. 实时采样日志

在 `SampledAcceptanceAugmentor` 中，每次采样都会输出：

```
[SampledAcceptance] anchor_pos=512, sampled_len=7, 
                    draft_logp_mean=-3.21, draft_logp_min=-5.87, 
                    target_logp_mean=-3.02, target_logp_min=-5.23, 
                    alpha_mean=0.82, alpha_min=0.45, 
                    final_survival=0.23, undercovered=5/7
```

**参数说明**:
- `anchor_pos`: 采样的锚点位置
- `sampled_len`: 采样的 draft token 数量
- `undercovered=5/7`: 7 个 token 中有 5 个是 undercovered (q < p)

## 监控建议

### 关键指标组合

1. **训练健康度**:
   ```
   sampled_credit_mean > 0.5
   sampled_alpha_mean > 0.7
   sampled_estimated_accept_len 逐渐增加
   ```

2. **对齐质量**:
   ```
   sampled_logp_ratio_mean 接近 0
   sampled_undercovered_ratio 在 0.3-0.7 之间
   sampled_alpha_mean 接近 1.0
   ```

3. **训练进度**:
   ```
   sampled_estimated_accept_len 从 ~1.5 增加到 ~3-4
   sampled_final_survival 从 ~0.1 增加到 ~0.3-0.5
   sampled_credit_std 减少 (更稳定的 credit 分配)
   ```

### 警告信号

⚠️ **需要调查的情况**:

1. `sampled_credit_mean < 0.1`: Draft 没有学习或过度自信
2. `sampled_alpha_mean < 0.5`: 对齐很差，可能需要调整学习率
3. `sampled_estimated_accept_len < 1.0`: 基本没有 token 被接受
4. `sampled_undercovered_ratio` 接近 0 或 1: 严重不平衡
5. `sampled_acceptance_loss` 不下降: 优化可能有问题

## 日志输出位置

所有 metrics 通过 `metric_logger.info()` 输出，格式为：

```json
{
  "train": {
    "loss": 2.345,
    "sampled_acceptance_loss": -1.234,
    "sampled_credit_mean": 0.823,
    "sampled_alpha_mean": 0.765,
    ...
  },
  "epoch": 1,
  "lr": 0.0006,
  "global_step": 100
}
```

日志文件位置（根据 training script）:
```
$OUTPUT_DIR/logs/train_<timestamp>.log
```

## 使用示例

### 训练时实时监控

```bash
# 启动训练
bash examples/train/dspark_qwen3_4b_trainer.sh

# 在另一个终端监控关键指标
tail -f outputs/dspark_qwen3_4b_test_only/logs/train_*.log | grep "sampled_"
```

### 提取关键指标

```bash
# 提取 credit 统计
grep "sampled_credit_mean" train.log | tail -20

# 提取 acceptance length
grep "sampled_estimated_accept_len" train.log | tail -20

# 查看实时采样日志
grep "\[SampledAcceptance\]" train.log | tail -10
```

## 验证脚本

运行验证脚本查看所有指标说明：

```bash
python script/check_enhanced_metrics.py
```

## 与 trials.md 的对应关系

| Metric | trials.md 公式 | 说明 |
|--------|---------------|------|
| `sampled_alpha_mean` | §2: `α_i = min(1, p_i/q_i)` | 单步接受率 |
| `sampled_survival_mean` | §4: `S_k = prod_{i<=k} α_i` | 前缀存活概率 |
| `sampled_credit_mean` | §7-9: `C_t = I_t * sum_{k>=t} S_k` | 精确 credit |
| `sampled_estimated_accept_len` | §3: `E[L] = sum_{k=1}^K S_k` | 期望接受长度 |
| `sampled_acceptance_loss` | §9, §13: `-sum_t C_t log q_t(Y_t) / K` | Monte Carlo loss |

## 总结

通过这些增强的日志，你可以：

1. ✅ 实时监控训练进度和健康度
2. ✅ 验证 loss 计算是否正确
3. ✅ 诊断训练问题（overcovered/undercovered）
4. ✅ 调整超参数（学习率、loss 权重）
5. ✅ 对比不同实验的效果

所有指标都经过理论验证，与 trials.md 中的数学推导完全一致。
