# 日志增强功能使用指南

## 已完成的工作

✅ **已同步 server 脚本**:
- `dspark_qwen3_4b_server.sh` 已同步到 main, detection, exact_acceptance_loss 三个分支
- 添加了 `--no-enable-prefix-caching --block-size 128` 修复 vLLM cache 问题

✅ **已实现 sample acceptance length targeting loss**:
- Detection 分支有完整的端到端实现
- 包括 credit 计算、loss 函数、on-policy 采样器、vLLM 交互

✅ **已添加全面的日志监控**:
- 30+ 个新的 metrics 用于监控训练
- 实时采样统计日志
- 完整的文档和测试脚本

## 快速开始

### 1. 查看所有新增的 metrics

```bash
python script/check_enhanced_metrics.py
```

这会显示所有新增 metrics 的说明和如何解读它们。

### 2. 启动训练并监控日志

```bash
# 启动训练（假设 vLLM server 已经在运行）
bash examples/train/dspark_qwen3_4b_trainer.sh

# 在另一个终端实时监控关键指标
tail -f outputs/dspark_qwen3_4b_test_only/logs/train_*.log | grep -E "sampled_(credit|alpha|estimated_accept_len|acceptance_loss)"
```

### 3. 查看实时采样日志

```bash
# 监控每个 batch 的采样统计
tail -f outputs/dspark_qwen3_4b_test_only/logs/train_*.log | grep "\[SampledAcceptance\]"
```

输出示例：
```
[SampledAcceptance] anchor_pos=512, sampled_len=7, 
                    draft_logp_mean=-3.21, draft_logp_min=-5.87, 
                    target_logp_mean=-3.02, target_logp_min=-5.23, 
                    alpha_mean=0.82, alpha_min=0.45, 
                    final_survival=0.23, undercovered=5/7
```

## 关键 Metrics 说明

### 最重要的 5 个指标

1. **`sampled_estimated_accept_len`**: 期望接受长度的直接估计
   - 这是训练的主要目标
   - 应该随时间增加（例如从 1.5 增加到 3-4）

2. **`sampled_credit_mean`**: 平均 credit 值
   - 应该 > 0（draft model 在学习）
   - 太低（< 0.1）可能表示过度自信

3. **`sampled_alpha_mean`**: 平均接受概率
   - 应该随时间增加，接近 1.0
   - < 0.5 表示对齐很差

4. **`sampled_undercovered_ratio`**: Draft 低估概率的 token 比例
   - 理想值在 0.3-0.7 之间
   - 接近 0 或 1 表示不平衡

5. **`sampled_acceptance_loss`**: 精确的 acceptance length loss
   - 应该逐渐减小（变得更负）

### 次要监控指标

- `sampled_final_survival`: 整个 block 被接受的概率
- `sampled_logp_ratio_mean`: log(p/q) 的平均值，衡量对齐程度
- `sampled_credit_std`: Credit 的标准差，衡量稳定性

## 监控脚本示例

```bash
#!/bin/bash
# monitor_training.sh - 实时监控训练关键指标

LOG_FILE="outputs/dspark_qwen3_4b_test_only/logs/train_$(date +%Y%m%d)*.log"

echo "=== Monitoring Sample Acceptance Length Training ==="
echo ""

# 监控主要指标
tail -f $LOG_FILE | while read line; do
    if echo "$line" | grep -q "sampled_estimated_accept_len"; then
        echo "[ACCEPT_LEN] $(echo $line | grep -oP 'sampled_estimated_accept_len[^,]+')"
    fi
    if echo "$line" | grep -q "sampled_alpha_mean"; then
        echo "[ALPHA] $(echo $line | grep -oP 'sampled_alpha_mean[^,]+')"
    fi
    if echo "$line" | grep -q "sampled_credit_mean"; then
        echo "[CREDIT] $(echo $line | grep -oP 'sampled_credit_mean[^,]+')"
    fi
    if echo "$line" | grep -q "\[SampledAcceptance\]"; then
        echo "[SAMPLE] $line"
    fi
done
```

## 问题排查

### 如果看不到 sampled_* metrics

1. 检查是否启用了 `--enable-sampled-acceptance-loss`
2. 确认 vLLM server 正在运行
3. 查看是否有 "Skipping sampled acceptance loss" 警告

### 如果 metrics 值异常

- `sampled_credit_mean` 一直为 0: Draft 可能过度自信，检查学习率
- `sampled_alpha_mean` 不增加: 可能需要调整 loss 权重或学习率
- `sampled_estimated_accept_len` 不增加: 检查 loss 是否正确反向传播

### 日志文件位置

训练日志默认输出到:
```
$OUTPUT_DIR/logs/train_<timestamp>.log
```

其中 `$OUTPUT_DIR` 在 trainer script 中设置（默认 `./outputs/dspark_qwen3_4b_test_only`）

## 更多信息

- 详细文档: `ENHANCED_LOGGING.md`
- 数学推导: `/mnt/c/Users/33301/Documents/Huawei Intern Notes/trials.md`
- 实现说明: `.codex/AGENT.md`

## Git 提交信息

```bash
# 查看最新的日志增强提交
git log -1 --stat

# 查看所有相关提交
git log --oneline --grep="logging\|metrics" | head -10
```

## 下一步

如果需要进一步定制日志:

1. 修改 `src/speculators/models/dspark/metrics.py` 添加新的 metrics
2. 修改 `src/speculators/train/sampled_acceptance.py` 添加采样日志
3. 运行 `script/test_enhanced_metrics.py` 验证计算正确性
4. 更新 `ENHANCED_LOGGING.md` 文档

## 总结

你现在可以通过详细的日志全面监控 sample acceptance length training 的训练过程，包括:

✅ Credit 分配（C_t）
✅ 接受概率（α_i）
✅ Draft/Target 概率比较（log p_t, log q_t）
✅ Survival 概率（S_k）
✅ 期望接受长度估计
✅ Coverage 统计（undercovered/overcovered）
✅ 实时采样统计

所有指标都与 trials.md 的数学推导完全一致！
