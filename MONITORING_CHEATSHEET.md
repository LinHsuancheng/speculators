# 快速参考卡 - 训练监控

## 关键命令

### 监控第一个位置（最关键）
```bash
tail -f train.log | grep "sampled_pos1_alpha"
```

### 对比所有位置的 alpha
```bash
tail -f train.log | grep -E "sampled_pos[1-7]_alpha"
```

### 监控 Loss 组成
```bash
tail -f train.log | grep -E "ce_loss|sampled_acceptance_loss|^loss:"
```

### 查看 Survival 塌缩
```bash
tail -f train.log | grep -E "sampled_pos[1-7]_survival"
```

### 查看哪些位置 Undercovered
```bash
tail -f train.log | grep -E "sampled_pos[1-7]_undercovered"
```

## 诊断流程图

```
训练开始
    ↓
查看 sampled_pos1_alpha 和 sampled_pos2_alpha
    ↓
    ├─ pos1_alpha < pos2_alpha * 1e-10 ?
    │   YES → 第一个位置有 prefix/shift 错位 ⚠️
    │         检查 score_sampled_tokens() 中的 prefix 构造
    │
    └─ NO → 继续检查
           ↓
       ce_loss 是否正常 (>1.0) ?
           ↓
       ├─ YES → 训练正常，等待 warm up ✓
       │         sampled_acceptance_loss 会逐渐增大
       │
       └─ NO → ce_loss 异常小 (<0.1)
               检查数据加载和模型初始化
```

## 预期数值范围

### 训练初期（随机初始化）
```
ce_loss:                  2.5 - 4.0   ✓ 正常
sampled_acceptance_loss: -1e-30       ✓ 正常（极小）
sampled_pos1_alpha:       1e-15 ~ 1e-8
sampled_pos2_alpha:       1e-15 ~ 1e-8
loss:                     2.5 - 4.0   ✓ 由 CE 主导
```

### 训练中期
```
ce_loss:                  1.5 - 2.5
sampled_acceptance_loss: -0.1 ~ -1.0
sampled_pos1_alpha:       0.1 - 0.5
sampled_pos2_alpha:       0.2 - 0.6
loss:                     1.5 - 3.0
```

### 训练后期
```
ce_loss:                  0.8 - 1.5
sampled_acceptance_loss: -1.0 ~ -3.0
sampled_pos1_alpha:       0.5 - 0.9
sampled_pos2_alpha:       0.6 - 0.95
loss:                     1.0 - 2.0
```

## 警告信号 ⚠️

| 症状 | 可能原因 | 下一步 |
|------|---------|--------|
| `pos1_alpha` 总是 `< 1e-10` 且 `pos2_alpha > 0.1` | 第一个位置 prefix/shift 错位 | 检查 `score_sampled_tokens()` |
| 所有 `pos*_alpha < 1e-10` | Draft 和 target 完全不对齐 | 检查 tokenizer/processor |
| `ce_loss < 0.1` | CE loss 被错误缩放 | 检查代码（不应该发生） |
| `loss ≈ sampled_acceptance_loss` | 原始 loss 消失 | 检查代码（不应该发生） |

## 一行监控命令

```bash
# 监控所有关键指标
tail -f train.log | grep -E "ce_loss:|sampled_pos[1-3]_(alpha|survival):|sampled_acceptance_loss:" | grep -v "_sum\|_total"
```

## 提取特定步数的数据

```bash
# 提取最近 20 步的 pos1 alpha
grep "step.*sampled_pos1_alpha" train.log | tail -20

# 绘制趋势（需要处理）
grep "sampled_pos1_alpha:" train.log | awk '{print $NF}' > pos1_alpha.txt
```

## 快速检查脚本

```bash
#!/bin/bash
# check_training.sh - 快速检查训练状态

LOG="train.log"

echo "=== 最新的 Loss 值 ==="
grep "ce_loss:" $LOG | tail -1
grep "sampled_acceptance_loss:" $LOG | tail -1
grep "^loss:" $LOG | tail -1

echo ""
echo "=== 前 3 个位置的 Alpha ==="
grep "sampled_pos1_alpha:" $LOG | tail -1
grep "sampled_pos2_alpha:" $LOG | tail -1
grep "sampled_pos3_alpha:" $LOG | tail -1

echo ""
echo "=== 诊断 ==="
POS1=$(grep "sampled_pos1_alpha:" $LOG | tail -1 | awk '{print $NF}')
POS2=$(grep "sampled_pos2_alpha:" $LOG | tail -1 | awk '{print $NF}')

if [[ $(echo "$POS1 < 1e-10" | bc -l) -eq 1 ]] && [[ $(echo "$POS2 > 0.01" | bc -l) -eq 1 ]]; then
    echo "⚠️  WARNING: pos1_alpha 异常低，可能存在 prefix/shift 错位"
elif [[ $(echo "$POS1 < 1e-10" | bc -l) -eq 1 ]] && [[ $(echo "$POS2 < 1e-10" | bc -l) -eq 1 ]]; then
    echo "ℹ️  INFO: 所有 alpha 都很低，Draft 仍在 warm up 阶段"
else
    echo "✓ 训练正常"
fi
```

## 实时仪表盘（tmux 分屏）

```bash
# 终端 1: Loss 监控
tail -f train.log | grep -E "ce_loss:|loss:"

# 终端 2: Position Alpha 监控
tail -f train.log | grep -E "sampled_pos[1-5]_alpha:"

# 终端 3: Survival 监控
tail -f train.log | grep -E "sampled_pos[1-5]_survival:"

# 终端 4: 实时采样日志
tail -f train.log | grep "\[SampledAcceptance\]"
```

## 记住

✓ **Loss 主要由 CE 驱动** - 即使 sampled_acceptance_loss ≈ 0 也正常
✓ **第一个位置最关键** - 如果 pos1_alpha 异常，优先检查
✓ **位置之间应该相近** - 如果 pos1 << pos2，说明有问题
✓ **Survival 会累积下降** - 这是正常的，但不应该从第一个位置就塌缩

---
📖 完整文档: LOSS_AND_METRICS_REFACTOR.md
