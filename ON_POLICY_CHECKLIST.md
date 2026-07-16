# On-Policy 训练上线 Checklist

## 刚改的三处(解决两个硬阻塞)

### 1. server 脚本:关 prefix caching(修复"训不动")
**文件**:`examples/train/dspark_qwen3_4b_server.sh`  
**改动**:`VLLM_EXTRA_ARGS` 加 `--no-enable-prefix-caching`

**根因**:`ExampleHiddenStatesConnector` 只对本次 forward 真正计算的 token dump hidden states。prefix caching 开启后,cache 命中的前缀被跳过 → dump 行数 < prompt token 数(甚至 0)→ `check_hidden_states` 抛 ValueError → dataloader 吞掉异常、样本被静默丢弃 → 训练卡在 `0/5573, 0 it/s`。

关掉后所有 token 位置都重算并 dump,返回完整 hidden states。

### 2. trainer 脚本:切到 on-policy loss + 温度
**文件**:`examples/train/dspark_qwen3_4b_trainer.sh`  
**改动**:
- `LOSS_FN='{"ce": 0.1, "accept_length": 0.9}'`(原 `{"ce":0.1,"tv":0.9}`)→ 触发 `onpolicy_sampling=True`,激活 draft rollout + vLLM 打分 + exact acceptance-length loss。
- 新增 `SAMPLING_TEMPERATURE=1.0` + `--sampling-temperature` 传参。
- 修了两个 shell bug:缺续行 `\` 导致 torchrun 前台运行;缺右引号导致 `bash -n` EOF 报错。

### 3. core.py:阻止 dynamo 追踪 on-policy 方法(修复"一跑就崩")
**文件**:`src/speculators/models/utils.py`、`src/speculators/models/dspark/core.py`  
**改动**:
- `utils.py` 新增 `disable_dynamo` 装饰器(Dynamo graph-break,让方法强制 eager 执行)。
- `core.py` 的 `_sample_block_rollout` 和 `_onpolicy_forward` 加 `@disable_dynamo`。

**根因**:Ascend 上 `torch.cuda.is_available()` 返回 True → `@conditional_torch_compile` 编译 `forward` → Dynamo 尝试追踪 on-policy 方法,但里面有 Python 采样循环 + **阻塞式 vLLM RPC + 文件读取**(不可追踪)→ 大概率 graph-break 失败或直接报错。现在显式 disable,TF 路径保持编译,on-policy 路径 eager 执行。

---

## 起之前的三个必查项

### A. 单元测试(CPU,几秒)
```
pytest tests/unit/models/test_dspark_accept_length.py tests/unit/models/test_dspark_onpolicy.py -xvs
```
挡住形状/接线/数学错误。全过再继续。

### B. 层配置自洽(已经对了,但确认一次)
```
# server 
TARGET_LAYER_IDS="1 9 17 25 33"  # 5层 + launch_vllm自动追加第36层 → dump 6层
VLLM_EXTRA_ARGS 里没有 --no-include-last-layer

# trainer
TARGET_LAYER_IDS="1 9 17 25 33"  # 同上
NUM_LAYERS=5  # draft fc 吃 [:, :-1] = 5层; [:, -1] = 第36层(verifier_last)

# 6 = 5 + 1 ✓
```

### C. vocab mapping 文件存在
trainer 脚本用 `DRAFT_VOCAB_SIZE=32000`,需要 `d2t.npy` / `t2d.npy` 或 `token_freq.pt` 在 `ARROW_DIR` 里,否则退化到全 151936 词表(静默 fallback)。检查:
```
ls $ARROW_DIR/{d2t,t2d}.npy $ARROW_DIR/token_freq.pt
```
若都不存在,跑 `build_vocab_mapping.py` 或统计词频后重建。

---

## 首跑姿势(逐步放大)

1. **缩小 MAX_ANCHORS**(从 512 → 32/64)先跑几十步,验证:
   - loss 有限(不是 NaN)
   - `accept_length_loss`、`accept_len` 两个 metric 有值且合理(0.x ~ K-1)
   - 进度条在动,`Failed to load/cache hidden states` 警告大幅减少或消失(若仍密集报,说明 prefix caching 还没真的关掉)
   - 显存不炸、vLLM 队列不堵死(512 打分请求/step 是压测项)

2. **盯 `/tmp/hidden_states` 或 `shared_storage_path` 涨速**:我的 scorer 只读文件不删,每 step 512 文件落盘、长训会撑爆磁盘。这是运维项,不是正确性 bug,但得加定期清理(cron / tmpfs)。

3. **逐步放大 MAX_ANCHORS**(32→64→128→256→512),盯每级的 step 时间 + vLLM 队列深度,确认 512 能扛再跑长训。

---

## 已知约束 / 后续优化点

1. **prefix caching 必须关**:这不是临时措施。connector 的 dump 机制与 APC 根本不兼容(APC 跳过前缀 → dump 残缺)。gold hidden-state 抽取和 on-policy 打分**都要关**。性能代价:gold 前缀每步重算,vLLM 打分成本比原估计高(节点4"开缓存"的判断作废)。

2. **torch.compile 已处理**:on-policy 方法强制 eager,TF 路径保持编译。NPU 上的 `Cannot create tensor with internal format` warning 无害(性能提示,不影响正确性),可忽略。

3. **confidence BCE 已改用 C_t 加权**(之前你提的;三项 CE/accept_length/confidence 统一用 C_t)。C_t 量级可到 ~K,`confidence_head_alpha` 的有效尺度会变,首跑留意该项占比,必要时调 alpha。

---

## 报错类型 / 快速诊断

- **`Sequence length of hidden states N doesn't match num tokens M`**(密集):prefix caching 没真关掉,重查 server 的 `VLLM_EXTRA_ARGS` 或 launch_vllm 有没有显式 override 回去。
- **Dynamo / compile 相关错(graph break failure、unsupported op)**:说明 `@disable_dynamo` 没生效,检查 utils.py 的装饰器逻辑和 torch 版本(fallback 到 identity 是否工作)。
- **NaN loss / inf gradient**:on-policy 路径的数学错误,回退到单测(test_dspark_accept_length.py 的 K=1 unbiasedness、test_dspark_onpolicy.py 的 rollout grad)定位。
- **vLLM timeout / queue full**:MAX_ANCHORS 过大 + 每 step 512 打分请求超过 vLLM 吞吐,缩小 anchors 或加 vLLM 并发。

---

## 验证成功的标志

1. 进度条正常前进,`it/s` 不为 0。
2. tensorboard 里 `accept_length_loss` / `ce_loss` / `confidence_loss` 三条曲线都有值、都收敛。
3. `accept_len` metric 在 [0, K-1] 范围,随训练上升(draft 变准 → 期望接受长度增加)。
4. 第一个 checkpoint 能存、能 resume(FSDP2 不会因为 `verifier_scorer` 非 nn.Module 属性报错)。

达到这四条,on-policy 链路就通了。后续是超参调优 / 压测吞吐 / 对比基线的事。
