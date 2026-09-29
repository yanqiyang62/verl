# V4 session GRPO baseline

统一维护入口：[V4 docs](../../../data/recipe_v4_session_agentloop/docs/README.md) · [启动与续跑](../../../data/recipe_v4_session_agentloop/docs/training.md)。

默认续跑最近保存的同策略 V4 完整 checkpoint：

```bash
bash tasks/q38_27B_session_v4_grpo.sh
```

凭据为 `ARK_API_KEY` 和 `SWANLAB_API_KEY`。当前本地脚本开头已有这两个变量的赋值，使用终端 export 时需检查脚本是否覆盖它们；文档不复制真实密钥。

续跑同一 V4 GRPO 实验（恢复模型、优化器、调度器/RNG、dataloader）：

```bash
bash tasks/q38_27B_session_v4_grpo.sh --resume /path/to/checkpoints/global_step_55
```

也可传 `checkpoints` 父目录。复用已有完整性检查、独立输出目录和最多两次进程重启；
重启保持 V4 GRPO，不切回 V3 或 DAPO。`RESUME_MAX_RESTARTS=0` 可关闭自动重启。

默认读取 `/shared/users/yangyq/data/recipe_v4_session_agentloop`，调用原 GRPO session
example。保持 clip 0.2/0.2、KL loss 0.001、token-mean、学习率 3e-6、LoRA 32/64。
同分组过滤关闭，不可评分组整体排除，最多补采一轮；仍有 2–4 个完整有效组时直接更新，
不足两组则跳过本批，不推进 optimizer、LR scheduler 或 global step。
4 输入 × 4 rollout、micro batch 2、vLLM 并发上限 16、显存预算 0.55；每五步保存
LoRA 权重及续训状态。默认一轮全量训练；未自动限制到 20–30 step。

Judge 和 tool mock 都固定到 `ep-20260513201344-xl7ph`（豆包 2.0）、thinking enabled、
effort low；关闭逐工具和逐用户审查。

SwanLab 项目默认 `GRPO-Qwen3.8`，新 run 名含 `V4-GRPO-Doubao2.0-low` 和北京时间。
复用 V4 sampler 的 `training/tool/*` 指标：平均调用数、平均重复数、重复调用比例、
含重复轨迹比例、裁判认定的无必要操作均值、效率均值和统计轨迹数。
只统计入选的可评分轨迹，不增加裁判调用；历史 run 不回填。

```bash
# 只预览，不启动训练或请求付费 API
DRY_RUN=1 bash tasks/q38_27B_session_v4_grpo.sh
```

## 自动续跑与从头开始

```bash
# 自动查找最近保存的完整 V4 GRPO checkpoint 并续跑
bash tasks/q38_27B_session_v4_grpo.sh
# 明确开启一轮全新的训练
bash tasks/q38_27B_session_v4_grpo.sh --refresh
# 指定旧实验/更高步数的 checkpoint，覆盖自动选择
bash tasks/q38_27B_session_v4_grpo.sh --resume /path/to/checkpoints/global_step_25
```

按 checkpoint 完成标记的保存时间选择，**不是按最大的 global step 选择**。
只匹配同策略、相同 MODEL_PATH、RECIPE_ROOT、SESSION_PROFILE；先检查单卡模型、
optimizer、extra、data.pt、LoRA 元数据是否齐全。最新一轮无完整保存点时向前查找；
没有任何匹配保存点则报错退出，不会从零开始。`--refresh` 与 `--resume` 互斥。
默认查找 `logs/$PROJECT_NAME`（可用 `AUTO_RESUME_LOG_ROOT` 改变检索目录），
也查找续跑控制器登记的独立输出目录，包含自定义 `RESUME_OUTPUT_DIR`。
续跑保留原 checkpoint，默认另建输出目录，并最多自动重启两次。
进程重启次数、API 的最多 2 次尝试和 sampler 的最多 1 轮补采是独立限制；此改动不增加 API 重试预算。
`--refresh` 启动仍使用原直接训练流程，不启用续跑控制器的进程自动重启。

### 有界采样恢复（V4）

- Judge 单次 HTTP 超时最多 120 秒，整个 judge 调用（含锁等待和最多一次修复）等待预算 180 秒。
  总预算耗尽会返回不可评分，取消信号阻止后台线程继续发起下一次请求；已发出的 HTTP 请求不能保证撤销计费。
- 缺失/无效 evidence 仅记录告警；影响评分的字段仍校验。不可评分的 prompt 组整体排除。
- 当前 4 prompt × n=4：补采最多一轮后，有 2/3/4 个完整组就分别使用 8/12/16 条真实轨迹。
  verl 的短占位序列只用于 mini-batch 整除，response/loss mask 为零，独立 UID，不参与真实轨迹指标。
  仅支持 sync、parameter_sync_step=1、GRPO 优势、token-mean、同步 checkpoint，且 train_batch_size 不大于 actor mini batch。
- 有效组不足两组时跳过本批并读取后续数据，不更新模型、优化器、学习率或 global step。
  连续三批不可用会保存最后完成 step 的模型/optimizer/extra/dataloader，写入 `CHECKPOINT_DIR/session_pause.json` 后正常退出。
  正常退出不会触发续跑控制脚本的崩溃自动重试，不会在服务故障时无限发起付费调用。
- 未结束的 worker 超过采样看门狗时直接进入保存暂停流程，先终止 worker 和在途生成，避免迟到写入混进下一批。
- `CHECKPOINT_DIR/session_skips.jsonl` 记录每次跳过，checkpoint 内的 `session_sampling_state.json` 保存累计跳过数。
  手动续跑恢复累计数和数据位置，但把连续失败计数归零。暂停发生在首次更新前也会保存 `global_step_0`。
- 新增 SwanLab `training/session/selected_groups`、`selected_trajectories`、`partial_batch`、`skipped_batches_total`、
  `consecutive_skips`、`paused`、`judge_requests`、`judge_retries`；judge 数量来自已写回轨迹（包括被丢弃的轨迹，缓存命中算 0），不代替供应商账单。
- 下次启动或续跑生效；运行中的 Ray worker 不会热更新。暂停后检查服务再运行原脚本（默认从最新完整 checkpoint 续跑）。
