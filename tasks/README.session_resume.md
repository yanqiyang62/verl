# Session 训练与续跑

统一使用 `q38_27B_session_throughput.sh`。不加 `--resume` 从头训练：

```bash
bash tasks/q38_27B_session_throughput.sh
```

在已分配的单张 B300 环境中，加 `--resume` 完整恢复 actor、优化器、学习率调度器、随机数状态及 dataloader：

```bash
bash tasks/q38_27B_session_throughput.sh --resume /绝对路径/checkpoints/global_step_65
```

也可以传 `checkpoints` 父目录；脚本读取 `latest_checkpointed_iteration.txt`，缺失或状态文件不完整会报错，不会退回从头训练。不要把 checkpoint 填到 `MODEL_PATH`：那只更换模型初始化来源，不能恢复训练状态。

`--resume=路径` 同样支持，其他 Hydra 参数照常传递，例如 `trainer.save_freq=5`。旧的 `q38_27B_session_resume.sh 路径` 调用方式继续兼容。

2026-09-21 20:35 启动的训练完成到 step 69，最近完整 checkpoint 是 step 65：

```bash
bash tasks/q38_27B_session_throughput.sh --resume \
  /shared/users/yangyq/projects/verl/logs/GRPO-Qwen3.8/Qwen3.8-27B-LoRA-FA4-B300-Session-Throughput-full-20260921_203518_BJT/20260921_123518Z_0JHwkz/checkpoints/global_step_65
```

从 step 66 开始，66–69 的未保存更新需要重算。保持 4 个 prompt × 4 条 rollout、LoRA rank 32 / alpha 64、原 base model，默认每 5 步保存。API 设置继承现有 session task。修改 batch 或数据配置会改变续训语义，不建议在恢复时调整。

新 checkpoint 默认保存到 `checkpoints/session-resume-65-<北京时间>-<pid>`，原 checkpoint 不受影响。新日志、rollout 和 SwanLab 使用带北京时间的新实验名。

- `RESUME_OUTPUT_DIR=/path/to/new/checkpoints`：指定新 checkpoint 目录。
- `RESUME_MAX_RESTARTS=2`：默认训练进程失败后最多重启两次，每次读取新目录的最新完整 checkpoint；没有新保存时仍使用初始 checkpoint。Ctrl-C 不触发重启。
- `DRY_RUN=1`：只检查 checkpoint 文件并打印命令，不启动训练。

自动重启需要当前计算任务仍存活，不能恢复已被平台回收的节点。平台 shell entrypoint 可使用下面的包装脚本形式：

```bash
#!/usr/bin/env bash
set -euo pipefail
exec bash /shared/users/yangyq/projects/verl/tasks/q38_27B_session_throughput.sh --resume /绝对路径/checkpoints/global_step_65
```

## 单条 rollout 卡住时

recipe 的 `training.trainer.v1.sampler.sampler_kwargs.trajectory_timeout_seconds` 默认 600 秒，覆盖整条 trajectory（多轮生成、mock、裁判、后处理）。到期取消该协程，生成中的 vLLM 请求也会被单独取消；等同组协程全部结束，再将组标记为失败并补采。其他完成组保留，不用零奖励冒充有效训练样本。

900 秒无进展和 3600 秒单次采样总上限仍作为保护：事件循环阻塞、后端失联、持续无有效数据等情况仍可能让训练进程退出，届时由续跑脚本尝试恢复。600 秒上限会丢弃确实超过此时长的正常长轨迹，属于吞吐和覆盖率的取舍。

新日志目录中的 `session_progress/*.json` 记录每条轨迹的 `group_uid`、`sample_id`、`source_line`、当前阶段和时间；阶段包括 `generation:turn_N`、`tools:<工具名>`、`user_continuation`、`judge_score`。被取消的轨迹状态为 `cancelled`。完整轨迹仍在 `session_traces`，训练实际采用的轨迹仍在 `rollouts/<step>.jsonl`。
