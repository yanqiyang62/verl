# V5 0928 重训

脚本：`tasks/q38_27B_session_v5_grpo_0928.sh`。默认从 `full-sft-v13-person` 重新训练，不自动续上旧checkpoint。

```bash
cd /shared/users/yangyq/projects/verl
bash tasks/q38_27B_session_v5_grpo_0928.sh
```

先预览配置，不启动训练：

```bash
DRY_RUN=1 bash tasks/q38_27B_session_v5_grpo_0928.sh
```

启动前在终端设置 `ARK_API_KEY`（hybrid工具模拟需要）；SwanLab使用环境中的 `SWANLAB_API_KEY`。脚本不内置凭据。

默认配置：V5 full 679 train / 70 val，reward `v5-reward-4-media-hard`；27条文生保留原始system、仅评分要求直接调用。启动前核验源输入与GT。LLM judge关闭，repeat mini bench启动时及每5步运行。batch=4、rollout_n=4、LR=3e-5、LoRA32/64，其他沿用V5 GRPO。

可用环境变量或Hydra参数覆盖训练参数，例如：

```bash
ACTOR_LR=1e-5 REPEAT_BENCH_EVERY_STEPS=5 bash tasks/q38_27B_session_v5_grpo_0928.sh
```

`--refresh`可省略；该脚本固定作为fresh入口。恢复已有训练使用通用V5入口的`--resume PATH`。日志在 `logs/GRPO-Qwen3.8/` 下，实验名包含 `0928-MediaHard` 和启动时间。

`_badcases0926`仍作为后续回测数据保存；训练文件使用已修正的V5 recipe。repeat mini bench检查重复行为，不会自动运行这36条文生回测。

验证记录：[本次启动配置检查](../report/iterations/20260928_195555_BJT_v5_0928_launcher/README.md)。本次仅完成脚本和干跑，尚未启动训练。
