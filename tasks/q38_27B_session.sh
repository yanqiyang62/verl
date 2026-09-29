#!/usr/bin/env bash
set -euo pipefail
export TZ=Asia/Shanghai
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

# 运行前 export ARK_API_KEY 或 JUDGE_API_KEY；SwanLab 使用已有登录或 SWANLAB_API_KEY。
# 默认全量训练；SESSION_PROFILE=smoke 跑 1 步，DRY_RUN=1 只预览命令。
export RECIPE_ROOT=${RECIPE_ROOT:-/shared/users/yangyq/data/recipe_v3_session_agentloop}
export MODEL_PATH=${MODEL_PATH:-/shared/users/xiongf/ckpts/full-sft-v13-person}
export PYTHON=${PYTHON:-/shared/users/yangyq/env/verl/bin/python}
export SESSION_PROFILE=${SESSION_PROFILE:-full}
export ARK_API_KEY=${ARK_API_KEY:-}
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-}
# 关闭 vLLM 内部异步调度，排查 sample_tokens RPC 超时；多轮 AgentLoop 保持启用。
exec bash "${REPO_ROOT}/examples/grpo_trainer/run_qwen38_27b_lora_grpo_fa_session.sh" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=False \
    "$@"
