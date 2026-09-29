#!/usr/bin/env bash
set -euo pipefail
export TZ=Asia/Shanghai
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

# Use the same exported ARK_API_KEY/JUDGE_API_KEY as the session GRPO job.
# Full training: bash tasks/q38_27B_session_dapo.sh
# Preview only: DRY_RUN=1 bash tasks/q38_27B_session_dapo.sh
export RECIPE_ROOT=${RECIPE_ROOT:-/shared/users/yangyq/data/recipe_v3_session_agentloop}
export MODEL_PATH=${MODEL_PATH:-/shared/users/xiongf/ckpts/full-sft-v13-person}
export PYTHON=${PYTHON:-/shared/users/yangyq/env/verl/bin/python}
export SESSION_PROFILE=${SESSION_PROFILE:-full}
exec bash "${REPO_ROOT}/examples/dapo_trainer/run_qwen38_27b_lora_dapo_fa_session.sh" "$@"
