#!/usr/bin/env bash
set -euo pipefail

# 可在下面的单引号内填写 key；留空时使用终端中已 export 的环境变量。
# 填入真实 key 后不要将此文件提交到 Git。

export ARK_API_KEY=${ARK_API_KEY:-}
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-}
export ARK_API_KEY="${SCRIPT_ARK_API_KEY:-${ARK_API_KEY:-}}"
export SWANLAB_API_KEY="${SCRIPT_SWANLAB_API_KEY:-${SWANLAB_API_KEY:-}}"

# Full V4 GRPO training.
# bash tasks/q38_27B_session_v4_grpo.sh [--refresh | --resume CHECKPOINT] [Hydra overrides...]
# DRY_RUN=1 prints/checks the configuration without GPU work or API requests.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
resume_checkpoint=
resume_requested=0
refresh_requested=0
training_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --refresh)
            refresh_requested=1
            shift
            ;;
        --resume|--resume=*)
            if [[ "$resume_requested" == 1 ]]; then
                echo 'ERROR: --resume may only be specified once' >&2
                exit 2
            fi
            resume_requested=1
            if [[ "$1" == --resume ]]; then
                if [[ $# -lt 2 || -z "$2" || "$2" == --* || "$2" == *=* ]]; then
                    echo 'ERROR: --resume requires a checkpoint directory' >&2
                    exit 2
                fi
                resume_checkpoint=$2
                shift 2
            else
                resume_checkpoint=${1#--resume=}
                shift
            fi
            if [[ -z "$resume_checkpoint" ]]; then
                echo 'ERROR: --resume requires a checkpoint directory' >&2
                exit 2
            fi
            ;;
        --help|-h)
            cat <<'EOF'
Usage: bash tasks/q38_27B_session_v4_grpo.sh [--refresh | --resume CHECKPOINT] [Hydra overrides...]
No flag: resume the most recently saved complete V4 GRPO checkpoint.
--refresh: start fresh. --resume PATH: restore the explicitly selected checkpoint.
No matching checkpoint: stop with an error; never silently start from zero.
CHECKPOINT accepts global_step_N or its checkpoints parent with a completion marker.
Resume and automatic retries retain V4 GRPO and Doubao 2.0 low for both services.
RESUME_MAX_RESTARTS=2 by default; set 0 to disable automatic retries.
DRY_RUN=1 previews the effective command without training or API calls.
EOF
            exit 0
            ;;
        *) training_args+=("$1"); shift ;;
    esac
done
if [[ "$refresh_requested" == 1 && "$resume_requested" == 1 ]]; then
    echo 'ERROR: --refresh and --resume cannot be combined' >&2
    exit 2
fi
# The resume controller owns these settings; public invocations must use the flags.
if [[ ${SESSION_RESUME_MANAGED:-0} != 1 ]]; then
    for arg in "${training_args[@]}"; do
        option=${arg#+}; option=${option#+}
        case "$option" in
            trainer.resume_mode=*|trainer.resume_from_path=*|trainer.default_local_dir=*|trainer.del_local_ckpt_after_load=*)
                echo 'ERROR: use --resume/--refresh and RESUME_OUTPUT_DIR for resume control' >&2
                exit 2 ;;
        esac
    done
fi
set -- "${training_args[@]}"
cd "${REPO_ROOT}"
export TZ=Asia/Shanghai
export RECIPE_ROOT=${RECIPE_ROOT:-/shared/users/yangyq/data/recipe_v4_session_agentloop}
export MODEL_PATH=${MODEL_PATH:-/shared/users/xiongf/ckpts/full-sft-v13-person}
export PYTHON=${PYTHON:-/shared/users/yangyq/env/verl/bin/python}
export PYTHONPATH="${RECIPE_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SESSION_PROFILE=${SESSION_PROFILE:-full}
export PROJECT_NAME=${PROJECT_NAME:-GRPO-Qwen3.8}
if [[ -z "${EXPERIMENT_NAME:-}" || "${EXPERIMENT_NAME}" == verl-grpo-example ]]; then
    export EXPERIMENT_NAME=Qwen3.8-27B-V4-GRPO-Doubao2.0-low-${SESSION_PROFILE}
fi

# This preset intentionally overrides inherited endpoints from older 2.1 runs.
# V4 JsonService enforces thinking=enabled and reasoning_effort=low for both roles.
# 默认仅规则评分；ENABLE_LLM_JUDGE=true 恢复最终 LLM Judge。工具 mock 独立。
export ENABLE_LLM_JUDGE=${ENABLE_LLM_JUDGE:-false}
export JUDGE_MODEL=ep-20260513201344-xl7ph
export TOOL_MOCK_MODEL=ep-20260513201344-xl7ph
export TOOL_MOCK_MODE=${TOOL_MOCK_MODE:-hybrid}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export ROLLOUT_N=${ROLLOUT_N:-4}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-4}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
export FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-True}
export SAVE_FREQ=${SAVE_FREQ:-5}
export ACTOR_LR=${ACTOR_LR:-3e-6}
export RESUME_MODE=disable
# V4 recipe bounds replacement of unscorable groups to four rounds.
# Equal-score groups remain in the batch for this GRPO baseline.

"${PYTHON}" - <<'PY'
from omegaconf import OmegaConf
from tf_rl.common import ROOT
from tf_rl.configuration import validated_agent_options
from tf_rl.service import JsonService
spec = OmegaConf.load(ROOT / 'recipe.yaml')
if spec.integration_version != 4:
    raise SystemExit('ERROR: this launcher requires the V4 recipe')
options = validated_agent_options()
if options['enable_tool_audit'] or options['enable_user_audit']:
    raise SystemExit('ERROR: V4 GRPO requires per-tool and per-user audits disabled')
for role in ('judge', 'simulator'):
    service = JsonService(role)
    if service.config['model'] != 'ep-20260513201344-xl7ph':
        raise SystemExit(f'ERROR: {role} did not load the Doubao 2.0 endpoint')
    print(f"{role}: model={service.config['model']} thinking={service.config['thinking']} "
          f"reasoning_effort={service.config['reasoning_effort']}", flush=True)
PY
if [[ "${DRY_RUN:-0}" != 1 && -z "${ARK_API_KEY:-}${JUDGE_API_KEY:-}${TOOL_MOCK_API_KEY:-}" ]]; then
    echo 'ERROR: export ARK_API_KEY before starting training' >&2
    exit 1
fi

if [[ "$resume_requested" == 0 && "$refresh_requested" == 0 && ${SESSION_RESUME_MANAGED:-0} != 1 ]]; then
    resume_checkpoint=$("${PYTHON}" "${SCRIPT_DIR}/session_checkpoint.py" \
        --repo "${REPO_ROOT}" --log-root "${AUTO_RESUME_LOG_ROOT:-${REPO_ROOT}/logs/${PROJECT_NAME}}" \
        --strategy grpo --recipe "${RECIPE_ROOT}" --model "${MODEL_PATH}" --profile "${SESSION_PROFILE}")
    resume_requested=1
    echo "[auto-resume] Selected: $resume_checkpoint"
fi

if [[ "$resume_requested" == 1 ]]; then
    export EXPERIMENT_NAME="${EXPERIMENT_NAME}-Resume"
    exec bash "${SCRIPT_DIR}/q38_27B_session_resume.sh" --v4-grpo "$resume_checkpoint" "$@"
fi

exec bash "${REPO_ROOT}/examples/grpo_trainer/run_qwen38_27b_lora_grpo_fa_session.sh" \
    'actor_rollout_ref.model.exclude_modules="(^|.*[.])(vision_tower|visual)([.].*|$)"' \
    algorithm.filter_groups.enable=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.optim.lr=3.0e-5 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=wsd \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=False \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend="${GDN_PREFILL_BACKEND:-triton}" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL=INFO \
    "$@"
