#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# One public entrypoint: no --resume starts a new run, --resume restores full state.
# The resume controller calls this preset again with explicit Hydra overrides;
# consume --resume here so that call cannot recursively enter the controller.
resume_checkpoint=
resume_requested=0
training_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
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
Usage: bash tasks/q38_27B_session_throughput.sh [--resume CHECKPOINT] [Hydra overrides...]
No --resume: start a fresh training run.
--resume /path/to/checkpoints/global_step_65: restore full state and continue at step 66.
--resume /path/to/checkpoints: restore the latest completed checkpoint in that directory.
Resume mode keeps the existing checkpoint validation and automatic retry behavior.
DRY_RUN=1 prints the training command without starting training.
EOF
            exit 0
            ;;
        *) training_args+=("$1"); shift ;;
    esac
done
if [[ "$resume_requested" == 1 ]]; then
    exec bash "${SCRIPT_DIR}/q38_27B_session_resume.sh" "$resume_checkpoint" "${training_args[@]}"
fi
set -- "${training_args[@]}"
# An inherited RESUME_MODE must not silently resume a normal launch. The resume
# controller supplies trainer.resume_mode=resume_path explicitly as a CLI override.
export RESUME_MODE=disable

# Full-session throughput preset. Reuse the existing task's model, recipe and
# credential setup; all variables and trailing Hydra arguments remain overridable.
# h70s2qpj: steps 2-4 spent ~81% of wall time in rollout/tool audit/refill;
# observed total trajectory lengths were 4.7K-7.5K. More independent sessions
# let vLLM work while other sessions await the judge. A worker is not a GPU:
# the four agent workers share the existing single vLLM engine.
# Four sessions x four trajectories => sixteen concurrent trajectories.
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export ROLLOUT_N=${ROLLOUT_N:-4}
export SAVE_FREQ=${SAVE_FREQ:-5}
# Read-only query tools: exact recorded result first, otherwise Doubao simulation.
# TOOL_MOCK_MODE=replay restores strict replay. Judge configuration is independent.
export TOOL_MOCK_MODE=${TOOL_MOCK_MODE:-hybrid}
export TOOL_MOCK_MODEL=${TOOL_MOCK_MODEL:-ep-20260513201344-xl7ph}
export TOOL_MOCK_BASE_URL=${TOOL_MOCK_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-4}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
# Reduce per-iteration prefill work after the 20260921_155428 run timed out
# inside execute_model on an 8192-token mixed prefill/decode batch.
# MAX_NUM_SEQS is derived from TRAIN_BATCH_SIZE * rollout.n in the base script.
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
GDN_PREFILL_BACKEND=${GDN_PREFILL_BACKEND:-triton}
if [ -z "${EXPERIMENT_NAME:-}" ] || [ "${EXPERIMENT_NAME}" = verl-grpo-example ]; then
    export EXPERIMENT_NAME=Qwen3.8-27B-LoRA-FA4-B300-Session-Throughput-${SESSION_PROFILE:-full}
fi

# Keep cache release for backward headroom; idle phases do not benefit from
# reserving KV memory. Micro batch 2 is based on the observed short sessions,
# not a verified 64K peak. For long sessions set both micro batch variables to 1.
export FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-True}

# Effective optimizer mini batch is 16 trajectories, with one mini batch per
# trainer step. Keep LR/reward unchanged; compare sessions/sec,
# reward and coverage, not seconds/step. Same-score filtering remains disabled.
# This does not fix unsupported replay branches or judge response failures.
# This checkpoint uses Qwen3.5 GDN layers. Select Triton/FLA explicitly to avoid
# the auto-selected FlashInfer GDN path as a timeout mitigation (not yet GPU
# verified). This does not change the actor's FA4 or full-attention backend.
# INFO logs expose the actual selected GDN backend on the next launch.
# Related upstream report: https://github.com/vllm-project/vllm/issues/38916
exec bash "${SCRIPT_DIR}/q38_27B_session.sh" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend="${GDN_PREFILL_BACKEND}" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL=INFO \
    "$@"
