#!/usr/bin/env bash
set -euo pipefail

# Session-adapted DAPO: equal-score group filtering + bounded replacement,
# asymmetric clipping, token-level loss, and no KL penalty. Keeps the session
# judge's reward/length caps; does not add the math recipe's overlong shaping.
# Requires the companion recipe's DAPO-aware configuration.py and sampler.py.
# Full training by default. DRY_RUN=1 prints the command without launching Ray.
# CLI overrides at the end win (recipe-owned context budgets and rollout.n stay fixed).
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
export PROJECT_NAME=${PROJECT_NAME:-DAPO-Qwen3.8}
if [ -z "${EXPERIMENT_NAME:-}" ] || [ "${EXPERIMENT_NAME}" = verl-grpo-example ]; then
    export EXPERIMENT_NAME=Qwen3.8-27B-LoRA-FA4-B300-Session-DAPO-${SESSION_PROFILE:-full}
fi

# The custom session sampler consumes these through apply_recipe_config.
# 17 total attempts = initial batch + at most 16 refills, still subject to the
# recipe's 900-second sample deadline. Validation never filters equal scores.
exec bash "${REPO_ROOT}/examples/grpo_trainer/run_qwen38_27b_lora_grpo_fa_session.sh" \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=score \
    algorithm.filter_groups.max_num_gen_batches="${DAPO_MAX_NUM_GEN_BATCHES:-17}" \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW:-0.2}" \
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH:-0.28}" \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=False \
    "$@"
