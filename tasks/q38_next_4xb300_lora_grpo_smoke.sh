#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'EOF'
Qwen3.8-Flash-Next BF16 LoRA GRPO smoke on one node with 4 B300 GPUs.

Required environment:
  MODEL_PATH                  Local BF16 Flash-Next checkpoint directory
  TRAIN_FILE                  Local training parquet with verl reward metadata
  VERL_NO_PLACEMENT_MMAP_DIR   Node-local disk directory for shared CPU weights

Optional:
  PYTHON       Prepared environment's interpreter (otherwise active uv environment)
  OUTPUT_DIR   Checkpoints; defaults to checkpoints/qwen38-next-smoke-4xb300-TIMESTAMP
  VAL_FILE     Defaults to TRAIN_FILE; evaluation is disabled in this smoke
  RAY_ADDRESS  Defaults to local (new local Ray instance); use auto for an existing cluster
  DRY_RUN=1    Print composed Hydra configuration without training or GPU allocation

Defaults: FSDP4 + TP4, LoRA16/32, 4 prompts x 2 responses, 2K prompt + 2K response,
one update, checkpoint save, no auto-resume, no KL, no MTP, no CUDA Graph.
The job must have 4 allocated GPUs. This script does not stop other training jobs.
Planning estimate: 2 TB host RAM, 128 logical CPU cores (256 for more headroom),
and 1 TB free local NVMe. These are resource estimates, not measured minima.
Model-specific Transformers/vLLM and expert LoRA support must already be installed.
Further arguments are Hydra overrides and take precedence over this preset.

Example:
  DRY_RUN=1 bash tasks/q38_next_4xb300_lora_grpo_smoke.sh
  bash tasks/q38_next_4xb300_lora_grpo_smoke.sh
EOF
    exit 0
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

: "${MODEL_PATH:?set the BF16 Qwen3.8-Flash-Next checkpoint directory}"
: "${TRAIN_FILE:?set the training parquet}"
: "${VERL_NO_PLACEMENT_MMAP_DIR:?set a node-local disk directory, not shared storage or /dev/shm}"
export MODEL_PATH TRAIN_FILE VERL_NO_PLACEMENT_MMAP_DIR
export VAL_FILE=${VAL_FILE:-${TRAIN_FILE}}
export OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/qwen38-next-smoke-4xb300-$(date +%Y%m%d_%H%M%S)}
export RAY_ADDRESS=${RAY_ADDRESS:-local}

if [[ "${DRY_RUN:-0}" != 1 ]]; then
    for input_file in "${MODEL_PATH}/config.json" "${TRAIN_FILE}" "${VAL_FILE}"; do
        if [[ ! -f "${input_file}" ]]; then
            echo "ERROR: input file not found: ${input_file}" >&2
            exit 1
        fi
    done
fi

exec bash "${REPO_ROOT}/examples/grpo_trainer/run_qwen38_next_lora_fsdp2.sh" \
    trainer.project_name=qwen38_next_lora_grpo \
    trainer.experiment_name=smoke_4xb300_fsdp4_tp4 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    trainer.total_training_steps=1 \
    trainer.resume_mode=disable \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=1 \
    data.train_batch_size=4 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.mtp.enable=False \
    actor_rollout_ref.model.mtp.enable_train=False \
    actor_rollout_ref.model.mtp.enable_rollout=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    "$@"
