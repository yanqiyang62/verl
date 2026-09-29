#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-Flash-Next uses its own BF16 checkpoint and output directory.
# The existing Qwen3.8-27B session launchers keep their original configuration.
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'EOF'
Qwen3.8-Flash-Next: BF16 LoRA GRPO with FSDP2 and vLLM.

Required environment:
  MODEL_PATH                  Shared BF16 Flash-Next checkpoint (not the 27B model)
  TRAIN_FILE                  Training parquet with compatible reward metadata
  VAL_FILE                    Validation parquet
  OUTPUT_DIR                  Separate Next checkpoint directory
  VERL_NO_PLACEMENT_MMAP_DIR   Node-local disk path shared by ranks on each host

Optional:
  PYTHON       Interpreter from a prepared environment; otherwise use active uv env
  RAY_ADDRESS  Existing Ray cluster address (default: auto)
  DRY_RUN=1    Compose and print Hydra config only; no training or Ray connection

Default recipe: 2 nodes x 8 GPUs, FSDP16, vLLM TP4, LoRA 16/32,
CUDA Graph + MTP3, dynamic batches, no packing, no KL loss.
Model-specific Transformers/vLLM and adapter-loading support must be installed.
Additional arguments are Hydra overrides, e.g. trainer.total_training_steps=1.
EOF
    exit 0
fi

: "${VERL_NO_PLACEMENT_MMAP_DIR:?set a node-local disk directory for frozen CPU weights}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PREVIEW=()
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    PREVIEW=(--cfg job)
fi

# The upstream filename refers to DAPO-Math data; its estimator is GRPO.
# Reuse its configuration so the two entry points cannot drift apart.
exec bash "${REPO_ROOT}/scripts/run_qwen38_lora_dapo.sh" \
    trainer.project_name=qwen38_next_lora_grpo \
    trainer.experiment_name=next_bf16_fsdp16_tp4 \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_NO_PLACEMENT_MMAP_DIR=${VERL_NO_PLACEMENT_MMAP_DIR}" \
    "${PREVIEW[@]}" \
    "$@"
