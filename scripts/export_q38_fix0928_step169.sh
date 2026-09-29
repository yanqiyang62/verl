#!/usr/bin/env bash
# Export the 2026-09-23 GRPO step 50 checkpoint as a standalone HF model.
set -euo pipefail

cd /shared/users/yangyq/projects/verl
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

checkpoint=${CHECKPOINT:-${PWD}/logs/GRPO-Qwen3.8/Qwen3.8-27B-V5-GRPO-0928-MediaHard-full-20260928_200725_BJT/20260928_120726Z_7FlGAA/checkpoints/global_step_169}
output=${OUTPUT:-/shared/users/yangyq/ckpts/sft-v13-subagent-grpo-s169-0928}
base_model=${BASE_MODEL:-/shared/users/xiongf/ckpts/full-sft-v13-person}

exec "${PYTHON:-/shared/users/yangyq/env/verl/bin/python}" -u \
    scripts/export_q38_session_checkpoint.py \
    "${checkpoint}" "${output}" --base-model "${base_model}"
