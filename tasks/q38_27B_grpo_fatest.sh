#!/usr/bin/env bash
set -euo pipefail

TRAIN_FILE=/shared/users/yangyq/data/recipe_v2_qwen38_native/data/rule_train.parquet \
TEST_FILE=/shared/users/yangyq/data/recipe_v2_qwen38_native/data/rule_val.parquet \
SWANLAB_API_KEY=${SWANLAB_API_KEY:-}
RECIPE_ROOT=/shared/users/yangyq/data/recipe_v2_qwen38_native \
MODEL_PATH=${MODEL_PATH:-/shared/models/Qwen3.8-27B} \
./examples/grpo_trainer/run_qwen38_27b_lora_grpo_fa.sh "$@"
