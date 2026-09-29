# TRAIN_FILE=/shared/users/yangyq/data/recipe_v2_fixed/data/all_train.parquet \
RECIPE_ROOT=/shared/users/yangyq/data/recipe_v2_fixed \
SWANLAB_API_KEY=${SWANLAB_API_KEY:-}
MODEL_PATH=/shared/models/Qwen3.8-27B \
./examples/grpo_trainer/run_qwen38_27b_lora_grpo_smoke.sh