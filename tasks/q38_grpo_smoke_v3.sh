# SwanLab 使用已有登录，或运行前 export SWANLAB_API_KEY。
TRAIN_FILE=/shared/users/yangyq/data/recipe_v3_qwen38_json/data/runtime/train.parquet \
TEST_FILE=/shared/users/yangyq/data/recipe_v3_qwen38_json/data/runtime/val.parquet \
RECIPE_ROOT=/shared/users/yangyq/data/recipe_v3_qwen38_json \
MODEL_PATH=/shared/models/Qwen3.8-27B \
./examples/grpo_trainer/run_qwen38_27b_lora_grpo_smoke_v3.sh
