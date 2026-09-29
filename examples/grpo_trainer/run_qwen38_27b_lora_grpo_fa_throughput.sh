#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B / single full B300 / FSDP2 + LoRA + FA4 + vLLM.
# Based on run_qwen38_27b_lora_grpo_fa.sh; tune concurrency before reserving
# more vLLM memory, since its cache stays resident during actor training.
# Baseline: runs/verl-grpo-20260916-075552b/logs/node-0.log, steps > 5,
# excluding checkpoint steps: 44.5 s/step, 34.4 s generation, 3.6 s actor update.
#
# Start: bash examples/grpo_trainer/run_qwen38_27b_lora_grpo_fa_throughput.sh
# Override the recipe with RECIPE_ROOT=/path/to/recipe_v2_qwen38_native.
# Preview without patching the environment or starting Ray: DRY_RUN=1 bash <this-script>
# More rollout throughput: TRAIN_BATCH_SIZE=8 MAX_NUM_SEQS=16 bash <this-script>
# Less training memory: PPO_MICRO_BATCH_SIZE_PER_GPU=1 LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1 bash <this-script>
# Optional CUDA graphs after validating this model/vLLM combination: ENFORCE_EAGER=False
# Hydra overrides passed as arguments take precedence over the defaults below.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON=${PYTHON:-/shared/users/yangyq/env/verl/bin/python}
PATCH_FILE=${PATCH_FILE:-${REPO_ROOT}/scripts/patch_transformers_fa4_max_seqlen.py}
DRY_RUN=${DRY_RUN:-0}

if [ -z "${RECIPE_ROOT:-}" ]; then
    # Use the same native-template recipe as the latest baseline run.
    for candidate in "${SCRIPT_DIR}" "${SCRIPT_DIR}/recipe_v2_fixed" /shared/users/yangyq/data/recipe_v2_qwen38_native; do
        if [ -f "${candidate}/tf_rl/reward.py" ]; then
            RECIPE_ROOT=${candidate}
            break
        fi
    done
fi
if [ -z "${RECIPE_ROOT:-}" ]; then
    echo "ERROR: set RECIPE_ROOT=/path/to/recipe_v2_qwen38_native" >&2
    exit 1
fi
RECIPE_ROOT=$(cd "${RECIPE_ROOT}" && pwd)
REWARD_FILE=${RECIPE_ROOT}/tf_rl/reward.py
DATASET_FILE=${RECIPE_ROOT}/tf_rl/dataset.py
AGENT_LOOP_CONFIG=${RECIPE_ROOT}/agent_loop.yaml
MODEL_PATH=${MODEL_PATH:-/shared/models/Qwen3.8-27B}
TRAIN_FILE=${TRAIN_FILE:-${RECIPE_ROOT}/data/rule_train.parquet}
TEST_FILE=${TEST_FILE:-${RECIPE_ROOT}/data/rule_val.parquet}

for file in "${REWARD_FILE}" "${DATASET_FILE}" "${AGENT_LOOP_CONFIG}" "${TRAIN_FILE}" "${TEST_FILE}" "${PATCH_FILE}"; do
    if [ ! -f "${file}" ]; then
        echo "ERROR: required file not found: ${file}" >&2
        exit 1
    fi
done
if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: Python not found or not executable: ${PYTHON}" >&2
    exit 1
fi

export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}:${RECIPE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Four prompts, two responses each; one optimizer update over eight responses.
# Increasing the prompt batch changes the effective optimizer batch and reduces
# updates per epoch. Keep LR unchanged initially and compare reward as well as speed.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
ROLLOUT_N=${ROLLOUT_N:-2}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
for name in TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE_PER_GPU LOG_PROB_MICRO_BATCH_SIZE_PER_GPU ROLLOUT_N MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH; do
    if [[ ! ${!name} =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${name} must be a positive integer" >&2
        exit 1
    fi
done
if (( ROLLOUT_N < 2 || TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE != 0 ||
      (PPO_MINI_BATCH_SIZE * ROLLOUT_N) % PPO_MICRO_BATCH_SIZE_PER_GPU != 0 ||
      (TRAIN_BATCH_SIZE * ROLLOUT_N) % LOG_PROB_MICRO_BATCH_SIZE_PER_GPU != 0 )); then
    echo "ERROR: GRPO needs n >= 2; prompt batch must divide into PPO mini batches; response batches must divide into micro batches." >&2
    exit 1
fi
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_NUM_SEQS=${MAX_NUM_SEQS:-$((TRAIN_BATCH_SIZE * ROLLOUT_N))}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}

# 0.50 is the vLLM engine budget, not the total training-process GPU usage.
# free_cache_engine=False keeps this allocation live during backward; increase
# only after measuring actor/update peak memory on representative long samples.
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.50}
# Retain the original hybrid-model compatibility settings by default.
ENFORCE_EAGER=${ENFORCE_EAGER:-True}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-False}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}

PROJECT_NAME=${PROJECT_NAME:-GRPO-Qwen3.8}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-Qwen3.8-27B-LoRA-FA4-B300-Throughput}
EXPERIMENT_LOG_DIR=${LOG_ROOT:-logs}/${PROJECT_NAME}/${EXPERIMENT_NAME}
if [ "${DRY_RUN}" = 1 ]; then
    RUN_DIR=${EXPERIMENT_LOG_DIR}/dry-run
else
    mkdir -p "${EXPERIMENT_LOG_DIR}"
    RUN_DIR=$(mktemp -d "${EXPERIMENT_LOG_DIR}/$(date -u +%Y%m%d_%H%M%SZ)_XXXXXX")
fi
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${RUN_DIR}/rollouts}

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.custom_cls.path="${DATASET_FILE}"
    data.custom_cls.name=FixedPrefixDataset
    ++data.apply_chat_template_kwargs.enable_thinking=false
    data.return_multi_modal_inputs=False
    data.dataloader_num_workers=0
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.train_max_samples=${TRAIN_MAX_SAMPLES:--1}
    data.val_max_samples=${VAL_MAX_SAMPLES:--1}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=False
)
MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.enable_gradient_checkpointing=${GRADIENT_CHECKPOINTING}
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_4
    # Preserve the original FA4 padding workaround.
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.lora_rank=${LORA_RANK:-32}
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA:-64}
    actor_rollout_ref.model.target_modules=all-linear
)
ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    reward.custom_reward_function.path="${REWARD_FILE}"
    reward.custom_reward_function.name=compute_score
    reward.reward_manager.source=register
    reward.reward_manager.name=naive
    reward.num_workers=1
)
ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-3e-6}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_remove_padding=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.offload_policy=False
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1
)
REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.ref.fsdp_config.dtype=bfloat16
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    actor_rollout_ref.ref.fsdp_config.offload_policy=False
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=1
)
ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.multi_turn.enable=False
    actor_rollout_ref.rollout.agent.default_agent_loop=fixed_prefix_agent
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}"
    # One agent worker already submits all responses concurrently via asyncio.
    actor_rollout_ref.rollout.agent.num_workers=1
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=False
    actor_rollout_ref.rollout.free_cache_engine=False
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.enable_prefix_caching=${ENABLE_PREFIX_CACHING}
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.dtype=bfloat16
)
TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","swanlab"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=1
    trainer.nnodes=1
    trainer.balance_batch=False
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ:-20}
    trainer.test_freq=-1
    trainer.log_val_generations=0
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-null}
    trainer.total_epochs=${TOTAL_EPOCHS:-1}
)
RAY=(
    ray_kwargs.ray_init.runtime_env.py_executable="${PYTHON}"
    ++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="${PYTHONPATH}"
)
COMMAND=(
    "${PYTHON}" -m verl.trainer.main_ppo
    "${DATA[@]}" "${MODEL[@]}" "${ALGORITHM[@]}" "${ACTOR[@]}"
    "${REF[@]}" "${ROLLOUT[@]}" "${TRAINER[@]}" "${RAY[@]}"
    transfer_queue.backend.SimpleStorage.num_data_storage_units=2
    "$@"
)
if [ "${DRY_RUN}" = 1 ]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

# Use the existing uv-managed environment; no installs during training.
"${PYTHON}" - <<'PY'
import importlib.metadata
import importlib.util

required = ("verl", "vllm", "transformers", "flash_attn", "swanlab", "jsonschema", "pyarrow")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("ERROR: missing packages: " + ", ".join(missing))
version = importlib.metadata.version("transformers")
if version != "5.9.0":
    raise SystemExit(f"ERROR: expected transformers==5.9.0, got {version}")
for name in ("torch", "transformers", "vllm"):
    print(f"{name}: {importlib.metadata.version(name)}")
PY
"${PYTHON}" "${PATCH_FILE}"
mkdir -p "${ROLLOUT_DATA_DIR}"
echo "Run directory: ${RUN_DIR}"
echo "Defaults before CLI overrides: prompts=${TRAIN_BATCH_SIZE}, n=${ROLLOUT_N}, concurrent sequences=${MAX_NUM_SEQS}, actor micro=${PPO_MICRO_BATCH_SIZE_PER_GPU}, log-prob micro=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
"${COMMAND[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
