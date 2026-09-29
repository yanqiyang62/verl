#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B / single full B300 / FSDP2 + LoRA + FA4 + vLLM.
# V3 session replay: generated assistant turns, recorded tool observations,
# final judge reward, and whole-group filtering/refill before GRPO.
#
# Full: bash examples/grpo_trainer/run_qwen38_27b_lora_grpo_fa_session.sh
# Smoke: SESSION_PROFILE=smoke bash <this-script> trainer.logger='[console]'
# Judge credentials: export ARK_API_KEY or JUDGE_API_KEY before training.
# Preview without patching the environment or starting Ray: DRY_RUN=1 bash <this-script>
# More rollout throughput after measuring long-session memory: TRAIN_BATCH_SIZE=4
# Keep vLLM resident only after checking full-session peaks: FREE_CACHE_ENGINE=False
# Optional CUDA graphs after validating this model/vLLM combination: ENFORCE_EAGER=False
# Each launch saves checkpoints under its own run directory and starts fresh.
# Explicit resume: trainer.resume_mode=resume_path trainer.resume_from_path=/path/to/global_step_N
# Save LoRA weights plus optimizer/scheduler/RNG state for resume with the same base model.
# Override with ++actor_rollout_ref.actor.checkpoint.save_lora_only=False to save full weights.
# recipe.yaml owns data paths, context budgets and sampler settings;
# ROLLOUT_N overrides the recipe's training group size and is synchronized with
# the sampler through recipe_rollout_n. Default remains 2.
# agent_loop.yaml owns turn limits, parser and optional process audits.
# Those settings override CLI values in tf_rl.main. Other CLI overrides win.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
PYTHON=${PYTHON:-/shared/users/yangyq/env/verl/bin/python}
PATCH_FILE=${PATCH_FILE:-${REPO_ROOT}/scripts/patch_transformers_fa4_max_seqlen.py}
DRY_RUN=${DRY_RUN:-0}
SESSION_PROFILE=${SESSION_PROFILE:-full}
case "${SESSION_PROFILE}" in
    smoke) DATA_PREFIX=smoke; DEFAULT_TRAINING_STEPS=1; DEFAULT_FREE_CACHE=False ;;
    full) DATA_PREFIX=rule; DEFAULT_TRAINING_STEPS=null; DEFAULT_FREE_CACHE=True ;;
    repeat_search_v2) DATA_PREFIX=rule_repeat_v2; DEFAULT_TRAINING_STEPS=null; DEFAULT_FREE_CACHE=True ;;
    repeat_search) DATA_PREFIX=rule_repeat_v1; DEFAULT_TRAINING_STEPS=null; DEFAULT_FREE_CACHE=True ;;
    *) echo "ERROR: SESSION_PROFILE must be smoke, full, repeat_search or repeat_search_v2" >&2; exit 1 ;;
esac

if [ -z "${RECIPE_ROOT:-}" ]; then
    for candidate in "${REPO_ROOT}/data/recipe_v3_session_agentloop" /shared/users/yangyq/data/recipe_v3_session_agentloop; do
        if [ -f "${candidate}/tf_rl/configuration.py" ]; then
            RECIPE_ROOT=${candidate}
            break
        fi
    done
fi
if [ -z "${RECIPE_ROOT:-}" ]; then
    echo "ERROR: set RECIPE_ROOT=/path/to/recipe_v3_session_agentloop" >&2
    exit 1
fi
RECIPE_ROOT=$(cd "${RECIPE_ROOT}" && pwd)
REWARD_FILE=${RECIPE_ROOT}/tf_rl/reward.py
DATASET_FILE=${RECIPE_ROOT}/tf_rl/dataset.py
AGENT_LOOP_CONFIG=${RECIPE_ROOT}/agent_loop.yaml
MODEL_PATH=${MODEL_PATH:-/shared/models/Qwen3.8-27B}
TRAIN_FILE=${RECIPE_ROOT}/data/${DATA_PREFIX}_train.parquet
TEST_FILE=${RECIPE_ROOT}/data/${DATA_PREFIX}_val.parquet
if [[ ${SESSION_PROFILE} == repeat_search* ]]; then TEST_FILE=${RECIPE_ROOT}/data/rule_val.parquet; fi

for file in "${REWARD_FILE}" "${DATASET_FILE}" "${AGENT_LOOP_CONFIG}" "${TRAIN_FILE}" "${TEST_FILE}" "${PATCH_FILE}" \
    "${RECIPE_ROOT}/recipe.yaml" "${RECIPE_ROOT}/tf_rl/main.py" "${RECIPE_ROOT}/tf_rl/sampler.py" \
    "${RECIPE_ROOT}/scripts/preflight.py" "${TRAIN_FILE%.parquet}.jsonl" "${TEST_FILE%.parquet}.jsonl"; do
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
export RECIPE_ROOT SESSION_PROFILE
export PYTHONPATH="${RECIPE_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Two sessions, two trajectories each; one update over four trajectories.
# Full sessions may reach 64K tokens, so retain micro batch 1 for the first run.
# Increasing the prompt batch changes the effective optimizer batch and reduces
# updates per epoch. Keep LR unchanged initially and compare reward as well as speed.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
# Read the same merged recipe used by the actual entry point, not a second set
# of hardcoded length/n defaults. This does not initialize the judge or Ray.
RECIPE_LIMITS=$("${PYTHON}" - <<'PY'
from omegaconf import OmegaConf
from tf_rl.configuration import apply_recipe_config
cfg = apply_recipe_config(OmegaConf.create({}))
print(cfg.actor_rollout_ref.rollout.n, cfg.data.max_prompt_length,
      cfg.data.max_response_length, cfg.actor_rollout_ref.rollout.max_model_len)
PY
)
read -r ROLLOUT_N MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH MAX_MODEL_LEN <<< "${RECIPE_LIMITS}"
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
MAX_NUM_SEQS=${MAX_NUM_SEQS:-$((TRAIN_BATCH_SIZE * ROLLOUT_N))}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}

# 0.50 is the vLLM engine budget, not total GPU usage. Full 64K sessions can
# require much more activation memory than the old 12K single-turn workload.
# Release the rollout cache during training in full mode; this trades CPU
# weight backup/restore overhead for more backward headroom.
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.50}
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-${DEFAULT_FREE_CACHE}}
# Retain the original hybrid-model compatibility settings by default.
ENFORCE_EAGER=${ENFORCE_EAGER:-True}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-False}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}

PROJECT_NAME=${PROJECT_NAME:-GRPO-Qwen3.8}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-Qwen3.8-27B-LoRA-FA4-B300-Session-${SESSION_PROFILE}}
# Launchers may inject this generic name, overriding the default above.
if [ "${EXPERIMENT_NAME}" = verl-grpo-example ]; then
    EXPERIMENT_NAME=Qwen3.8-27B-LoRA-FA4-B300-Session-${SESSION_PROFILE}
fi
# Explicit timezone: do not depend on the training node's local timezone.
EXPERIMENT_NAME="${EXPERIMENT_NAME}-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_BJT"
EXPERIMENT_LOG_DIR=${LOG_ROOT:-logs}/${PROJECT_NAME}/${EXPERIMENT_NAME}
if [ "${DRY_RUN}" = 1 ]; then
    RUN_DIR=${EXPERIMENT_LOG_DIR}/dry-run
else
    mkdir -p "${EXPERIMENT_LOG_DIR}"
    RUN_DIR=$(mktemp -d "${EXPERIMENT_LOG_DIR}/$(date -u +%Y%m%d_%H%M%SZ)_XXXXXX")
    # mktemp creates owner-only directories. Let the inherited workspace group
    # read logs/traces when training runs as root in a container.
    chmod g+rx "${RUN_DIR}"
    RUN_DIR=$(cd "${RUN_DIR}" && pwd)
fi
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${RUN_DIR}/rollouts}
export SESSION_TRACE_DIR=${SESSION_TRACE_DIR:-${RUN_DIR}/session_traces}

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.custom_cls.path="${DATASET_FILE}"
    data.custom_cls.name=FixedPrefixDataset
    ++data.apply_chat_template_kwargs.enable_thinking=false
    data.return_multi_modal_inputs=False
    data.dataloader_num_workers=0
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    # Fetch one UID at a time for exact refill; V1 coalesces train_batch_size
    # UIDs before dispatch, so this does not serialize session generation.
    data.gen_batch_size=1
    data.train_max_samples=${TRAIN_MAX_SAMPLES:--1}
    data.val_max_samples=${VAL_MAX_SAMPLES:--1}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=False
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
    ++actor_rollout_ref.actor.checkpoint.save_lora_only=True
    actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra]'
    actor_rollout_ref.actor.checkpoint.load_contents='[model,optimizer,extra]'
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
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.agent.default_agent_loop=fixed_prefix_agent
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}"
    # One agent worker already submits all responses concurrently via asyncio.
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS:-1}
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=False
    actor_rollout_ref.rollout.free_cache_engine=${FREE_CACHE_ENGINE}
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
    trainer.use_v1=True
    trainer.v1.trainer_mode=sync
    trainer.critic_warmup=0
    trainer.logger='["console","swanlab"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    # Schedulers may reuse experiment_name across recipes; never auto-load an
    # unrelated single-turn checkpoint or overwrite it with session training.
    trainer.default_local_dir="${CHECKPOINT_DIR:-${RUN_DIR}/checkpoints}"
    trainer.resume_mode=${RESUME_MODE:-disable}
    trainer.n_gpus_per_node=1
    trainer.nnodes=1
    trainer.balance_batch=False
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ:-20}
    trainer.test_freq=-1
    trainer.log_val_generations=0
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-${DEFAULT_TRAINING_STEPS}}
    trainer.total_epochs=${TOTAL_EPOCHS:-1}
)
RAY=(
    ray_kwargs.ray_init.runtime_env.py_executable="${PYTHON}"
    ++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="${PYTHONPATH}"
)
COMMAND=(
    # tf_rl.main installs SessionTrainer and SessionReplayBuffer. The stock
    # main_ppo in this checkout does not dispatch recipe directories.
    # -P keeps the current repo's scripts package from shadowing the recipe's
    # scripts.preflight; PYTHONPATH above explicitly orders recipe before repo.
    "${PYTHON}" -P -m tf_rl.main
    "${DATA[@]}" "${MODEL[@]}" "${ALGORITHM[@]}" "${ACTOR[@]}"
    "${REF[@]}" "${ROLLOUT[@]}" "${TRAINER[@]}" "${RAY[@]}"
    transfer_queue.backend.SimpleStorage.num_data_storage_units=2
    transfer_queue.enable=True
    ++recipe_root="${RECIPE_ROOT}"
    ++recipe_profile="${SESSION_PROFILE}"
    ++recipe_rollout_n="${ROLLOUT_N}"
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

required = ("verl", "vllm", "transformers", "flash_attn", "swanlab", "jsonschema", "pyarrow", "openai", "transfer_queue")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("ERROR: missing packages: " + ", ".join(missing))
version = importlib.metadata.version("transformers")
if version != "5.9.0":
    raise SystemExit(f"ERROR: expected transformers==5.9.0, got {version}")
for name in ("torch", "transformers", "vllm"):
    print(f"{name}: {importlib.metadata.version(name)}")
# Configuration check only; never print credentials or send a judge request.
from tf_rl.judge import HttpJudge
HttpJudge()
PY
"${PYTHON}" "${PATCH_FILE}"
mkdir -p "${ROLLOUT_DATA_DIR}"
mkdir -p "${SESSION_TRACE_DIR}"
echo "Run directory: ${RUN_DIR}"
echo "Session profile: ${SESSION_PROFILE}; context=${MAX_MODEL_LEN}; trace directory: ${SESSION_TRACE_DIR}"
echo "Defaults before CLI overrides: prompts=${TRAIN_BATCH_SIZE}, n=${ROLLOUT_N}, concurrent sequences=${MAX_NUM_SEQS}, actor micro=${PPO_MICRO_BATCH_SIZE_PER_GPU}, log-prob micro=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
"${COMMAND[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
