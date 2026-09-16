#!/usr/bin/env bash
set -xeuo pipefail

# ============================================================
# Qwen3.8-27B
# verl GRPO + FSDP2 + LoRA + vLLM
# B300 full-GPU smoke test
#
# 1 GPU
# prompt <= 8192
# response <= 4096
# rollout.n = 2
# train_batch_size = 1
# total_training_steps = 1
#
# + custom FixedPrefixDataset
# + custom fixed-prefix AgentLoop
# + custom reference_agreement reward
# + SwanLab logging
# ============================================================

export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false

# ------------------------------------------------------------
# Python environment
# ------------------------------------------------------------

# 已通过 uv sync 安装好的固定环境；训练时不再调用 uv run
PYTHON=${PYTHON:-/shared/users/yangyq/env/verl/bin/python}

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: Python not found or not executable: ${PYTHON}" >&2
    exit 1
fi

# ------------------------------------------------------------
# recipe_v2_fixed
# ------------------------------------------------------------
# 推荐把本脚本放在 recipe_v2_fixed/ 目录内；也支持放在它的父目录。
# 如果放在其他位置，运行前：
#   export RECIPE_ROOT=/path/to/recipe_v2_fixed
# ------------------------------------------------------------

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ -z "${RECIPE_ROOT:-}" ]; then
    if [ -f "${SCRIPT_DIR}/tf_rl/reward.py" ]; then
        RECIPE_ROOT="${SCRIPT_DIR}"
    elif [ -f "${SCRIPT_DIR}/recipe_v2_fixed/tf_rl/reward.py" ]; then
        RECIPE_ROOT="${SCRIPT_DIR}/recipe_v2_fixed"
    else
        echo "ERROR: Cannot locate recipe_v2_fixed." >&2
        echo "Set it explicitly: export RECIPE_ROOT=/path/to/recipe_v2_fixed" >&2
        exit 1
    fi
fi

RECIPE_ROOT=$(cd "${RECIPE_ROOT}" && pwd)

REWARD_FILE="${RECIPE_ROOT}/tf_rl/reward.py"
DATASET_FILE="${RECIPE_ROOT}/tf_rl/dataset.py"
AGENT_LOOP_CONFIG="${RECIPE_ROOT}/agent_loop.yaml"

for f in "${REWARD_FILE}" "${DATASET_FILE}" "${AGENT_LOOP_CONFIG}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: Required recipe file not found: ${f}" >&2
        exit 1
    fi
done

# tf_rl.agent_loop / tf_rl.rendering 必须能被 driver 和 Ray worker import
export PYTHONPATH="${RECIPE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ------------------------------------------------------------
# SwanLab
# ------------------------------------------------------------
# 二选一即可：
#   1) 提前执行: /shared/users/yangyq/env/verl/bin/swanlab login
#   2) 运行前:   export SWANLAB_API_KEY=xxxx
#
# 不在脚本里写死 API Key。
# ------------------------------------------------------------

if [ -z "${SWANLAB_API_KEY:-}" ]; then
    echo "INFO: SWANLAB_API_KEY is not set; SwanLab will use existing login credentials if available."
fi

# 只做 import 检查，不安装任何东西
"${PYTHON}" - <<'PY'
import importlib.util
missing = [m for m in ("verl", "vllm", "swanlab", "jsonschema", "pyarrow")
           if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("ERROR: missing packages in current environment: " + ", ".join(missing))
print("Environment import check passed.")
PY

# ------------------------------------------------------------
# Model / data
# ------------------------------------------------------------

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3.8-27B"}

# 这个 reward + dataset + agent loop 是和 rule_*.parquet 配套的。
# 需要换路径可以在运行时覆盖 TRAIN_FILE / TEST_FILE。
TRAIN_FILE=${TRAIN_FILE:-"${RECIPE_ROOT}/data/rule_train.parquet"}
TEST_FILE=${TEST_FILE:-"${RECIPE_ROOT}/data/rule_val.parquet"}

if [ ! -f "${TRAIN_FILE}" ]; then
    echo "ERROR: TRAIN_FILE not found: ${TRAIN_FILE}" >&2
    exit 1
fi
if [ ! -f "${TEST_FILE}" ]; then
    echo "ERROR: TEST_FILE not found: ${TEST_FILE}" >&2
    exit 1
fi

# ------------------------------------------------------------
# Hardware
# ------------------------------------------------------------

NDEVICES_PER_NODE=1
NNODES=1

# 单张 B300 整卡
GEN_TP=1
FSDP_SIZE=1
SP_SIZE=1

# ------------------------------------------------------------
# Smoke-test parameters
# ------------------------------------------------------------

TRAIN_BATCH_SIZE=1
PPO_MINI_BATCH_SIZE=1
PPO_MICRO_BATCH_SIZE_PER_GPU=1

MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=4096
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

# 真正 GRPO，保留同一个 prompt 的两个 response
ROLLOUT_N=2

# 只取少量数据做 smoke；峰值显存主要由单 step 配置决定，不由总数据量决定
#TRAIN_MAX_SAMPLES=8
#VAL_MAX_SAMPLES=2

# NORMAL
TRAIN_MAX_SAMPLES=-1
VAL_MAX_SAMPLES=-1

# LoRA
LORA_RANK=32
LORA_ALPHA=64
ACTOR_LR=3e-6

# ------------------------------------------------------------
# vLLM
# ------------------------------------------------------------

# vLLM 显存预算占整张 GPU 显存的比例
ROLLOUT_GPU_MEM_UTIL=0.50

# 8K prompt 分块 prefill
MAX_NUM_BATCHED_TOKENS=2048

PROJECT_NAME=${PROJECT_NAME:-GRPO-Qwen3.8-Smoke}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-Qwen3.8-27B-LoRA-FSDP2-B300-ReferenceReward}

# Keep each launch separate, including launches in the same second.
START_TIME=$(date -u +%Y%m%d_%H%M%SZ)
EXPERIMENT_LOG_DIR="${LOG_ROOT:-logs}/${PROJECT_NAME}/${EXPERIMENT_NAME}"
mkdir -p "${EXPERIMENT_LOG_DIR}"
RUN_DIR=$(mktemp -d "${EXPERIMENT_LOG_DIR}/${START_TIME}_XXXXXX")
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${RUN_DIR}/rollouts}
mkdir -p "${ROLLOUT_DATA_DIR}"
echo "Run directory: ${RUN_DIR}"
echo "Rollout directory: ${ROLLOUT_DATA_DIR}"

# ============================================================
# DATA
# ============================================================

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"

    # 这套 parquet 不能直接交给默认 RLHFDataset
    data.custom_cls.path="${DATASET_FILE}"
    data.custom_cls.name=FixedPrefixDataset

    # 和 recipe_v2_fixed/run.py 的默认模板设置保持一致
    ++data.apply_chat_template_kwargs.enable_thinking=false

    # Qwen3.8 may expose a processor, but this recipe is text-only.
    data.return_multi_modal_inputs=False

    # 单卡 smoke，避免额外 dataloader worker 干扰
    data.dataloader_num_workers=0

    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.train_max_samples=${TRAIN_MAX_SAMPLES}
    data.val_max_samples=${VAL_MAX_SAMPLES}

    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}

    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=False
)

# ============================================================
# MODEL / LoRA
# ============================================================

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
    ################################################################################################################################################
    ##[Bug] Multiple crashes when training Qwen3.5 with Ulysses SP + FlashAttention varlen (Contiguous, Illegal Memory Access, Shape Mismatch)
    ##https://github.com/verl-project/verl/issues/6284
    #####################################################################################################################################################
    actor_rollout_ref.model.use_remove_padding=False #bug 

    actor_rollout_ref.model.lora_rank=${LORA_RANK}
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    actor_rollout_ref.model.target_modules=all-linear
)

# ============================================================
# ALGORITHM
# ============================================================

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
)

# ============================================================
# CUSTOM REWARD
# ============================================================
# reward.py 中：
#   compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs)
# 返回：
#   score / acc / field_accuracy / format_valid
# 其中 score 是真正用于 GRPO 的 reward。
# ============================================================

REWARD=(
    reward.custom_reward_function.path="${REWARD_FILE}"
    reward.custom_reward_function.name=compute_score

    reward.reward_manager.source=register
    reward.reward_manager.name=naive

    # reward 很轻，smoke 单 worker 足够
    reward.num_workers=1
)

# ============================================================
# ACTOR: FSDP2 + aggressive offload
# ============================================================

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2

    # IMPORTANT: verl FSDP defaults model_dtype to fp32.
    # Load the 27B base model directly as BF16 to avoid ~110GB host-RAM usage.
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16

    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}

    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}

    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.use_dynamic_bsz=False



    ################################################################################################################################################
    ##[Bug] Multiple crashes when training Qwen3.5 with Ulysses SP + FlashAttention varlen (Contiguous, Illegal Memory Access, Shape Mismatch)
    ##https://github.com/verl-project/verl/issues/6284
    #####################################################################################################################################################
    actor_rollout_ref.actor.use_remove_padding=False


    # KL against base/reference policy
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0

    actor_rollout_ref.actor.use_torch_compile=False

    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE}
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True

    # FSDP offload 配置
    actor_rollout_ref.actor.fsdp_config.offload_policy=False

    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False

    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
)

# ============================================================
# REFERENCE
# ============================================================

REF=(
    actor_rollout_ref.ref.strategy=fsdp2

    # Keep reference/base-policy forward in BF16 as well.
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16

    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.use_torch_compile=False

    actor_rollout_ref.ref.fsdp_config.param_offload=False
    actor_rollout_ref.ref.fsdp_config.offload_policy=False
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True

    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
)

# ============================================================
# vLLM rollout + fixed-prefix AgentLoop
# ============================================================

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm

    # 当前 verl 的 AgentLoop 走 async rollout
    actor_rollout_ref.rollout.mode=async

    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.ignore_eos=False

    # 这里仍然是“固定历史前缀 -> 只生成下一步 assistant”
    # 不是在线真实工具多轮执行，所以 multi_turn 保持 false。
    actor_rollout_ref.rollout.multi_turn.enable=False

    # 使用你包里的 fixed_prefix_agent
    actor_rollout_ref.rollout.agent.default_agent_loop=fixed_prefix_agent
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}"
    actor_rollout_ref.rollout.agent.num_workers=1
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072
 
    
    

    # 关键 1：
    # Hybrid engine 本来就应该让 actor 给 vLLM 同步权重，
    # 不需要 vLLM 自己重新从 safetensors 加载完整 27B。
    actor_rollout_ref.rollout.load_format=dummy
    # mig
    # actor_rollout_ref.rollout.load_format=safetensors

    # 整卡不需要 layered summon
    actor_rollout_ref.rollout.layered_summon=False
    # mig
    #actor_rollout_ref.rollout.layered_summon=True

    # 关键 2：
    # 不要让 vLLM sleep level 1 再把 55~70GB 权重备份到 CPU
    actor_rollout_ref.rollout.free_cache_engine=False
    # offload
    #actor_rollout_ref.rollout.free_cache_engine=True

    # memory
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}

    # n=2，但一次只调度一条 sequence，牺牲速度换峰值显存
    actor_rollout_ref.rollout.max_num_seqs=1

    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}

    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.enforce_eager=True
    

    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
)

# ============================================================
# Trainer + SwanLab
# ============================================================

TRAINER=(
    trainer.critic_warmup=0

    # SwanLab：console + swanlab
    trainer.logger='["console","swanlab"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"

    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.balance_batch=False

    # smoke：不额外做 validation / checkpoint
    trainer.val_before_train=False
    trainer.save_freq=20
    trainer.test_freq=-1
    trainer.log_val_generations=0

    # 把实际 rollout + reward extra info 落盘，便于核对 acc/field_accuracy/format_valid
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}"

    # 只跑一个 GRPO training step
    trainer.total_training_steps=null
    trainer.total_epochs=1
)

# ============================================================
# Ray
# ============================================================

RAY=(
    ray_kwargs.ray_init.runtime_env.py_executable="${PYTHON}"

    # agent_loop.yaml 里的 _target_=tf_rl.agent_loop... 需要 Ray worker 能 import tf_rl
    ++ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH="${PYTHONPATH}"
)

# ============================================================
# Launch
# ============================================================

ray stop --force || true

# 直接使用已经同步好的 /shared/users/yangyq/env/verl；不再执行 uv run / uv sync
"${PYTHON}" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ALGORITHM[@]}" \
    "${REWARD[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${RAY[@]}" \
    transfer_queue.backend.SimpleStorage.num_data_storage_units=2 \
    "$@" \
    2>&1 | tee "${RUN_DIR}/train.log"
