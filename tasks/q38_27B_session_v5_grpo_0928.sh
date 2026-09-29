#!/usr/bin/env bash
set -euo pipefail


export ARK_API_KEY=${ARK_API_KEY:-}
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-}
export ARK_API_KEY=${ARK_API_KEY:-}
export SWANLAB_API_KEY=${SWANLAB_API_KEY:-}

# 2026-09-28: fresh V5 training with original-system media hard cases.
# Credentials come only from the environment (ARK_API_KEY / SWANLAB_API_KEY).
# DRY_RUN=1 performs offline preflight and prints the command; it starts no training.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
training_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            cat <<'EOF'
Usage: bash tasks/q38_27B_session_v5_grpo_0928.sh [--refresh] [Hydra overrides...]
Starts fresh from MODEL_PATH (default: full-sft-v13-person); no automatic resume.
V5 data: 679 train / 70 val, including 27 media tasks with original system preserved.
Reward: v5-reward-4-media-hard. LLM judge defaults off.
Repeat mini bench: at start and every 5 steps (REPEAT_BENCH_EVERY_STEPS).
Default actual learning rate: 3e-5; override with ACTOR_LR or a Hydra override.
Export ARK_API_KEY for the hybrid tool simulator; optionally SWANLAB_API_KEY.
DRY_RUN=1 previews without training, GPU work, output directories or API calls.
EOF
            exit 0 ;;
        --refresh) shift ;;  # Accepted for compatibility; this is already a fresh run.
        --resume|--resume=*)
            echo 'ERROR: 0928 is a fresh-training launcher; use q38_27B_session_v5_grpo.sh --resume PATH for continuation.' >&2
            exit 2 ;;
        *) training_args+=("$1"); shift ;;
    esac
done
for arg in "${training_args[@]}"; do
    option=${arg#+}; option=${option#+}
    case "$option" in
        trainer.resume_mode=*|trainer.resume_from_path=*|trainer.default_local_dir=*|trainer.del_local_ckpt_after_load=*)
            echo 'ERROR: 0928 starts fresh; checkpoint resume overrides are not accepted.' >&2
            exit 2 ;;
    esac
done
set -- "${training_args[@]}"
cd "${REPO_ROOT}"
export TZ=Asia/Shanghai
export RECIPE_ROOT=${RECIPE_ROOT:-/shared/users/yangyq/data/recipe_v5_session_agentloop}
export MODEL_PATH=${MODEL_PATH:-/shared/users/xiongf/ckpts/full-sft-v13-person}
export PYTHON=${PYTHON:-/shared/users/yangyq/env/verl/bin/python}
export PYTHONPATH="${RECIPE_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SESSION_PROFILE=${SESSION_PROFILE:-full}
export PROJECT_NAME=${PROJECT_NAME:-GRPO-Qwen3.8}
if [[ -z "${EXPERIMENT_NAME:-}" || "${EXPERIMENT_NAME}" == verl-grpo-example ]]; then
    export EXPERIMENT_NAME=Qwen3.8-27B-V5-GRPO-0928-MediaHard-${SESSION_PROFILE}
fi

# This preset intentionally overrides inherited endpoints from older 2.1 runs.
# V5 JsonService enforces thinking=enabled and reasoning_effort=low for both roles.
# 默认仅规则评分；ENABLE_LLM_JUDGE=true 恢复最终 LLM Judge。工具 mock 独立。
export ENABLE_LLM_JUDGE=${ENABLE_LLM_JUDGE:-false}
export ENABLE_BENCH_JUDGE=${ENABLE_BENCH_JUDGE:-false}
# Local current-policy regression probes; no judge/mock API calls. 0 disables.
export REPEAT_BENCH_EVERY_STEPS=${REPEAT_BENCH_EVERY_STEPS:-5}
export JUDGE_MODEL=ep-20260513201344-xl7ph
export TOOL_MOCK_MODEL=ep-20260513201344-xl7ph
export TOOL_MOCK_MODE=${TOOL_MOCK_MODE:-hybrid}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export ROLLOUT_N=${ROLLOUT_N:-4}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
export PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
export AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-4}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
export FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-True}
export SAVE_FREQ=${SAVE_FREQ:-5}
export ACTOR_LR=${ACTOR_LR:-3e-5}
export RESUME_MODE=disable
# V5 keeps bounded recovery; per-fixture GT rules are the default reward.
# Equal-score groups remain in the batch for this GRPO baseline.

"${PYTHON}" - <<'PY_PREFLIGHT'
import json
import sys
from collections import Counter
from omegaconf import OmegaConf
from tf_rl.common import ROOT, digest, read_jsonl
from tf_rl.configuration import apply_recipe_config, validated_agent_options
from tf_rl.media_policy import MEDIA_TOOLS, GT_VERSION, VERSION
from tf_rl.quality_rules import REWARD_VERSION
from tf_rl.service import JsonService
sys.path.insert(0, str(ROOT))
from scripts.upgrade_media_direct import original_media_messages

manifest = json.loads((ROOT / 'manifest.json').read_text())
expected = {
    'reward_version': 'v5-reward-4-media-hard',
    'media_policy_version': 'media-direct-v2-original-system',
    'splits': {'train': 679, 'val': 70},
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(f'ERROR: 0928 requires {key}={value}, got {manifest.get(key)}')
if REWARD_VERSION != expected['reward_version'] or VERSION != expected['media_policy_version']:
    raise SystemExit('ERROR: recipe code and data versions do not match the 0928 preset')
annotations = read_jsonl(ROOT / 'data/gt_annotations.jsonl')
if digest(annotations) != manifest['gt_annotations_sha256']:
    raise SystemExit('ERROR: GT annotations do not match the manifest hash')
media = [r for r in annotations if r['category'] in MEDIA_TOOLS]
if Counter(r['category'] for r in media) != {'文生图': 4, '文生视频': 10, '文生音乐': 13}:
    raise SystemExit('ERROR: expected 27 original-system media hard cases')
for row in media:
    f = json.loads((ROOT / 'fixtures' / (row['session_id'] + '.json')).read_text())
    if f['initial_messages'] != original_media_messages(f):
        raise SystemExit('ERROR: media system/user differs from original source: ' + row['session_id'])
    if f['gt'] != row['gt'] or f['gt']['version'] != GT_VERSION or not f['gt'].get('media_direct'):
        raise SystemExit('ERROR: media direct-call GT mismatch: ' + row['session_id'])
options = validated_agent_options()
if options['enable_tool_audit'] or options['enable_user_audit']:
    raise SystemExit('ERROR: V5 GRPO requires per-tool and per-user audits disabled')
config = apply_recipe_config(OmegaConf.create({}))
for role in ('judge', 'simulator'):
    service = JsonService(role)
    if service.config['model'] != 'ep-20260513201344-xl7ph':
        raise SystemExit(f'ERROR: {role} did not load the expected Doubao 2.0 endpoint')
print(f"[0928] reward={REWARD_VERSION}; train=679 val=70; original-system media=27", flush=True)
print(f"[0928] LLM judge={config.recipe_enable_llm_judge}; bench judge={config.recipe_enable_bench_judge}; "
      f"repeat mini bench: at_start=True every_steps={config.recipe_repeat_bench.every_steps}", flush=True)
PY_PREFLIGHT
if [[ "${DRY_RUN:-0}" != 1 && -z "${ARK_API_KEY:-}${JUDGE_API_KEY:-}${TOOL_MOCK_API_KEY:-}" ]]; then
    echo 'ERROR: export ARK_API_KEY (or service-specific keys) for the hybrid tool simulator before training' >&2
    exit 1
fi

exec bash "${REPO_ROOT}/examples/grpo_trainer/run_qwen38_27b_lora_grpo_fa_session.sh" \
    'actor_rollout_ref.model.exclude_modules="(^|.*[.])(vision_tower|visual)([.].*|$)"' \
    algorithm.filter_groups.enable=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.lr_scheduler_type=wsd \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=False \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.gdn_prefill_backend="${GDN_PREFILL_BACKEND:-triton}" \
    ++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_LOGGING_LEVEL=INFO \
    ++recipe_enable_llm_judge="${ENABLE_LLM_JUDGE}" \
    ++recipe_enable_bench_judge="${ENABLE_BENCH_JUDGE}" \
    ++recipe_repeat_bench.every_steps="${REPEAT_BENCH_EVERY_STEPS}" \
    ++recipe_repeat_bench.at_start=True \
    "$@"
