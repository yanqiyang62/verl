#!/usr/bin/env bash
# Full-state continuation of the B300 4 prompts x 4 rollouts session recipe.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
PYTHON=${PYTHON:-/shared/users/yangyq/env/verl/bin/python}
# Internal dispatch from the V4 entrypoint; retries must preserve its recipe and policy.
training_entrypoint=${SCRIPT_DIR}/q38_27B_session_throughput.sh
v4_strategy=
if [[ ${1:-} == --v4-dapo ]]; then
    v4_strategy=dapo
    training_entrypoint=${SCRIPT_DIR}/q38_27B_session_v4_dapo.sh
    shift
elif [[ ${1:-} == --v5-grpo ]]; then
    v4_strategy=grpo
    training_entrypoint=${SCRIPT_DIR}/q38_27B_session_v5_grpo.sh
    shift
elif [[ ${1:-} == --v4-grpo ]]; then
    v4_strategy=grpo
    training_entrypoint=${SCRIPT_DIR}/q38_27B_session_v4_grpo.sh
    shift
fi

usage() {
    cat <<'EOF'
Usage: bash tasks/q38_27B_session_resume.sh /path/to/checkpoints/global_step_65 [Hydra overrides...]
       bash tasks/q38_27B_session_resume.sh /path/to/checkpoints [Hydra overrides...]
The second form reads latest_checkpointed_iteration.txt; missing state is an error.
Restores model, optimizer, scheduler/RNG and dataloader. Does not start from step zero.
New checkpoints go to RESUME_OUTPUT_DIR (default: a new checkpoints/session-resume-* directory).
RESUME_MAX_RESTARTS=2 retries failed processes from the latest complete saved checkpoint.
RESUME_MAX_RESTARTS=0 disables process retries. Ctrl-C never restarts the job.
DRY_RUN=1 prints the full command without starting training or creating output directories.
EOF
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
RESUME_INPUT=${1:-${RESUME_FROM:-}}
if [[ -z "$RESUME_INPUT" ]]; then usage >&2; exit 2; fi
if [[ $# -gt 0 ]]; then shift; fi
for arg in "$@"; do
    option=${arg#+}
    option=${option#+}
    case "$option" in
        trainer.resume_mode=*|trainer.resume_from_path=*|trainer.default_local_dir=*|trainer.del_local_ckpt_after_load=*)
            echo "ERROR: use the checkpoint argument and RESUME_OUTPUT_DIR; do not override resume control: ${arg%%=*}" >&2
            exit 2 ;;
    esac
done
resolve_checkpoint() {
    "$PYTHON" - "$1" <<'PY'
import json, re, sys
from pathlib import Path
p = Path(sys.argv[1]).expanduser().resolve()
if not re.fullmatch(r'global_step_\d+', p.name):
    tracker = p / 'latest_checkpointed_iteration.txt'
    if not tracker.is_file():
        sys.exit(f'ERROR: missing checkpoint completion marker: {tracker}')
    step = tracker.read_text().strip()
    if not step.isdecimal():
        sys.exit('ERROR: invalid latest_checkpointed_iteration.txt')
    p = p / f'global_step_{step}'
required = ['data.pt', 'actor/model_world_size_1_rank_0.pt', 'actor/optim_world_size_1_rank_0.pt',
            'actor/extra_state_world_size_1_rank_0.pt', 'actor/fsdp_config.json', 'actor/lora_train_meta.json']
for name in required:
    f = p / name
    if not f.is_file() or f.stat().st_size == 0:
        sys.exit(f'ERROR: incomplete single-GPU checkpoint: {f}')
meta = json.loads((p / 'actor/lora_train_meta.json').read_text())
if meta.get('r') != 32 or meta.get('lora_alpha') != 64:
    sys.exit('ERROR: this launcher expects LoRA rank=32 / alpha=64; checkpoint differs')
print(p)
PY
}
current_checkpoint=$(resolve_checkpoint "$RESUME_INPUT")
initial_step=${current_checkpoint##*global_step_}
max_restarts=${RESUME_MAX_RESTARTS:-2}
if [[ ! "$max_restarts" =~ ^[0-9]+$ ]]; then echo 'ERROR: RESUME_MAX_RESTARTS must be >= 0' >&2; exit 2; fi
export SESSION_PROFILE=${SESSION_PROFILE:-full}
export RESUME_MODE=resume_path
export CHECKPOINT_DIR=${RESUME_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/session-resume-${initial_step}-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)-$$}
# Keep source checkpoints intact when rolling back to an older step.
"$PYTHON" - "$current_checkpoint" "$CHECKPOINT_DIR" <<'PY'
import sys
from pathlib import Path
source, output = (Path(p).resolve() for p in sys.argv[1:])
if source.parent == output or output.is_relative_to(source):
    sys.exit('ERROR: RESUME_OUTPUT_DIR must be separate from the source checkpoint directory')
PY
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-Qwen3.8-27B-LoRA-FA4-B300-Session-Resume-step${initial_step}}
export SAVE_FREQ=${SAVE_FREQ:-5}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export ROLLOUT_N=${ROLLOUT_N:-4}
if [[ -n "$v4_strategy" && ${DRY_RUN:-0} != 1 ]]; then
    # Remember custom resume destinations for the next public no-argument launch.
    "$PYTHON" - "$REPO_ROOT" "$v4_strategy" "$CHECKPOINT_DIR" "$current_checkpoint" <<'PY'
import json, os, sys
from pathlib import Path
from uuid import uuid4
repo, strategy, output, source = sys.argv[1:]
index = Path(repo) / 'checkpoints/.session-resume-index'
index.mkdir(parents=True, exist_ok=True)
record = {'strategy': strategy, 'recipe': os.environ.get('RECIPE_ROOT', '/shared/users/yangyq/data/recipe_v4_session_agentloop'),
          'model': os.environ.get('MODEL_PATH', '/shared/users/xiongf/ckpts/full-sft-v13-person'),
          'profile': os.environ['SESSION_PROFILE'], 'output': str(Path(output).resolve()), 'source': source}
target = index / f'{strategy}-{uuid4().hex}.json'
temporary = target.with_suffix('.tmp')
temporary.write_text(json.dumps(record))
temporary.replace(target)
PY
fi
attempt=0
trap 'echo "Resume launcher interrupted; no automatic restart." >&2; exit 130' INT TERM
while true; do
    echo "[resume] Loading: $current_checkpoint"
    echo "[resume] Continuing at step $(( ${current_checkpoint##*global_step_} + 1 )); output: $CHECKPOINT_DIR"
    echo "[resume] Attempt $((attempt+1))/$((max_restarts+1)); completed updates since last checkpoint are replayed after a crash."
    set +e
    SESSION_RESUME_MANAGED=1 bash "$training_entrypoint" "$@" \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="$current_checkpoint" \
        trainer.del_local_ckpt_after_load=False \
        actor_rollout_ref.actor.checkpoint.load_contents='[model,optimizer,extra]' \
        actor_rollout_ref.actor.checkpoint.save_contents='[model,optimizer,extra]'
    status=$?
    set -e
    if [[ "$status" == 0 || ${DRY_RUN:-0} == 1 ]]; then exit "$status"; fi
    if [[ "$status" == 130 || "$status" == 143 || "$attempt" -ge "$max_restarts" ]]; then
        echo "[resume] Stopped with exit $status. Last source: $current_checkpoint; saved checkpoints: $CHECKPOINT_DIR" >&2
        exit "$status"
    fi
    if [[ -f "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" ]]; then
        current_checkpoint=$(resolve_checkpoint "$CHECKPOINT_DIR")
    fi
    attempt=$((attempt+1))
    echo "[resume] Training process exited $status; restarting from $current_checkpoint in 10 seconds." >&2
    sleep 10
done
