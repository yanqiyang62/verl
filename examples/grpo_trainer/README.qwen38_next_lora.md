# Qwen3.8-Flash-Next LoRA GRPO

This integration imports verl PR #7749 at `4a97a4b5261c42cebe847ae41fdd8b33c019f425`.
It adds FSDP2 initialization, frozen CPU parameter placement/sharing, buffer and
tied-weight preservation, wrap selection, and 3D position-ID fixes.

## Keep the working 27B version

- `qwen38-27b-stable` preserves commit `b365d705` and its original core implementation.
- `qwen38-lora-local` and the original working directory remain at that commit.
- `qwen38-next-lora-fsdp2` contains the integration in a separate worktree.
- Existing 27B launchers, session helpers, FA4 settings, WSD, timeout handling,
  and shared-memory adapter transport are retained in the integration branch.

Use the original working directory for existing 27B jobs. The integration changes
shared FSDP2 code, so unchanged launchers alone do not establish 27B runtime parity.

## Requirements

The default recipe uses two nodes with eight GPUs each, BF16 training and rollout,
FSDP16, vLLM TP4, LoRA rank 16 / alpha 32, CUDA Graph and three-token MTP. Packing
is disabled; dynamic batching is enabled; both KL paths are disabled.

Prepare a **Qwen3.8-Flash-Next BF16 checkpoint** and use the same checkpoint for
training and rollout. Do not point this launcher at a 27B checkpoint or directly
at an FP8 checkpoint. Model-specific Transformers, vLLM, and expert adapter-loading
support must already be installed on every node; this merge does not install
those dependencies or convert checkpoints. An ordinary 27B environment is not
automatically a complete Flash-Next environment.

Use a prepared uv-managed environment. `PYTHON` can select its interpreter;
otherwise the upstream launcher runs `uv run --active --no-sync python`.
Start the Ray cluster using the same environment and integration code on all nodes.

`VERL_NO_PLACEMENT_MMAP_DIR` must name **node-local disk**, accessible at the same
path by all ranks on each host. Do not use shared network storage or a RAM-backed
filesystem for the large CPU embedding. The wrapper forwards this setting to Ray
workers. Allow sufficient disk and host-memory headroom for initialization.

## Launch

Run from the integration worktree:

For a **single-node, four-B300 smoke**, use
`tasks/q38_next_4xb300_lora_grpo_smoke.sh` with the same environment variables
below. This preset runs one step and saves a checkpoint: FSDP4 / TP4, four prompts
with two responses each, 2K prompt + 2K response, no MTP/CUDA Graph, no evaluation
or automatic resume. `VAL_FILE` can be omitted (it reuses `TRAIN_FILE` while
evaluation is disabled). `OUTPUT_DIR` defaults to a new timestamped checkpoint
directory. `RAY_ADDRESS` defaults to `local`; use `auto` only when intentionally
connecting to an existing four-GPU Ray allocation. `DRY_RUN=1` composes the config
without requiring real checkpoint/data paths or starting training. This is a
smoke preset, not a measured guarantee of four-GPU capacity or learning progress.

Single-node resource planning for that smoke (estimates, not measured minima):

| Resource | Initial allocation | More headroom |
| --- | --- | --- |
| GPU | 4 B300, 288 GB nominal HBM each, same NVLink/NVSwitch domain | 8 B300 for larger batches; change topology accordingly |
| Host RAM | 1 TB is a constrained trial; prefer 2 TB | 2 TB for the four-GPU smoke |
| CPU | 128 logical CPUs allocated to the job | 256 logical CPUs |
| Local NVMe | 1 TB free for checkpoint files, mmap and temporary data | 2 TB free if retaining both FP8 source and BF16 conversion |

With SMT2, 128/256 logical CPUs correspond to 64/128 physical cores; confirm the
scheduler's accounting and actual CPU topology. Core counts are throughput
planning assumptions, not a model-imposed minimum. RAM capacity and bandwidth can
matter more than adding cores. The CPU embedding is about 100 GB; per-rank CPU
weight backups and initialization copies consume additional hundreds of GB.
Request host RAM explicitly when sharing an eight-GPU node: allocating four GPUs
does not guarantee access to the whole node's RAM. Keep at least 10–15% memory
headroom and measure initialization, rollout sleep, update and save peaks.

```bash
export PYTHON=/path/to/prepared-uv-env/bin/python
export MODEL_PATH=/models/Qwen3.8-Flash-Next-BF16
export TRAIN_FILE=/data/dapo_math/train.parquet
export VAL_FILE=/data/aime2024/test.parquet
export OUTPUT_DIR=/checkpoints/qwen38-next-grpo
export VERL_NO_PLACEMENT_MMAP_DIR=/local_nvme/verl-frozen
export RAY_ADDRESS=auto

# Config composition only: no GPU work, training, or Ray connection.
DRY_RUN=1 bash examples/grpo_trainer/run_qwen38_next_lora_fsdp2.sh

# One training step in a separate output directory.
bash examples/grpo_trainer/run_qwen38_next_lora_fsdp2.sh \
    trainer.total_training_steps=1 trainer.save_freq=1
```

The upstream `scripts/run_qwen38_lora_dapo.sh` uses the **GRPO** estimator; its
filename refers to the DAPO-Math dataset. The new GRPO entry point reuses that
recipe and adds distinct experiment naming and worker mmap configuration.
Training parquet must contain the fields and reward metadata expected by verl's
dataset and reward manager. Custom session/tool tasks need their own reward and
agent configuration; this example does not automatically adopt the 27B session recipe.

Keep credentials in the environment, never in launch scripts. Use a separate
output directory and model path from 27B jobs. The provided script does not launch
Ray, export adapters, or convert model checkpoints.

## Validation scope

Merge preflight passed the PR's four CPU test files plus the local agent-loop
tests (56 tests). GPU FSDP2 loading, full-size Next training, 27B training under
the new initialization path, and checkpoint resume require separate validation.
No full-size training is started by merging or composing this recipe.

The PR includes a two-GPU regression:

```bash
"$PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 \
    tests/special_distributed/test_fsdp2_full_state_load.py
```
