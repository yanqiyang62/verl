"""Find complete checkpoints for a V4/V5 preset without loading tensors or starting Ray."""

import argparse
import json
import re
import sys
from pathlib import Path


def complete_checkpoint(directory):
    step = (directory / "latest_checkpointed_iteration.txt").read_text().strip()
    if not step.isdecimal():
        raise ValueError("invalid completion marker")
    checkpoint = directory / f"global_step_{step}"
    required = ["data.pt", "actor/fsdp_config.json", "actor/lora_train_meta.json"]
    required += [f"actor/{name}_world_size_1_rank_0.pt" for name in ("model", "optim", "extra_state")]
    for name in required:
        if (checkpoint / name).stat().st_size == 0:
            raise ValueError(f"empty checkpoint file: {name}")
    meta = json.loads((checkpoint / "actor/lora_train_meta.json").read_text())
    if meta.get("r") != 32 or meta.get("lora_alpha") != 64:
        raise ValueError("expected LoRA rank=32 / alpha=64")
    return checkpoint.resolve()


def latest_checkpoint(repo, log_root, strategy, recipe, model, profile):
    candidates = {}
    # Existing fresh runs: read only the relevant configuration lines, never credentials.
    for log in log_root.glob(f"*V[45]-{strategy.upper()}-*/*/train.log"):
        try:
            text = log.read_text(errors="replace")
            model_match = any(s in text for s in (
                f"model.path={model}", f"'path': '{model}'", f"model='{model}'"
            ))
            recipe_match = any(s in text for s in (
                f"recipe_root={recipe}", f"'recipe_root': '{recipe}'", f"{recipe}/tf_rl/main.py:"
            ))
            if not (model_match and recipe_match and f"-{profile}-" in log.parent.parent.name):
                continue
            # A custom output directory can be recorded in Hydra's config or overrides.
            paths = re.findall(r"'default_local_dir': '([^']+)'", text)
            paths += re.findall(r"trainer.default_local_dir=([^'\n]+)'", text)
            directory = Path(paths[-1]) if paths else log.parent / "checkpoints"
            marker = directory / "latest_checkpointed_iteration.txt"
            if marker.is_file():
                candidates[directory.resolve()] = marker.stat().st_mtime_ns
        except OSError as exc:
            print(f"[auto-resume] Cannot inspect {log}: {type(exc).__name__}", file=sys.stderr)
    # Continuations may save outside logs/; the controller registers their destinations.
    for index in (repo / "checkpoints/.session-resume-index").glob("*.json"):
        try:
            record = json.loads(index.read_text())
            if any(record.get(k) != v for k, v in {
                "strategy": strategy, "recipe": str(recipe), "model": str(model), "profile": profile
            }.items()):
                continue
            directory = Path(record["output"])
            marker = directory / "latest_checkpointed_iteration.txt"
            if marker.is_file():
                candidates[directory.resolve()] = marker.stat().st_mtime_ns
        except (OSError, ValueError, KeyError) as exc:
            print(f"[auto-resume] Cannot inspect {index}: {type(exc).__name__}", file=sys.stderr)
    for directory in sorted(candidates, key=lambda p: (candidates[p], str(p)), reverse=True):
        try:
            return complete_checkpoint(directory)
        except (OSError, ValueError, KeyError) as exc:
            print(f"[auto-resume] Skipping incomplete checkpoint at {directory}: {exc}", file=sys.stderr)
    raise ValueError("No complete matching session checkpoint found. Use --resume PATH or --refresh to start fresh.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--strategy", choices=("grpo", "dapo"), required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    try:
        print(latest_checkpoint(args.repo, args.log_root, args.strategy, args.recipe, args.model, args.profile))
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")
