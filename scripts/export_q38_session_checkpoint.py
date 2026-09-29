# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Export a session FSDP checkpoint as a standalone HF model with LoRA merged."""

import argparse
import gc
import json
import os
import re
import shutil
import tempfile
from pathlib import Path


def export_lora_only(state, actor, output, base_model, hf_source):
    """Stream a single-rank, standard linear LoRA checkpoint into HF shards.

    Keep only a few GB of weights resident, including when the base model has
    very large safetensors shards. Match PEFT's CPU BF16 merge arithmetic.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    from verl.model_merger.output_validation import validate_hf_model_output

    meta = json.loads((actor / "lora_train_meta.json").read_text())
    if set(meta) - {"r", "lora_alpha", "task_type"}:
        raise ValueError("Streaming export supports standard LoRA metadata only")
    rank, alpha = int(meta["r"]), float(meta["lora_alpha"])
    if rank <= 0 or not 0 < alpha < float("inf"):
        raise ValueError("Expected positive, finite LoRA rank and alpha")
    pairs = {}
    for key, tensor in state.items():
        match = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])\.default\.weight", key)
        if match is None:
            raise ValueError(f"Unsupported adapter parameter: {key}")
        if hasattr(tensor, "to_local"):
            local = tensor.to_local()
            if local.shape != tensor.shape:
                raise ValueError(f"Expected complete single-rank tensor: {key}")
            tensor = local
        if tensor.ndim != 2 or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid LoRA tensor: {key}")
        pairs.setdefault(match[1] + ".weight", {})[match[2]] = tensor

    index_path = base_model / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
    else:
        with safe_open(base_model / "model.safetensors", framework="pt", device="cpu") as source:
            weight_map = dict.fromkeys(source.keys(), "model.safetensors")
    for key, pair in pairs.items():
        if key not in weight_map or set(pair) != {"A", "B"}:
            raise ValueError(f"Missing base weight or incomplete LoRA A/B pair: {key}")
        if pair["A"].shape[0] != rank or pair["B"].shape[1] != rank:
            raise ValueError(f"LoRA rank mismatch: {key}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.export-", dir=output.parent) as temporary:
        merged = Path(temporary) / "merged"
        merged.mkdir()
        pending, pending_bytes, total_bytes = {}, 0, 0
        output_map, shards = {}, []

        def flush():
            nonlocal pending_bytes
            if not pending:
                return
            name = f"part-{len(shards) + 1:05d}.safetensors"
            save_file(pending, merged / name, metadata={"format": "pt"})
            output_map.update(dict.fromkeys(pending, name))
            shards.append(name)
            print(f"Wrote shard {len(shards)} ({pending_bytes / 1e9:.2f} GB)", flush=True)
            pending.clear()
            pending_bytes = 0

        print(f"Streaming {len(pairs)} LoRA modules into base: {base_model}", flush=True)
        with torch.no_grad():
            for filename in sorted(set(weight_map.values())):
                source_path = (base_model / filename).resolve()
                if not source_path.is_relative_to(base_model):
                    raise ValueError(f"Unsafe base shard path: {filename}")
                with safe_open(source_path, framework="pt", device="cpu") as source:
                    for key in sorted(k for k, v in weight_map.items() if v == filename):
                        weight = source.get_tensor(key).clone()
                        if key in pairs:
                            a, b = pairs[key]["A"], pairs[key]["B"]
                            if weight.shape != (b.shape[0], a.shape[1]):
                                raise ValueError(f"Base/LoRA shape mismatch: {key}")
                            # PEFT casts CPU half adapters to FP32 for matmul,
                            # then casts the scaled delta back before addition.
                            delta = ((b.float() @ a.float()) * (alpha / rank)).to(a.dtype)
                            weight.add_(delta.to(weight.dtype))
                            del delta
                            if not torch.isfinite(weight).all():
                                raise ValueError(f"Non-finite merged weight: {key}")
                        size = weight.numel() * weight.element_size()
                        if pending_bytes + size > 2_000_000_000:
                            flush()
                        pending[key] = weight
                        pending_bytes += size
                        total_bytes += size
                        del weight
                flush()
        for i, old_name in enumerate(shards, 1):
            new_name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
            (merged / old_name).rename(merged / new_name)
            output_map = {k: new_name if v == old_name else v for k, v in output_map.items()}
        (merged / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": output_map}, indent=2) + "\n"
        )
        # Base supplies processor assets absent from some training checkpoints;
        # the checkpoint's tokenizer/config/generation settings take precedence.
        artifact_names = {
            "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
            "tokenizer.model", "preprocessor_config.json", "processor_config.json",
            "video_preprocessor_config.json", "chat_template.jinja", "chat_template.json",
        }
        for source_dir in (base_model, hf_source):
            for name in artifact_names:
                if (source_dir / name).is_file():
                    shutil.copy2(source_dir / name, merged / name)
            if (source_dir / "chat_templates").is_dir():
                shutil.copytree(source_dir / "chat_templates", merged / "chat_templates", dirs_exist_ok=True)
        validate_hf_model_output(merged)
        (merged / "merge_provenance.json").write_text(json.dumps({
            "checkpoint": str(actor), "base_model": str(base_model),
            "lora_modules": len(pairs), "lora_rank": rank, "lora_alpha": alpha,
            "method": "standard linear LoRA; CPU PEFT-compatible merge arithmetic",
        }, indent=2) + "\n")
        os.rename(merged, output)
    print(f"Deployable model: {output}", flush=True)


def export_checkpoint(checkpoint, output, base_model):
    import torch
    from peft import PeftModel

    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger
    from verl.model_merger.output_validation import validate_hf_model_output
    from verl.utils import hf_processor, hf_tokenizer

    checkpoint, output, base_model = (Path(p).expanduser().resolve() for p in (checkpoint, output, base_model))
    actor = checkpoint if checkpoint.name == "actor" else checkpoint / "actor"
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if output.is_relative_to(checkpoint) or output.is_relative_to(base_model):
        raise ValueError("Export outside the checkpoint and base model directories")
    for filename in ("fsdp_config.json", "lora_train_meta.json"):
        if not (actor / filename).is_file():
            raise FileNotFoundError(actor / filename)
    hf_source = actor / "huggingface"
    if not (hf_source / "config.json").is_file():
        hf_source = base_model
    if not (hf_source / "config.json").is_file():
        raise FileNotFoundError(hf_source / "config.json")

    fsdp_config = json.loads((actor / "fsdp_config.json").read_text())
    world_size = fsdp_config["world_size"]
    state = torch.load(actor / f"model_world_size_{world_size}_rank_0.pt", map_location="cpu", weights_only=False)
    if state and all("lora_" in key for key in state):
        if world_size != 1:
            raise ValueError("LoRA-only streaming export currently requires world_size=1")
        return export_lora_only(state, actor, output, base_model, hf_source)
    del state

    output.parent.mkdir(parents=True, exist_ok=True)
    # Intermediate and final weights coexist temporarily. Publish only after both
    # stages succeed; no files are written into the training checkpoint.
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.export-", dir=output.parent) as temporary:
        unmerged, merged = Path(temporary) / "unmerged", Path(temporary) / "merged"
        config = ModelMergerConfig(
            operation="merge",
            backend="fsdp",
            local_dir=str(actor),
            hf_model_config_path=str(hf_source),
            target_dir=str(unmerged),
            use_cpu_initialization=True,
        )
        merger = FSDPModelMerger(config)
        model_class = merger.get_transformers_auto_model_class()
        print(f"Converting FSDP checkpoint: {actor}; config/tokenizer: {hf_source}", flush=True)
        merger.merge_and_save()
        del merger
        gc.collect()

        adapter = unmerged / "lora_adapter"
        if not (adapter / "adapter_model.safetensors").is_file():
            raise ValueError("Expected LoRA weights in this session checkpoint; refusing a base-only export")
        print("Merging trained LoRA into BF16 base weights on CPU...", flush=True)
        model = model_class.from_pretrained(
            unmerged,
            dtype=torch.bfloat16,
            device_map={"": "cpu"},
            attn_implementation="eager",
            local_files_only=True,
        )
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False, autocast_adapter_dtype=False)
        model = model.merge_and_unload(safe_merge=True)
        model.save_pretrained(merged, safe_serialization=True, max_shard_size="5GB")
        del model
        gc.collect()
        tokenizer = hf_tokenizer(str(unmerged))
        tokenizer.save_pretrained(merged)
        processor = hf_processor(str(unmerged))
        if processor is not None:
            processor.save_pretrained(merged)
        validate_hf_model_output(str(merged))
        os.rename(merged, output)
    print(f"Deployable model: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="global_step_N directory, or its actor subdirectory")
    parser.add_argument("output", help="New output directory for standalone HF model")
    parser.add_argument(
        "--base-model",
        default="/shared/users/xiongf/ckpts/full-sft-v13-person",
        help="Exact training base model; also supplies missing config/tokenizer files",
    )
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output, args.base_model)


if __name__ == "__main__":
    main()
