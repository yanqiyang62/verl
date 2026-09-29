#!/usr/bin/env python3

"""
Patch Hugging Face Transformers FlashAttention integration for FA4/CuTe.

Problem:
    max_seqlen_q / max_seqlen_k may remain 0-d torch.Tensor in eager mode:

        if not isinstance(max_seqlen_q, int) and is_tracing(max_seqlen_q):
            max_seqlen_q = max_seqlen_q.item()

    FA4 CuTe on Blackwell expects max_seqlen_q / max_seqlen_k to be:
        - Python int
        - CuTe Int32
        - None

    Passing torch.Tensor causes:
        DSLRuntimeError:
        expects argument #26 (max_seqlen_q) to be one of
        (Int32, int, NoneType), but got torch.Tensor

Patch:
    Remove the `and is_tracing(...)` restriction so eager execution also
    converts scalar tensors to Python int.

This script is idempotent:
    - already patched -> SKIP
    - old buggy code -> PATCH
    - partially patched -> patch remaining part
    - unknown source layout -> abort instead of modifying blindly
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import py_compile
import shutil
import sys
from pathlib import Path


BAD_Q = (
    "if not isinstance(max_seqlen_q, int) and is_tracing(max_seqlen_q):"
)
GOOD_Q = (
    "if not isinstance(max_seqlen_q, int):"
)

BAD_K = (
    "elif not isinstance(max_seqlen_k, int) and is_tracing(max_seqlen_k):"
)
GOOD_K = (
    "elif not isinstance(max_seqlen_k, int):"
)


def get_transformers_version() -> str:
    try:
        return importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return "UNKNOWN"


def find_target_file() -> Path:
    """
    Locate modeling_flash_attention_utils.py from the CURRENT Python env.
    Do not hardcode venv paths.
    """
    spec = importlib.util.find_spec("transformers")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError(
            "transformers is not installed in the current Python environment."
        )

    package_dir = Path(next(iter(spec.submodule_search_locations)))

    target = package_dir / "modeling_flash_attention_utils.py"

    if not target.is_file():
        raise RuntimeError(
            f"Cannot find Transformers FlashAttention source file:\n{target}"
        )

    return target


def count_state(text: str) -> dict[str, int]:
    return {
        "bad_q": text.count(BAD_Q),
        "good_q": text.count(GOOD_Q),
        "bad_k": text.count(BAD_K),
        "good_k": text.count(GOOD_K),
    }


def main() -> int:
    print("=" * 72)
    print("[FA4 PATCH] Checking Transformers max_seqlen_q/k compatibility")
    print("=" * 72)

    print(f"[FA4 PATCH] Python      : {sys.executable}")
    print(f"[FA4 PATCH] Transformers: {get_transformers_version()}")

    try:
        target = find_target_file()
    except Exception as exc:
        print(f"[FA4 PATCH] ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"[FA4 PATCH] Target      : {target}")

    text = target.read_text(encoding="utf-8")
    state = count_state(text)

    print(
        "[FA4 PATCH] State       : "
        f"bad_q={state['bad_q']}, "
        f"good_q={state['good_q']}, "
        f"bad_k={state['bad_k']}, "
        f"good_k={state['good_k']}"
    )

    # ---------------------------------------------------------
    # Already patched
    # ---------------------------------------------------------
    if (
        state["bad_q"] == 0
        and state["bad_k"] == 0
        and state["good_q"] >= 1
        and state["good_k"] >= 1
    ):
        print("[FA4 PATCH] SKIP: patch is already applied.")
        print("=" * 72)
        return 0

    # ---------------------------------------------------------
    # Safety check
    #
    # Each side must be either:
    #   old form
    # or
    #   already-patched form.
    #
    # Otherwise upstream source probably changed and we should
    # not blindly edit it.
    # ---------------------------------------------------------
    q_known = state["bad_q"] >= 1 or state["good_q"] >= 1
    k_known = state["bad_k"] >= 1 or state["good_k"] >= 1

    if not q_known or not k_known:
        print(
            "[FA4 PATCH] ERROR: Transformers source layout is not recognized.",
            file=sys.stderr,
        )
        print(
            "[FA4 PATCH] Refusing to modify the file automatically.",
            file=sys.stderr,
        )
        print(
            "[FA4 PATCH] Please inspect:",
            target,
            file=sys.stderr,
        )
        return 2

    # Multiple buggy occurrences would be unexpected.
    if state["bad_q"] > 1 or state["bad_k"] > 1:
        print(
            "[FA4 PATCH] ERROR: found multiple matching buggy blocks. "
            "Refusing blind replacement.",
            file=sys.stderr,
        )
        return 3

    # ---------------------------------------------------------
    # Backup once
    # ---------------------------------------------------------
    backup = target.with_suffix(
        target.suffix + ".bak_before_fa4_max_seqlen_patch"
    )

    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"[FA4 PATCH] Backup      : {backup}")
    else:
        print(f"[FA4 PATCH] Backup      : already exists")

    # ---------------------------------------------------------
    # Apply only missing portions
    # ---------------------------------------------------------
    new_text = text
    changes: list[str] = []

    if state["bad_q"] == 1:
        new_text = new_text.replace(BAD_Q, GOOD_Q, 1)
        changes.append("max_seqlen_q")

    if state["bad_k"] == 1:
        new_text = new_text.replace(BAD_K, GOOD_K, 1)
        changes.append("max_seqlen_k")

    if not changes:
        # Normally covered by "already patched", but keep this safe.
        print("[FA4 PATCH] SKIP: nothing needs changing.")
        return 0

    target.write_text(new_text, encoding="utf-8")

    # ---------------------------------------------------------
    # Verify source after writing
    # ---------------------------------------------------------
    verify_text = target.read_text(encoding="utf-8")
    verify = count_state(verify_text)

    if verify["bad_q"] != 0 or verify["bad_k"] != 0:
        print(
            "[FA4 PATCH] ERROR: verification failed; restoring backup.",
            file=sys.stderr,
        )
        shutil.copy2(backup, target)
        return 4

    if verify["good_q"] < 1 or verify["good_k"] < 1:
        print(
            "[FA4 PATCH] ERROR: patched patterns not found; restoring backup.",
            file=sys.stderr,
        )
        shutil.copy2(backup, target)
        return 5

    # Syntax check
    try:
        py_compile.compile(str(target), doraise=True)
    except Exception as exc:
        print(
            f"[FA4 PATCH] ERROR: Python compile check failed: {exc}",
            file=sys.stderr,
        )
        print(
            "[FA4 PATCH] Restoring original file.",
            file=sys.stderr,
        )
        shutil.copy2(backup, target)
        return 6

    print(
        "[FA4 PATCH] PATCHED    : "
        + ", ".join(changes)
    )
    print("[FA4 PATCH] Verification: OK")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())