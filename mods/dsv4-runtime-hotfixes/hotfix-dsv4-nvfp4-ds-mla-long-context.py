#!/usr/bin/env python3
"""Hotfix: route nvfp4_ds_mla KV through the fast FP8 MLA kernel path (Issue #22).

The DSpark/Anemll runtime's ``flashmla_sparse.py`` dispatches the attention
KV to the fast ``_forward_fp8_kv`` path only when ``kv_cache_dtype ==
"fp8_ds_mla"``. For ``nvfp4_ds_mla`` the condition is False, so every forward
falls to the slow ``_forward_bf16_kv`` path. The 584-byte KV layout is
identical for both dtypes on DSV4; only the kernel dispatch differs. At long
context (600K+ tokens) the difference is ~16x decode throughput — without this
the advertised 1M context is unusable (long prefills/decode are so slow they
appear to hang). (Issue #22.)

Fix: change the dispatch condition to accept ``nvfp4_ds_mla`` too:
``use_fp8_cache = self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")``.

Idempotent: re-applying is a no-op once the marker is present; ``--status``
reports APPLIED / NOT APPLIED. Patches the container's installed vLLM
``flashmla_sparse.py`` in place (called from the mod entrypoint before
``exec vllm serve``).
"""
from pathlib import Path
import sys

P = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashmla_sparse.py")
OLD = "use_fp8_cache = self.kv_cache_dtype == \"fp8_ds_mla\""
NEW = "use_fp8_cache = self.kv_cache_dtype in (\"fp8_ds_mla\", \"nvfp4_ds_mla\")"
MARK = "  # [dspark-issue22] nvfp4_ds_mla uses fast fp8 KV path"


def apply() -> int:
    if not P.exists():
        print(f"[issue22] FATAL: {P} missing", file=sys.stderr)
        return 1
    text = P.read_text()
    if MARK in text:
        print("[issue22] already applied, skipping")
        return 0
    if NEW in text:
        # already fixed by an upstream image; just mark
        text = text.replace(NEW, NEW + "\n" + MARK)
        P.write_text(text)
        print("[issue22] fast-path present; marked applied")
        return 0
    if OLD not in text:
        print("[issue22] FATAL: target line not found; refusing to patch", file=sys.stderr)
        return 1
    text = text.replace(OLD, NEW + "\n" + MARK, 1)
    P.write_text(text)
    print("[issue22] applied: nvfp4_ds_mla now uses fast fp8 KV kernel path")
    return 0


def status() -> int:
    if not P.exists():
        print("NOT APPLIED")
        return 0
    text = P.read_text()
    if "fp8_ds_mla\", \"nvfp4_ds_mla\"" in text or '("fp8_ds_mla", "nvfp4_ds_mla")' in text:
        print("APPLIED")
        return 0
    print("NOT APPLIED")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        raise SystemExit(status())
    raise SystemExit(apply())
