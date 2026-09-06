#!/bin/bash
#
# mods/dsv4-runtime-hotfixes — DSpark runtime mod: targeted Codex / stability
# hotfixes for the DeepSeek-V4-Flash-Vision-Exp deployment.
#
# Translated from the MiaAI-Lab reference boot sequence
# (DeepSeek-v4-Flash-DSpark-2x-DGX-Spark @ 2660d31) onto the eugr
# single-container recipe/mod surface. This mod applies ONLY the subset of
# upstream runtime hotfixes that matter for a Codex agentic vision path and
# for stability under concurrent / long-prefill load. It deliberately omits the
# pure-throughput hotfixes (skip-topk, mtp-buffer, adaptive-chunk,
# replicate-markov, sp-indexer, deepgemm, ...) to stay lean.
#
#   Staged + run before `vllm serve` on EVERY node:
#     hotfix-encoding-dsv4-issue21.py                     (dict tool args)
#     hotfix-dsv4-issue55-tool-truncation.py              (tool-call truncation)
#     hotfix-vllm-issue138-responses-history.py           (gate: ISSUE138=1)
#     hotfix-vllm-codex-agent-message.py                  (gate: CODEX...=1)
#     hotfix-dsv4-issue27-partial-prefill-concurrency.py  (in-flight prefill cap)
#     hotfix-dsv4-issue43-decode-fairness-and-diag.py     (decode fairness)
#     hotfix-dsv4-issue133-triton-specialization.py       (JIT reliability)
#     hotfix-dsv4-nvfp4-ds-mla-long-context.py            (issue22: nvfp4 fast KV path)
#     hotfix-dsv4-responses-store.py                      (issue62: bounded store, gate: STORE=1)
#     hotfix-dsv4-issue144-effort-align.py                (effort directive -> cross-bucket prefix cache)
#     hotfix-vllm-rope-swa-fix.py                         (sparse-SWA layers use plain RoPE, not YaRN)
#     hotfix-vllm-dspark-swa-prefix.py                    (recompute draft SWA window on prefix-cache hit)
#
# Runs inside the container (CWD = this mod dir) on EVERY node, before exec.
set -euo pipefail

PY_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[dsv4-runtime-hotfixes] $*"; }
die() { log "FATAL: $*" >&2; exit 1; }

HF=(hotfix-encoding-dsv4-issue21.py
    hotfix-dsv4-issue27-partial-prefill-concurrency.py
    hotfix-dsv4-issue43-decode-fairness-and-diag.py
    hotfix-dsv4-issue55-tool-truncation.py
    hotfix-dsv4-issue133-triton-specialization.py
    hotfix-vllm-issue138-responses-history.py
    hotfix-vllm-codex-agent-message.py
    hotfix-dsv4-nvfp4-ds-mla-long-context.py
    hotfix-dsv4-responses-store.py
    hotfix-dsv4-issue144-effort-align.py
    hotfix-vllm-rope-swa-fix.py
    hotfix-vllm-dspark-swa-prefix.py)

# --- 1. Stage hotfix scripts into /opt ---------------------------------------
for f in "${HF[@]}"; do
    [ -f "$MOD_DIR/$f" ] || die "missing $f"
    cp "$MOD_DIR/$f" "/opt/$f"
done
log "staged ${#HF[@]} runtime hotfixes"

# --- 2. Apply in reference order/gate ----------------------------------------
# issue21 needs the Vision-Exp DSV4 encoder present (the vision mod copies it).
ENC="$PY_ROOT/vllm/tokenizers/deepseek_v4_encoding.py"
if [ -f "$ENC" ]; then
    python3 /opt/hotfix-encoding-dsv4-issue21.py
    log "issue21 applied"
else
    log "WARN: $ENC missing (vision mod did not install encoder); skip issue21"
fi

python3 /opt/hotfix-dsv4-issue55-tool-truncation.py
log "issue55 applied"

if [ "${DSPARK_ENABLE_ISSUE138_RESPONSES_HISTORY_COMPAT:-0}" = "1" ]; then
    python3 /opt/hotfix-vllm-issue138-responses-history.py
    log "issue138 applied"
fi
if [ "${DSPARK_ENABLE_CODEX_AGENT_MESSAGE_COMPAT:-0}" = "1" ]; then
    python3 /opt/hotfix-vllm-codex-agent-message.py
    log "codex agent_message applied"
fi

python3 /opt/hotfix-dsv4-issue27-partial-prefill-concurrency.py
log "issue27 applied"
python3 /opt/hotfix-dsv4-issue43-decode-fairness-and-diag.py
log "issue43 applied"
python3 /opt/hotfix-dsv4-issue133-triton-specialization.py
log "issue133 applied"

# issue22: route nvfp4_ds_mla KV through the fast FP8 MLA kernel path.
# Required for usable long (>600K-token) context; without it the slow bf16
# path makes 1M context ~16x slower and appears to hang.
python3 /opt/hotfix-dsv4-nvfp4-ds-mla-long-context.py
log "issue22 applied"

# issue62: bound the Responses terminal store so stateful
# previous_response_id continuation does not grow memory unbounded.
# Applied UNCONDITIONALLY: eugr exports the recipe env block (incl.
# VLLM_ENABLE_RESPONSES_API_STORE=1) only on the `exec vllm serve` line, which
# runs AFTER this mod, so the previous `VLLM_ENABLE_RESPONSES_API_STORE=1` gate
# never saw it here and silently skipped the patch (store was left unbounded).
# The patch is hash-verified/idempotent and only activates its cap logic when
# the serving code's enable_store is true, so applying it unconditionally is safe.
python3 /opt/hotfix-dsv4-responses-store.py
log "bounded responses store applied"

# --- 3. New upstream (>= 957890a) opt-in hotfixes -----------------------------
# These are gated in the reference compose entrypoint via DSPARK_ENABLE_* env.
# In eugr the recipe `env:` block is exported only on the `exec vllm serve`
# line, AFTER this mod runs, so those gates never reach this script.  The three
# patches are source-exact + version-pinned (vLLM 752a3a504, the image's build)
# and idempotent, so applying them UNCONDITIONALLY is safe (issue62 precedent).
# Order: issue144 patches the encoder (must be after issue21 / the vision mod's
# encoder install); rope-swa and dspark-swa-prefix are independent, and
# dspark-swa-prefix patches scheduler.py co-owned by issue27 (already applied).

# issue144: relocate the reasoning-effort directive after the leading system
# block so prefix-cache blocks are shared across effort buckets (high/max/low).
if [ -f "$PY_ROOT/vllm/tokenizers/deepseek_v4_encoding.py" ]; then
    python3 /opt/hotfix-dsv4-issue144-effort-align.py
    log "issue144 effort-align applied"
else
    log "WARN: encoder missing; skip issue144 effort-align"
fi

# rope-swa: sparse-SWA layers use plain RoPE, not YaRN (upstream vllm#54815).
python3 /opt/hotfix-vllm-rope-swa-fix.py
log "rope-swa fix applied"

# dspark-swa-prefix: on a prefix-cache hit, cap the cached length so the DSpark
# draft recomputes its sliding window (Anemll dspark-vllm-gx10#2).
python3 /opt/hotfix-vllm-dspark-swa-prefix.py
log "dspark-swa-prefix applied"


echo "[dsv4-runtime-hotfixes] done"
