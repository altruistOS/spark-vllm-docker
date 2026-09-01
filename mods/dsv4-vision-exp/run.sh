#!/bin/bash
#
# mods/dsv4-vision-exp — DSpark runtime mod: native DeepSeek-V4-Flash-Vision-Exp
# image support for the Anemll dspark-vllm-gx10:0.1.1 image.
#
# Mirrors the Vision-Exp boot sequence from the MiaAI-Lab reference repo
# (feat/vision-exp + fix/as-pil-chw + fix/issue109-empty-encoder-output),
# translated onto the eugr single-container recipe/mod surface:
#   - Stage hotfix-dsv4-vision-exp.py (ViT + Aligner + image_url processor)
#   - Stage the vision_exp/ package into /opt/dspark-patches/vision_exp
#   - Stage the issue109 encoder-only output hotfix
#   - Copy the Vision-Exp DSV4 encoder from the HF cache into vLLM tokenizers
#   - Run the vision hotfix(es) before vllm serve launches
#
# Runs inside the container (CWD = this mod dir) on EVERY node, before exec.
#
set -euo pipefail

PY_ROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
HF_HOME_DIR="${HF_HOME:-/root/.cache/huggingface}"
DSPARK_PATCHES="/opt/dspark-patches"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[dsv4-vision-exp] $*"; }
die() { log "FATAL: $*" >&2; exit 1; }

# --- 1. Stage hotfix scripts + vision_exp package ---------------------------
mkdir -p "$DSPARK_PATCHES/vision_exp"
cp "$MOD_DIR/hotfix-dsv4-vision-exp.py"           /opt/hotfix-dsv4-vision-exp.py
cp "$MOD_DIR/hotfix-vllm-empty-encoder-output.py" /opt/hotfix-vllm-empty-encoder-output.py
cp "$MOD_DIR"/vision_exp/*.py "$DSPARK_PATCHES/vision_exp/"
log "staged hotfixes + vision_exp package to $DSPARK_PATCHES"

# --- 2. Install the Vision-Exp DSV4 encoder into vLLM tokenizers -------------
# Looks in the HF cache snapshot (honours DSPARK_REVISION pin, else tip of main),
# with a fallback to the 0731 encoder to match the reference boot.
enc_src=""
hub_dir="$HF_HOME_DIR/hub/models--deepseek-ai--DeepSeek-V4-Flash-Vision-Exp"
if [ -n "${DSPARK_REVISION:-}" ] && [ -f "$hub_dir/snapshots/$DSPARK_REVISION/encoding/encoding_dsv4.py" ]; then
    enc_src="$hub_dir/snapshots/$DSPARK_REVISION/encoding/encoding_dsv4.py"
else
    for cand in "$hub_dir"/snapshots/*/encoding/encoding_dsv4.py; do
        [ -f "$cand" ] && { enc_src="$cand"; break; }
    done
fi
if [ -z "$enc_src" ]; then
    for cand in "$HF_HOME_DIR"/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/*/encoding/encoding_dsv4.py; do
        [ -f "$cand" ] && { enc_src="$cand"; break; }
    done
fi
if [ -n "$enc_src" ] && [ -f "$enc_src" ]; then
    cp "$enc_src" "$PY_ROOT/vllm/tokenizers/deepseek_v4_encoding.py"
    log "encoder installed: $enc_src"
else
    log "WARN: encoding_dsv4.py not found under $hub_dir (set DSPARK_REVISION / verify HF cache)"
fi

# --- 3. Apply the this-image hotfix + encoder-only output fix ----------------
python3 /opt/hotfix-dsv4-vision-exp.py
python3 /opt/hotfix-vllm-empty-encoder-output.py
log "vision hotfixes applied"
echo "[dsv4-vision-exp] done"
