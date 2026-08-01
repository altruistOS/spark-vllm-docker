#!/bin/bash
# =============================================================================
# DeepSeek V4 Flash 0731 DSpark — launch-cluster.sh example
#
# Cluster-only launch script for use with launch-cluster.sh's --launch-script
# mode. Mirrors recipes/deepseek-v4-flash-dspark-0731.yaml. launch-cluster.sh
# injects the multi-node args (--nnodes --node-rank --master-addr
# --master-port [--headless]), so --distributed-executor-backend mp is omitted.
# =============================================================================

HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/root/.cache/huggingface/vllm-cache}"
FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-/root/.cache/huggingface/flashinfer}"

export HF_HOME
export VLLM_CACHE_ROOT
export FLASHINFER_WORKSPACE_BASE

export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
export VLLM_USE_B12X_MOE="${VLLM_USE_B12X_MOE:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB="${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-256}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"
export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_121a}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export TILELANG_CLEANUP_TEMP_FILES="${TILELANG_CLEANUP_TEMP_FILES:-1}"
export DG_JIT_USE_NVRTC="${DG_JIT_USE_NVRTC:-0}"
export DG_JIT_NVCC_COMPILER="${DG_JIT_NVCC_COMPILER:-/usr/local/cuda/bin/nvcc}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# These are injected by launch-cluster.sh's make_node_script:
# --nnodes N --node-rank RANK --master-addr IP --master-port PORT [--headless]
exec vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
    --served-model-name "${SERVED_MODEL_NAME:-deepseek-v4-flash-0731}" \
    --trust-remote-code \
    --port "${VLLM_PORT:-8888}" \
    --host "${VLLM_HOST:-0.0.0.0}" \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 1 \
    --kv-cache-dtype nvfp4_ds_mla \
    --block-size 256 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}" \
    --max-model-len "${MAX_MODEL_LEN:-1048576}" \
    --max-num-seqs "${MAX_NUM_SEQS:-6}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
    --max-cudagraph-capture-size "${CUDA_GRAPH_CAPTURE_SIZE:-36}" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --async-scheduling \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
    --moe-backend flashinfer_b12x \
    --attention-backend flashinfer \
    --enable-flashinfer-autotune \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --tokenizer-mode deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":" thinking","reasoning_end_str":" response"}' \
    --default-chat-template-kwargs '{"thinking":false}' \
    --generation-config vllm
