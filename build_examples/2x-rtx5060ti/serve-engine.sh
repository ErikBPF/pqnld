#!/usr/bin/env bash
# Serve the 2x RTX 5060 Ti reference profile: one vLLM engine, TP=2, that
# answers both chat completions and decisions.
#
# This is the profile the receipts in results/ were measured with. The numbers
# and the reasoning behind the flags are in README.md and docs/benchmarks.md.
#
#   MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4 ./serve-engine.sh
#
# The measured runs additionally enabled the LMCache MP connector (KV offload);
# it helps long-context restores and does not change the short decision path,
# so it is left out here to keep the example self-contained.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the model snapshot directory}
SERVED_NAME=${SERVED_NAME:-qwen38-27b-nvfp4}
PORT=${PORT:-11542}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-200000}
GPU_UTIL=${GPU_UTIL:-0.90}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-1727}
NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-6}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-fp8}

# Pinned so the flags and the results line up. Bump deliberately.
VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai@sha256:5f5e535216848d0c52159c8c13a0af04be5f6fe1a84e79914300610796f76d40}

exec docker run --rm --gpus all --ipc=host --network=host \
    -v "$MODEL_DIR:/model:ro" \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    -e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864 \
    "$VLLM_IMAGE" \
    --model /model --served-model-name "$SERVED_NAME" --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size 2 --no-enable-flashinfer-autotune --disable_custom_all_reduce \
    --trust-remote-code --kv-cache-dtype "$KV_CACHE_DTYPE" --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --gpu_memory_utilization "$GPU_UTIL" \
    --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype bfloat16 --mamba-cache-mode align \
    --enable-prefix-caching --enable-chunked-prefill \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS}" \
    --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
    --default-chat-template-kwargs '{"preserve_thinking":true}' \
    --limit-mm-per-prompt '{"video":0}' --mm-processor-kwargs '{"max_pixels":1000000}'
