#!/usr/bin/env bash
set -euo pipefail

trial=/mnt/data/ai/validation/20260922-qwen38-repro
engine=${ENGINE_NAME:-apollo-qwen38-gittensor-lmcache}
sidecar_port=${SIDECAR_PORT:-11560}
podman=(sudo -n /mnt/microvms/ai-evaluation/runtimes/tools/podman/bin/podman
    --root /mnt/microvms/ai/cache/podman-root --runroot /run/apollo-ftw-containers)

[[ -f "$trial/decision_sidecar.py" && -f "$trial/s2-rows.jsonl" && -d "$trial/kit/decision_index" ]]

mkdir -p "$trial/results"

echo "preflight: kit import + sidecar compile"
"${podman[@]}" run --rm --entrypoint python3 \
    -v "$trial:/trial:ro" localhost/apollo-qwen38-lmcache:trial -c \
    "import sys; sys.path.insert(0, '/trial/kit'); import decision_index.engines.http, decision_index.runner; print('kit import ok')"
"${podman[@]}" run --rm --entrypoint python3 \
    -v "$trial:/trial:ro" localhost/apollo-qwen38-lmcache:trial -c \
    "compile(open('/trial/decision_sidecar.py').read(), 'decision_sidecar.py', 'exec'); print('sidecar compile ok')"

echo "starting engine"
( cd "$trial" && MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144} GPU_UTIL=${GPU_UTIL:-0.94} L1_GB=${L1_GB:-8} TAG=s2 ./run-lmcache.sh )

echo "starting sidecar inside $engine"
"${podman[@]}" exec -d "$engine" env PYTHONDONTWRITEBYTECODE=1 python3 /trial/decision_sidecar.py \
    --vllm-url http://127.0.0.1:11542 --model qwen38-27b --descriptor qwen38-27b-nvfp4 \
    --host 127.0.0.1 --port "$sidecar_port"

for _ in $(seq 1 30); do
    if "${podman[@]}" exec "$engine" python3 -c \
        "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$sidecar_port/healthz', timeout=2)" 2>/dev/null; then
        break
    fi
    sleep 1
done

echo "running the Decision Index kit (http engine) over the synthetic sample"
"${podman[@]}" exec "$engine" env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/trial/kit python3 -m decision_index run \
    --engine http --option "base_url=http://127.0.0.1:$sidecar_port" --option model=qwen38-27b \
    --rows /trial/s2-rows.jsonl --out /cache/s2-http --fresh \
    > "$trial/results/s2-kit-run.log" 2>&1

"${podman[@]}" exec "$engine" cat /cache/s2-http/results.jsonl > "$trial/results/s2-kit-results.jsonl"
"${podman[@]}" exec "$engine" cat /cache/s2-http/environment.json > "$trial/results/s2-kit-environment.json"
echo "S2 KIT RUN COMPLETE"
