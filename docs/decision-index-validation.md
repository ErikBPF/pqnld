# Decision Index validation: PQNLD + Qwen on the official suite

**Stage / revision:** RV / 4
**Status:** pipeline runs end to end; the >26-option echo path is replaced by split exact-ID scoring; the Rust sidecar is the canonical drop-in and matches Python to one ULP on the full sample; no meaningful index yet (sample too small); decisions are sequential by default
**Owner / date:** PQNLD / 2026-10-01 UTC

Goal: run PQNLD's readout against the Decision Index to see where it stands.
The Space `multimodalart/jev-decision-index` is `sdk: static` — a leaderboard
that renders `data/index.json`; it cannot run a model. The runnable grader is the
official kit <https://github.com/apolinario/decision-index>, used here directly.

## Result

| Run | Engine | Rows | Outcome |
|---|---|---|---|
| full sample-100 | `--gpu-memory-utilization 0.90` (original) | 100 | Engine **died** (CUDA OOM) on the echo path; run unusable |
| full sample-100 | `0.80` + `--max-model-len 32768` | 100 | Completed: **91 ok / 9 error**; engine survived |
| sample-le26 (all questions <= 26 options) | restored original `0.90` | 89 | Completed: **89 ok** |
| full sample-100 | split-merge exact-ID (original `0.90`) | 100 | Completed: **100 ok**; engine healthy after |
| full sample-100 | split-merge + Rust/UDS sidecar (`0.90`) | 100 | Completed: **100 ok**, ~162 s concurrent |
| full sample-100 | full-parity Rust (`0.90`) | 100 | **100 ok**; Python vs Rust: zero choice changes, max probability diff 5.6e-16 (1 ULP) |

The split-merge readout removes both failure modes: no question is refused for
overlapping keys and the echo path is never used for choice questions, so the
engine no longer OOMs even at the original `--gpu-memory-utilization 0.90`.

`index.json` `decision_index` is **null** with coverage ~0.0028
(100 / 35714 rows). A 100-row sample is too small for a non-null index; a large
share of the suite must be scored before any headline number exists.

The 9 errors are PQNLD's own refusals for echo (prefix-overlapping) keys, which
the kit records as errors because the message does not match its capacity markers.
Every exact-token question the engine was asked answered.

## Why the engine died

The echo fallback posts `/v1/completions` with `echo=true`, so
vLLM computes **prompt logprobs for every prompt token**. Long state x many
options allocates large per-token logits and OOMs EngineCore in
`compute_prompt_logprobs` -> `logits_processor._gather_logits` ->
`tensor_model_parallel_all_gather`. The exact-ID lettered path is one
`/v1/chat/completions` with `max_tokens=1` and never computes prompt logprobs, so
it is safe. This is an engine-capacity failure on the echo path, not PQNLD logic.

Levers found:

- Lowering `--max-model-len` does **not** free VRAM: vLLM sizes the KV pool from
  `--gpu-memory-utilization`, not from the context length.
- The prompt-logprob chunk size is hardcoded (`prompt_logprob.py`
  `compute_prompt_logprobs_with_chunking`, `CHUNK_SIZE = 1024`) and is not
  controlled by `--max-num-batched-tokens`.
- `--gpu-memory-utilization 0.80` leaves enough headroom for the echo path to
  complete the full sample without killing the engine.

More sidecar RAM cannot fix this (the sidecar is a small process and the
failure is GPU VRAM in the vLLM worker).

## Fix and determinism

>26-option choice questions are relabelled with tokenizer-verified single-token
symbols (`a-z` first; then ASCII letters/digits/punctuation plus Greek and
Cyrillic, 206 distinct measured) and scored by the answer-slot exact-ID readout.
Because the engine caps `logprob_token_ids` at 128 (hardcoded
`MAX_LOGPROB_TOKEN_IDS`), one rendered prompt is split across `ceil(n/128)`
requests and the returned logprobs are merged; the merge is lossless (overlapping
IDs matched exactly in live checks). The suite's maximum is 151 options.

`sidecar-rs/` is the implementation (Rust-only): a full-parity drop-in (lettered
and echo readouts, `auto` probe, LRU cache, chat shim, descriptors, env/flags,
and an MCP stdio server via `--mcp`). It reproduced the historical Python
sequential run 100/100 on the sample. The Python Decision Index UDS client engine
was removed in the Rust-only port; the binary's `--uds` socket remains available
to an equivalent client.

Run-to-run reproducibility depends on question concurrency, and the cause is the
engine, not the client:

- Batched decode is **not numerically identical to single-stream decode**:
  identical prompts get different answer-slot logits depending on batch
  slot/position (batch-shape-dependent kernels + fp8 KV). This is reproducible
  for a given batching layout and persists with **MTP disabled** — with MTP off,
  a row of 16 identical questions still returned the same 4 distinct values.
- **MTP speculative decoding** adds run-to-run variation when independent
  concurrent requests are batched: 8 concurrent identical requests gave 2
  signatures with MTP on and 1 with MTP off.
- With **sequential** execution (`--workers 1`) two full runs were 100/100
  identical.

The deltas are tiny in absolute logprob terms and only flip near-ties. Sequential
costs ~40% wall time on the 100-row sample (219 s vs ~155 s concurrent), so the
default is sequential and concurrency is opt-in; a Decision Index score must be
run sequentially.

## Python-Rust parity (historical)

Before the Python implementation was removed, the multi-engine refactor
(`sidecar-rs/src/engine.rs`) exposed two real Rust defects; once fixed, Python
and Rust agreed to **one ULP** on the full 100-row sample with **zero choice
changes** (global max probability diff `5.6e-16`; 27 rows differ only in the last
float bit, which `agree.py`'s exact-string comparison counts as different).

1. **Float rendering in the prompt.** Python's `repr`/`json.dumps` writes small
   floats in scientific notation (`7.249627201966407e-05`) while Rust
   `serde_json` wrote plain decimal (`0.00007249627201966407`), and
   `serde_json`'s default float parser is not correctly rounded. Since the
   rendered state goes verbatim into the prompt, the model saw different text.
   Fixed with the serde_json `float_roundtrip` feature plus a Python-compatible
   `PyFormatter`/`py_float` in `text()`.
2. **Per-row question order.** `readout_call` spawned one task per question, so
   with `--workers 1` the engine-request order was scheduler-dependent; Python
   uses stored order. Fixed by running questions in stored order when
   `workers == 1`.

`cargo test --release` covers the adapters and the float formatter (10 tests).

The apparent refactor regression first reported (only 5/100 matching the stored
baseline) was engine-state drift: the engine restarted at `2026-10-02T00:08:46Z`
between the baseline and the new run, and the same prompt returns different
logits before/after a restart.

## Reproduction (Apollo)

`W=/mnt/data/ai/validation/decision-index-2026-10-01` on the host; `W` is `/work`
inside the `dibuild` container (shares the engine netns). `W/pm.sh` wraps the
custom podman.

```sh
# suite (built once, gated cais/hle snapshot)
$W/venv/bin/python -m decision_index suite import --dir /work/suite \
  --rows /work/work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
  --added-rows /work/work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz
$W/venv/bin/python -m decision_index suite sample --dir /work/suite --n 100 --out /work/sample-100.jsonl.gz

# sidecar (exact-token descriptor; engine binds the tailnet IP, not localhost)
$W/pm.sh exec -d dibuild bash -c 'pqnld-rs \
  --vllm-url http://100.77.14.27:11542 --model qwen38-27b-nvfp4 \
  --descriptor qwen38-27b-nvfp4 --models-dir /work/desc --host 127.0.0.1 --port 11560'
# descriptor /work/desc/qwen38-27b-nvfp4.json = sidecar-rs/models descriptor + "specific_token_scores": true

# score
$W/venv/bin/python -m decision_index pipeline --engine http \
  --option base_url=http://127.0.0.1:11560 --option model=qwen38-27b-nvfp4 \
  --rows /work/sample-100.jsonl.gz --suite-dir /work/suite --edition 0.2.1 \
  --out /work/runs/http-80

# The Rust sidecar's --uds socket still works, but the Python UDS client engine
# was removed in the Rust-only port; use the kit's http engine against the
# sidecar's HTTP endpoint (as above).
```

The `0.80` + `--max-model-len 32768` engine was only needed to survive the old
echo path. With the split-merge readout the original
`apollo-qwen38-gittensor-lmcache` container runs the full sample at `0.90` and is
left untouched; no temp engine is required.

## Limits and next gates

- No calibration or quality claim: synthetic smoke and the 100-row sample only.
- The `0.80` diagnostic workaround is no longer needed; the original container
  runs the full sample at `0.90`.
- Decisions are sequential by default so results are reproducible. Concurrency
  (`--workers`) trades reproducibility for speed.
  Sequential execution is still not guaranteed reproducible under simultaneous
  real chat load.
- To obtain a real `decision_index`, run a large share of the suite against a
  candidate engine and restore afterward.
- Artifacts on Apollo: `/work/runs/{http-80,http-le26,http-uds,http-uds2,http-uds-seq1,http-uds-seq2,http-py-ref1,http-py-ref3,http-uds-v7}/`,
  `/work/run-80.log`, `/work/run-32k.log`.
