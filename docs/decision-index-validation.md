# Decision Index validation: PQNLD + Qwen on the official suite

**Stage / revision:** RV / 4
**Status:** historical sample pipeline and official offline rescoring complete; Apollo artifacts and provenance verified. Fresh repaired-release sample also completed 100/100; current native metrics and limitations are in [the quality report](decision-index-quality.md).
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
| full sample-100 | full-parity Rust (`0.90`) | 100 | **100 ok**; Python vs Rust: zero choice changes, max absolute probability diff 5.6e-16; ULP distance not established |

In the exact-ID sample run, split-merge removed both failure modes: no question
was refused for overlapping keys and no choice question used echo, avoiding the
observed OOM at the original `--gpu-memory-utilization 0.90`. This requires the
opt-in exact-ID descriptor; it does not describe the shipped descriptor's fallback.

Apollo artifact verification on 2026-10-02 found suite edition `release-v2.1`,
official scorer `0.2.1` at `87d4650b42b377c0291a89c1f1a879f9b31082bf`, and
matching uncompressed corpus hashes. The actual manifest lists 119,898 scoreable
base requests plus 30,419 added requests: **150,317**, not the previously reported
35,714. The verified sample contains 100 rows; this count alone is not the
official weighted coverage metric.

Saved `http-uds-v7`, `http-uds-seq1`, and `http-uds-seq2` summaries report
`complete: false`, **decision_index 0.0**, raw_index 0.11, and weighted coverage
0.0031. The prior null-index claim was incorrect. A coverage-adjusted aggregate
of 0.0 is not sample accuracy. Official offline rescoring is complete; these
saved outputs measure historical binaries. The separately authorized fresh repaired
run matched all 760 historical v7 answers; it demonstrates observed parity, not
an accuracy improvement or full-suite quality.

The 9 errors are PQNLD's own refusals for echo (prefix-overlapping) keys, which
the kit records as errors because the message does not match its capacity markers.
Every exact-token question the engine was asked answered.

## Why the engine died

The echo fallback posts `/v1/completions` with `echo=true`, so
vLLM computes **prompt logprobs for every prompt token**. Long state x many
options allocates large per-token logits and OOMs EngineCore in
`compute_prompt_logprobs` -> `logits_processor._gather_logits` ->
`tensor_model_parallel_all_gather`. The exact-ID lettered path requests
`/v1/chat/completions` with `max_tokens=1` and does not request prompt logprobs;
large label sets require multiple requests. This avoids the observed prompt-logprob
allocation failure, not every possible engine failure.

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
requests and the returned absolute logprobs are merged. Overlapping IDs matched
exactly in historical live checks; correct merging requires identical conditioning
and comparable full-vocabulary scores across requests. The suite's maximum is 151
options; the 255-option wire ceiling exceeds the default 208 candidate labels.

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
and Rust had **zero choice changes** on the full 100-row sample with global max
absolute probability difference `5.6e-16`. The historical comparison counted 27
rows as different by exact string comparison; an absolute difference does not
establish a one-ULP bound, and no ULP-distance receipt is available here.

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

`make test` covers the adapters, readout, and float formatter without a model.

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

- The fresh repaired-release sample measures native quality on the same tiny
  subset; it does not establish calibration or broad model superiority. Historical
  saved outputs remain distinct evidence.
- The `0.80` diagnostic workaround is no longer needed; the original container
  runs the full sample at `0.90`.
- Questions within a request are sequential by default; two isolated historical
  sample runs were identical. The repaired sidecar now enforces a global worker
  budget across its clients and probes, not ordinary chat sent directly to the
  engine. Concurrency (`--workers`) adds within-request parallelism.
  Sequential execution is still not guaranteed reproducible under simultaneous
  real chat load.
- To obtain a real `decision_index`, run a large share of the suite against a
  candidate engine and restore afterward.
- Artifacts on Apollo: `/work/runs/{http-80,http-le26,http-uds,http-uds2,http-uds-seq1,http-uds-seq2,http-py-ref1,http-py-ref3,http-uds-v7}/`,
  `/work/run-80.log`, `/work/run-32k.log`.
