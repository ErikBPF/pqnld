# PQNLD readout improvements: implementation and live verification

**Stage / revision:** RV + E2E / 13\
**Status:** reliability repairs and JSON compatibility passed 47 Apollo tests in each debug/release profile; independent conformance clear; repaired sample completed 100/100, matching historical v7 answers; MTP shared-chat serving still gated\
**Owner / date:** PQNLD / 2026-10-02 UTC

**Basis:** improve Jev-like typed decisions and normal chat using one resident LLM.
Current candidate: branch `fix/reliability-20261002`, based on `6651289`.
Historical readout candidate: `test/readout-baseline`, based on `f541536`.
**Human correction:** MTP is mandatory. Fast completions are the primary value;
decisions are a lower-priority capability of the same resident model. A separate
model deployment and permanently disabling MTP are excluded; the no-MTP
experiment was diagnostic only. Latest decision: question execution defaults to
**sequential within one request**. Two isolated historical sample runs were
identical; the current reliability repair also bounds sidecar scoring globally,
without promising repeatability under direct engine chat or engine restarts.

## What changed

Current fresh execution and native task metrics are recorded in
[the quality report](decision-index-quality.md). All 760 answers matched
historical v7 in this run; no measured accuracy improvement is claimed.
The tiny official sample still has mixed quality and incomplete suite coverage.

The lettered readout accepts only exact label-shaped tokens instead of interpreting
words as labels and refuses missing option scores. At the reviewed release, echo
could still turn missing evidence into uniform confidence; R2 in
[the reliability plan](reliability-improvements.md) now refuses malformed,
misaligned, empty, or non-finite evidence, with observed Apollo RED/GREEN.
Cache keys preserve option order, matching the rendered prompt.

A descriptor flag, `specific_token_scores`, resolves literal label tokens through
the engine tokenizer and requests those exact IDs with `logprob_token_ids`,
requiring complete returned coverage. It does not bias logits or constrain decoding.

**>26-option exact-ID path.** With `specific_token_scores: true` and a compatible
adapter, choice questions above 26 options avoid echo (whose
prompt-logprob path OOM-crashed the engine and refused prefix-overlapping keys).
They are relabelled with tokenizer-verified single-token symbols — `a-z` for the
first 26, then ASCII letters/digits/punctuation plus Greek and Cyrillic (206
distinct single-token labels measured) — and scored by the same answer-slot
readout. Because the engine caps `logprob_token_ids` at 128, one rendered prompt
is split across `ceil(n/128)` requests and the returned logprobs are merged. The
historical overlap checks matched exactly (max abs logprob diff 0.0). Correct
merging requires absolute, comparable full-vocabulary scores for the same
conditioned slot in every request; it is not a universal backend guarantee.
The suite's largest question is 151 options, so two requests cover it. The wire
ceiling is 255, but the reference tokenizer yielded 206 usable labels from 208
default candidates. Exact-ID scoring is opt-in and absent from the shipped descriptor.

Question execution within a request defaults to sequential (`--workers 1`).
R4 now makes the worker budget global across sidecar requests and scoring probes,
with cancellation/draining on concurrent failure. Raising `--workers` adds
concurrency. Direct engine clients remain outside the budget; neither setting
guarantees engine reproducibility.

## Evidence and limits

| Check | Result | Limit |
|---|---|---|
| Exact-ID live smoke | 10/10 answered and correct; median ~267 ms | Synthetic examples; includes token-ID discovery; no result cache |
| Option permutations | Both orders passed at 2, 3, 10, 20, 26 options | Not unseen-domain quality evidence |
| GPU numerical check of patched scorer | Passed on 2 real GPUs vs PyTorch `log_softmax`; sampled + explicit-ID columns, `-inf` padding, offsets `[0,2,3,6]` | Component check only; no chunked unequal-width, adaptive, TP2 or serving case |
| Decision Index sample (old echo path) | `0.90`: 19 ok / 81 error, engine OOM; `0.80`: 91 ok / 9 error | Historical failures; actual verified edition has 150,317 scoreable requests, not the previously reported 35,714 |
| Decision Index sample (split-merge) | 100/100 ok at `0.90`; engine healthy after | Verified saved summary: index 0.0, raw_index 0.11, weighted coverage 0.0031, incomplete; not sample accuracy or calibration |
| Readout latency | Reported per-question median ~258 ms; reported sidecar overhead ~25-30 ms | Single runs; if comparable, the ratio is ~10-12%, not the previously claimed ~5%; original denominator needs a receipt |
| Rust vs Python sidecar | After the float-rendering and per-row ordering fixes, the full 100-row sample had global max absolute probability diff `5.6e-16` and zero choice changes | Absolute delta alone does not establish one ULP; reported rewrite gain ~1-3%, workload is model-bound |
| Determinism | Root cause is engine batch numerics + MTP, not the client: 16 identical batched questions → 4 distinct answers (persists with MTP off); 8 concurrent identical requests → 2 signatures with MTP, 1 without; sequential → 1 distinct and two full runs 100/100 identical | Sequential still not guaranteed reproducible under simultaneous real chat load |
| Unit regressions (historical Python harness) | 49 tests pass; both benchmark self-tests pass | Canned scores establish plumbing, not calibration; existing socket/resource warnings remain |

## The Rust sidecar (canonical)

`sidecar-rs/` is the implementation: a single static binary with **full parity**
with the historical Python sidecar — exact-ID lettered and echo readouts, `auto`
probe, LRU result cache, the `/v1/chat/completions` shim, descriptors and
env/flags. Build with `cargo build --release` (or `make build`); a musl target
gives a fully static binary. The Python reference implementation was removed in
the Rust-only port; the Rust tests under `sidecar-rs/src` are the reference for
the contract.

**Multi-engine.** `sidecar-rs/src/engine.rs` puts the tokenize/score/top-k/echo
surfaces behind one `Engine` trait with vLLM, llama.cpp, SGLang and OpenAI API
adapters (`--engine-kind`). The vLLM adapter is the only one exercised live;
the others are compile- and parse-unit-tested. Capability tiers and the exact
per-engine logprob limits are in [engine-capabilities.md](engine-capabilities.md).

**Python-Rust parity (historical).** Before the Python implementation was removed,
the two implementations had global max absolute probability difference `5.6e-16`
and zero choice changes on the full 100-row sample. No ULP-distance receipt is
available here.
Two real Rust defects were fixed to get there: a float-rendering difference
(Python `repr` uses scientific notation for small floats, Rust `serde_json` used
plain decimal and was not correctly rounded — fixed with `float_roundtrip` and a
Python-compatible `PyFormatter`), and per-row question order under `--workers 1`
(fixed to stored order). `cargo test --release` covers both.

**Harness plugin.** `pqnld-rs --mcp` runs the MCP stdio server exposing a `decide`
tool, with config snippets for opencode and Codex in
[harness-plugins.md](harness-plugins.md) and `examples/harness/`. Running the
binary itself in MCP mode replaces the old Python bridge process; a vLLM plugin
would bind to vLLM internals.

**Server-side levers** (not yet run) are ranked in
[server-optimizations.md](server-optimizations.md): the engine's 864-token KV block
means the local prefix cache currently achieves ~0% on decision prompts.


## Why batched runs differ from sequential

The divergence is engine-side, not a client bug. Batched decode is not
numerically identical to single-stream decode, so identical prompts get different
answer-slot logits depending on batch slot/position (batch-shape-dependent
kernels + fp8 KV) — reproducible for a given layout and present with **MTP
disabled** (a row of 16 identical questions still returned 4 distinct values with
MTP off). **MTP** additionally adds run-to-run variation when independent
concurrent requests are batched (8 concurrent identical requests: 2 signatures
with MTP, 1 without). Sequential execution produced identical results in the
measured isolated runs; engine restarts and simultaneous chat remain limitations.
The measured deltas were tiny and flipped near-ties.

## Speed opportunities (ranked)

Decisions are model/prefill-bound; a faster sidecar language is worth ~1-3% at
most, so the levers are engine- and request-shape-side.

1. **Hybrid reproducibility.** Run a row concurrently (`--workers > 1`) and re-run
   only near-tie questions sequentially. Recovers most of the ~40% sequential cost
   while keeping the answers that actually depend on the numerics reproducible.
   Not implemented.
2. **One prefill per row.** Score a row's questions as multiple answer slots in one
   prompt (or the upstream fixed-token *prefill* scoring API) so the shared state is
   prefilled once. Largest ceiling; needs engine support and changes conditioning.
   Not implemented.
3. **Engine flag A/B (decision workload).** `--max-num-seqs 16` +
   `--gpu-memory-utilization 0.90` measured ~3% (within noise); untested: remove
   `--disable_custom_all_reduce` (enable custom all-reduce for TP2),
   `cudagraph_mode: FULL` for decode, larger `--max-num-batched-tokens`, async
   scheduling.
4. **Keep prefix reuse with MTP off.** The no-MTP temp engine dropped LMCache (864
   vs 800 block). Reconfigure the cache to 800 so cross-run state prefix reuse
   survives when MTP is off for a decision-only window.
5. **Shorter prompts.** Trim the letter system prompt and instruction boilerplate;
   per-question prefill is mostly the state, but 760 questions amplify every fixed
   token.
6. **Pre-warm label token IDs at startup** (batch the `/tokenize` calls) so the
   first requests do not pay discovery.

Planned verification: send one row's questions as separate concurrent requests with
MTP off (a `same-16` probe) to measure parity against the sequential run and confirm
the MTP-off concurrent value is stable.

Deployability note: MTP stays mandatory for the primary chat workload, so the
measured ~9% "MTP off + bigger prefill batches" gain is a diagnostic result, not a
deployable default, unless vLLM gains per-request speculative control.

## What continues / what waits

The historical candidate MTP patch generator, CPU control-flow checks and the
executed GPU numerical check were under `build_examples/2x-rtx5060ti/` (removed in
the Rust-only port). Next: build a pinned
candidate image from the patched source and rerun concurrent chat/decision checks
plus GPU chunked/adaptive and TP2 gather cases. A sidecar-only lock cannot
coordinate ordinary chat sent directly to the engine, so it is not a sound fix.

The [historical Decision Index sample pipeline](decision-index-validation.md)
and its official offline scores completed. The fresh repaired-release sample
also completed 100/100 with all 760 answers matching historical v7; native task
metrics remain mixed. Apollo artifacts contain index 0.0 rather than the previously
claimed null, with incomplete suite coverage; official scorer/corpus are verified.
Match the actual suite manifest and evaluate held-out coverage, task-specific
official metrics, NLL/Brier where applicable, reliability and risk/coverage before
claiming calibrated decisions. Use supervised decision LoRA only if a measured
quality gap justifies it; the published RLCD recipe is not available for a
faithful reproduction. Keep deployment gated on mixed-load success and quality
claims gated on held-out evidence. See [baseline receipts](readout-baseline.md)
and the [engine diagnosis](mixed-readout-diagnosis.md).
