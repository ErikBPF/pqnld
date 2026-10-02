# Benchmarks

All numbers are **measured on the reference rig** and are directional, not
guarantees. Your model, GPU, and flags dominate the result.

Rig: 2× RTX 5060 Ti (16 GB each), Qwen3.8-27B NVFP4 (Gittensor), vLLM
tensor-parallel 2, FP8 KV cache, MTP speculative decoding (6 tokens), LMCache
connector, `max_model_len` 200000, `gpu_memory_utilization` 0.90,
`--enable-prefix-caching --enable-chunked-prefill`. The engine image was a
locally built `localhost/apollo-qwen38-lmcache:trial`, based on
`vllm/vllm-openai@sha256:5f5e535216848d0c52159c8c13a0af04be5f6fe1a84e79914300610796f76d40`
plus two LMCache packed-KV patches; pqnld ran from source inside it. The raw
receipts were kept under the Python-era `build_examples/2x-rtx5060ti/` directory,
removed in the Rust-only port.

## The readout ladder

| Configuration | Latency (1 question, warm) |
|---|---|
| Per-option echo readout | ~0.76 s |
| Lettered readout, one prefill | ~0.31 s |
| Lean engine (no MTP, no KV connector), first call | ~0.238 s |
| Lean engine, cache repeat | ~0.0017 s |

`--enforce-eager` and `--max-num-seqs 1` were tried and were worse (~0.27 s),
and rejected.

## Single decision latency (full profile)

| Questions per decision | Warm p50 |
|---|---|
| 1 | 0.119 s |
| 2 | 0.229 s |

pqnld's own overhead is ~0: a one-question decision through pqnld matches a
direct lettered chat call to the engine to the millisecond.

## Concurrent decisions

Against the engine directly (distinct prompts, warm), tiny `max_tokens=1`
requests are admitted in a ramp — roughly 1, then 1, then 2, then 4 per ~0.12 s
step:

| In flight | Wall | Completed at (s) | Parallelism |
|---|---|---|---|
| 2 | 0.233 s | 0.12, 0.23 | 1.5 |
| 4 | 0.486 s | 0.13, 0.23, 0.49, 0.49 | 2.7 |
| 8 | 0.745 s | … 0.24, 0.13, 0.49 ×6 | 5.8 |

The staircase is a scheduler property, not a pqnld one.

## Raising `--max-num-seqs`

Mixed 4 decisions + 4 chats in flight:

| `--max-num-seqs` | Decision p50 | Wall | Throughput | Failures |
|---|---|---|---|---|
| 2 | 2.86 s | 3.10 s | 2.58 /s | 0 |
| 8 | 1.39 s | 1.39 s | 5.74 /s | 0 |

16 in flight at `--max-num-seqs 8`: parallelism 10.5, 5.19 /s, 0 failures.
Raising `--max-num-batched-tokens` (the other lever vLLM suggests) is
**infeasible** at a 200k context in this rig: the KV cache starves and the engine
refuses to start at 4096 and 8192.

## Mixed load: long conversations + decisions

The historical Python `benchmarks/bench_mixed.py`, two ~95k-token conversations
(1024 output tokens) with 100 interleaved one-question decisions, streamed so
prefill (TTFT) and decode are measured separately:

| Phase | Chats | TTFT | Decode TPS | Decisions |
|---|---|---|---|---|
| single | 1 | 78.9 s | 62.4 | — |
| pair | 2 | 78.9 / 159.1 s | 11.1 / 55.6 → **22.1 agg** | — |
| mixed | 2 + 100 | 78.9 / 231.5 s | 6.4 / 59.5 → **12.9 agg** | 100/100 ok, p50 11.1 s, p95 79.3 s, 1.2/s, parallelism 27.5 |
| recovery | 2 | 78.9 / 159.0 s | 11.0 / 58.5 → **22.0 agg** | — |

Read it this way:

- A single ~100k conversation costs ~79 s of prefill (≈1200 tok/s) and then
  decodes at 62 tok/s. Report **decode-only** TPS; total-wall TPS is dominated by
  prefill.
- **Two long conversations do not prefill in parallel** — TTFT₂ is exactly 2×
  TTFT₁, because chunked prefill serializes prompts larger than the step budget.
- Decisions run genuinely in parallel with long generations (parallelism 27.5,
  zero failures), but are delayed behind ~95k prefill chunks: p50 11 s while both
  conversations are active, versus 0.119 s unloaded.
- **When decisions stop, chat throughput returns to 99.6 %** of its
  pre-decision baseline (recovery 22.0 vs pair 22.1).

## What this means for agentic loops

The 11 s / 79 s figures are the **worst case**: a cold, cache-defeated whole
context. A multi-turn agentic loop grows its prompt by a small delta each turn,
so with prefix caching (plus a KV connector) only the delta is prefilled and
per-turn TTFT collapses to sub-second. The absolute numbers shrink; the
contention with long generations remains.

## Reproduce

The [strict readout smoke baseline](readout-baseline.md) records a later
correctness-focused run: removing fabricated missing scores exposed incomplete
label coverage and the reference engine's 20-logprob limit. Earlier successful
request counts do not establish complete score coverage under the strict contract.

The current tree ships only the Rust sidecar; run its tests with
`cargo test --release --manifest-path sidecar-rs/Cargo.toml` (or `make test`).
The measurements above came from a now-removed Python harness; its historical
commands were:

```sh
# concurrency + mixed chat/decision throughput
python benchmarks/bench_parallel.py --each 4 --json parallel.json

# long conversation vs decisions
python benchmarks/bench_mixed.py \
  --context-tokens 95000 --chat-max-tokens 1024 --decisions 100 --json mixed.json
```

Both took `--selftest` and ran with no GPU (they then simply reported failures,
or skipped the live phase), pointed at a `--decide-url` / `--chat-url`. The
scripts were removed in the Rust-only port; no current benchmark script is
equivalent.
# Readout follow-up

See [the implementation and live-test one-pager](readout-one-pager.md): exact-token
scoring answered all ten isolated synthetic cases, but simultaneous ordinary chat
exposed an upstream HTTP 500. The new scoring mode remains opt-in.
# Approach comparison

The [four-arm live experiment](approach-comparison.md) compares exact label
scores, constrained JSON, bounded analysis, and the existing echo baseline.
# Decision Index validation

The [Decision Index run](decision-index-validation.md) scores the readout against
the official suite. Both full-sample runs completed once the engine's
`--gpu-memory-utilization` was lowered; the original container is restored. The
100-row sample is too small for a non-null `decision_index`.
