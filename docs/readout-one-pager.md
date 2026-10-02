# PQNLD readout: current status

**Status:** HTTP and MCP typed decisions implemented; exact-ID shared-chat/MTP validation pending.\
**Owner / date:** PQNLD / 2026-10-02 UTC.

## What ships

One resident LLM, two jobs: normal chat and typed decisions. Fast completions
remain primary; **MTP stays enabled**. No second model deployment.

The Rust sidecar exposes HTTP decisions, a chat shim, and native
[MCP stdio](harness-plugins.md). Shared validation preserves arbitrary JSON
instructions and criterion descriptions while requiring supported question
types, instructions, and a `choice` criteria object with **2–255 keys**.
See [wire contract](wire.md) and [implementation](../sidecar-rs/src/main.rs).

## Current readout and scheduling

- Lettered readout turns complete answer-slot label scores into typed answers.
  Missing evidence gets a refusal, not a made-up probability.
- Exact-ID scoring requires descriptor `specific_token_scores: true` and a
  compatible backend. Labels must tokenize to distinct single tokens. The
  reference tokenizer yielded
  **206 usable labels**, below the **255-key wire ceiling**.
- Exact-ID requests use **128-ID chunks** of the same rendered prompt. Merging
  requires absolute, comparable full-vocabulary scores at the same conditioned
  answer slot; this is not a universal backend guarantee.
- Echo scores keyed continuations using the model's chat template. Evidence must
  be aligned and finite; prefix-overlapping keys are refused. Normalization uses
  a finite, strictly positive temperature.
- Scoring and probes share a [worker budget](../sidecar-rs/README.md#scheduling).
  Ordering, concurrency and cancellation have one documented operating contract.
- [Engine adapters](engine-capabilities.md) exist for vLLM, llama.cpp, SGLang and
  OpenAI. vLLM is live-tested; other adapters have compile and parser-unit coverage.

## Latest verification

- [Reliability receipts](reliability-evidence/receipt.md) and
  [final Apollo verification](reliability-evidence/apollo-merge-validation.txt):
  **47 tests in each debug/release profile** (46 unit + one integration).
  Formatter/Clippy components are unavailable; the homelab prerequisite gate is blocked.

## Operating reference

Exact-ID scoring is experimental. Its GPU component check passed; pinned-engine
mixed chat/decision validation and chunked/adaptive/TP2 cases are pending. See
[serving diagnosis](mixed-readout-diagnosis.md) before enabling it on shared MTP.

[Architecture](architecture.md) owns readout capacity and echo memory requirements.
[Runtime guidance](../sidecar-rs/README.md) owns scheduling and UDS path safety.
Usage accounting and inbound/body resource limits are tracked in the
[reliability review record](reliability-review.md).
