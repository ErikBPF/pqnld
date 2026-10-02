# PQNLD readout: current status

**Status:** reliability checks passed; shared chat/decision serving remains gated.\
**Owner / date:** PQNLD / 2026-10-02 UTC.

## Outcome and constraints

One resident LLM serves normal chat and typed decisions. Fast chat completions
are primary; **MTP is mandatory**. A separate model deployment or permanently
disabling MTP is not an accepted solution.

The canonical Rust sidecar exposes HTTP decisions, a chat shim, and native
[MCP stdio](harness-plugins.md). Shared validation preserves arbitrary JSON
instructions and criterion descriptions while requiring supported question
types, instructions, and a `choice` criteria object with **2–255 keys**.
See [wire contract](wire.md) and [implementation](../sidecar-rs/src/main.rs).

## Current readout and scheduling

- Lettered readout requires complete answer-slot label evidence; missing labels
  are refused, not assigned fabricated scores. No parsed-choice fallback exists.
- Exact-ID scoring requires descriptor `specific_token_scores: true` and a
  compatible backend; it is not automatically enabled or shipped by default.
  Labels must tokenize to distinct single tokens. The reference tokenizer yielded
  **206 usable labels**, below the **255-key wire ceiling**.
- Exact-ID requests use **128-ID chunks** of the same rendered prompt. Merging
  requires absolute, comparable full-vocabulary scores at the same conditioned
  answer slot; this is not a universal backend guarantee.
- Echo requires a chat template and aligned, usable finite continuation evidence;
  it refuses prefix-overlapping keys. Large prompts retain engine capacity/memory
  risk. Normalization temperature must be finite and strictly positive.
- `--workers 1` is the global sidecar scoring/probe budget. Questions execute in
  stored order; raising the budget permits concurrency. Concurrent failures cancel
  and drain remaining local tasks. Direct engine chat is outside this budget;
  cancellation cannot unsend upstream requests already accepted.
- [Engine adapters](engine-capabilities.md) exist for vLLM, llama.cpp, SGLang and
  OpenAI. **Only vLLM has live evidence**; other adapters remain compile- and
  parse-unit-tested, not verified live integrations. Scheduling is not an
  engine-wide determinism guarantee.

## Evidence and limits

- [Reliability receipts](reliability-evidence/receipt.md) and
  [final Apollo verification](reliability-evidence/apollo-merge-validation.txt):
  **47 tests in each debug/release profile** (46 unit + one integration).
  Formatter/Clippy and the full homelab gate remain blocked; not all-gates-green.
- [Fresh quality report](decision-index-quality.md): **100/100 sampled requests,
  zero errors/refusals, 760 answers**. Native task quality is mixed; valid response
  count is not correctness. This measures **model + prompt + readout + backend**;
  causes are not isolated, and calibration is not established.
- Official index is **0.0**, raw index **0.11**, weighted coverage **0.31%**,
  `complete=false`. Zero incomplete index is not zero sample accuracy or evidence
  of full-suite quality. No measured decision-accuracy gain is claimed.

## Remaining gates and operating limits

[Mixed-MTP diagnosis](mixed-readout-diagnosis.md) retains the unresolved
explicit-ID serving gate. A GPU numerical component check is not full-server
proof; pinned-engine mixed chat/decision checks and chunked/adaptive/TP2 cases
remain necessary. A sidecar-only lock cannot coordinate direct chat.

[Runtime guidance](../sidecar-rs/README.md) documents UDS path safety. Usage
accounting and inbound/body resource limits remain follow-ups in the
[reliability review](reliability-review.md); worker admission does not bound them.
