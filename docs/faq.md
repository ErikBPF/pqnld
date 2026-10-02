# FAQ

**Does pqnld need a GPU or a model to run?**
No. It is a proxy over an endpoint. The unit tests run with no GPU and no model
at all.

**Which models does it work with?**
The vLLM adapter is live-tested. Other adapters require backend/model-specific
verification; an OpenAI-compatible API with logprobs alone does not establish
compatibility. Add a descriptor with the model's chat template for echo scoring.

**Is pqnld a replacement for my LLM?**
No. It is a secondary decision capability that runs on the model you already
serve. Keep your existing LLM for chat and generation; pqnld adds typed,
closed-set decisions on top of the same endpoint, so you get more out of the
infrastructure you already run without replacing it.

**Why does latency grow under load?**
Requests wait for the sidecar worker budget and engine admission. Prompt length,
shared chat load, batching and cache state all affect latency; an unloaded
single-question measurement is not a concurrent-load prediction.

**Why is my throughput so low for long prompts?**
Long prompts spend more time in prefill. Separate time to first token from decode
throughput, and distinguish cached from uncached requests when measuring.

**Do decisions and chat really share one engine?**
Yes. Lettered decisions use `max_tokens=1` plus logprobs on the existing endpoint.
Shared-chat/MTP production readiness still requires mixed-load validation; sidecar
worker limits do not govern chat sent directly to the engine.

**Should I lower speculative decoding for faster decisions?**
Keep MTP enabled for the primary chat workload. Optimizing a one-token decision
must not silently sacrifice completion performance; pqnld does not change engine
flags or speculative decoding policy.

**A decision came back slow even though nothing else was running.**
The first call can include label-token discovery and an uncached prefill.
Identical repeated decisions may hit pqnld's LRU cache; engine cache state and
latency remain workload-dependent.

**What about questions with more than 26 options?**
Set `specific_token_scores: true` with a compatible adapter to use an extended,
tokenizer-verified alphabet. Enough distinct single-token labels must exist.
Above the tested 128-ID cap, the same prompt is scored in chunks; merging requires
comparable absolute full-vocabulary scores. Without this opt-in, questions larger
than the configured alphabet use echo, subject to its capacity and overlap limits.

**Does it support the `score` question type?**
No. pqnld supports `choice` (2–255 options) and `noul` (a yes-probability). An
unsupported type is refused with `422`, never approximated.

**Can it hallucinate?**
Not in the option sense: it only ever scores the options you supplied, so an
answer outside your criteria is impossible. It can still be *wrong* or
miscalibrated. A normalized distribution does not establish correctness.

**Is it accurate / calibrated?**
Accuracy depends on model, prompt, readout and backend together. Evaluate
calibration and wrong-at-high-confidence on held-out labeled data before using
confidence thresholds. See [the measured sample](decision-index-quality.md).

**Does it need the thinking block disabled?**
For thinking models you generally want `enable_thinking: false` so the answer
slot is the option and not the start of a reasoning trace. It is a descriptor
field; the shipped Qwen3.8 descriptor disables it.

**How do I expose it to a router like LiteLLM?**
Register `http://<host>:11560/v1` as an OpenAI provider with the base model equal
to the `--model` name you served. The `/v1/chat/completions` shim accepts a
Decision Index JSON user message and answers it as a decision.

**Why `pqnld`?**
Parallel Query Node, Logit Decisions.
