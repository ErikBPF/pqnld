# FAQ

**Does pqnld need a GPU or a model to run?**
No. It is a proxy over an endpoint. The unit tests run with no GPU and no model
at all.

**Which models does it work with?**
The vLLM adapter is live-tested. Other adapters require backend/model-specific
verification; an OpenAI-compatible API with logprobs alone does not establish
compatibility. Add a descriptor with the model's chat template for echo scoring.

**Is pqnld a replacement for my LLM?**
Keep the model. Add a job. pqnld reads typed decisions from its logits while your
existing chat and generation workload stays on the same endpoint.

**Why does latency grow under load?**
Requests wait for the sidecar worker budget and engine admission. Prompt length,
shared chat load, batching and cache state all affect latency; an unloaded
single-question measurement is not a concurrent-load prediction.

**Why is my throughput so low for long prompts?**
Long prompts spend more time in prefill. Separate time to first token from decode
throughput, and distinguish cached from uncached requests when measuring.

**Do decisions and chat really share one engine?**
Yes. Same engine, different job. Lettered decisions request one token plus
logprobs from your existing endpoint. Experimental exact-ID scoring has a
[shared-chat/MTP validation gate](readout-one-pager.md#operating-reference).

**Should I lower speculative decoding for faster decisions?**
Keep MTP. Chat is the primary workload; a one-token decision is no reason to
hobble completions. pqnld leaves engine flags and speculative decoding policy alone.

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
It cannot invent an option: answers stay inside your supplied criteria. Wrong
choice? Still possible. Closed-set output is a contract, not clairvoyance.

**Is it accurate / calibrated?**
Accuracy depends on model, prompt, readout and backend together. Evaluate
calibration and wrong-at-high-confidence on held-out labeled data before using
confidence thresholds.

**Does it need the thinking block disabled?**
For thinking models you generally want `enable_thinking: false` so the answer
slot is the option and not the start of a reasoning trace. It is a descriptor
field; the shipped Qwen3.8 descriptor disables it.

**How do I expose it to a router like LiteLLM?**
Register `http://<host>:11560/v1` as an OpenAI provider with the base model equal
to the `--model` name you served. The `/v1/chat/completions` shim accepts a
decision-request JSON user message and answers it as a decision.

**Why `pqnld`?**
Parallel Query Node, Logit Decisions.
