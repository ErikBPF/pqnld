# FAQ

**Does pqnld need a GPU or a model to run?**
No. It is a proxy over an endpoint. The unit tests and benchmark self-tests run
with no GPU and no model at all.

**Which models does it work with?**
Anything served behind an OpenAI-compatible endpoint that exposes
`logprobs`/`top_logprobs` on `/v1/chat/completions`. Add a descriptor for your
model's chat template if you need the echo fallback.

**Is pqnld a replacement for my LLM?**
No. It is a secondary decision capability that runs on the model you already
serve. Keep your existing LLM for chat and generation; pqnld adds typed,
closed-set decisions on top of the same endpoint, so you get more out of the
infrastructure you already run without replacing it.

**Why is a decision ~0.12 s but 100 concurrent decisions take tens of seconds?**
Those are different questions. A warm single decision is one prefill plus one
step. Under load, every request queues behind the batch and the engine admits
tiny one-token requests in a ramp (~1/1/2/4 per step), so latency grows in
~0.12 s steps. That is an engine scheduling property, not pqnld overhead — a
one-question decision through pqnld matches a direct engine call.

**Why is my throughput so low for long prompts?**
Because prefill dominates. A ~100k-token prompt takes ~79 s to prefill on the
reference rig, so total-wall TPS looks tiny. pqnld reports nothing about
throughput; `bench_mixed.py` separates TTFT from decode TPS for exactly this
reason.

**Do decisions and chat really share one engine?**
Yes. A decision is a `max_tokens=1` + `logprobs` chat request, so vLLM's
continuous batching and chunked prefill serve both. Only split engines if the
speculative-decode / KV-connector overhead on a short decision prompt outweighs
the simplicity of one engine — measure before splitting.

**Should I lower speculative decoding for faster decisions?**
Careful. Speculative decoding (e.g. MTP) speeds up long completions and does
almost nothing for one-token decisions, while shrinking the per-step token
budget. Lowering it helps decisions and hurts completions. If you want both, the
clean option is two profiles (lean for decisions, full for chat); otherwise keep
speculation and accept that concurrent decisions admit slower. When chat is the
primary workload, keep MTP on — pqnld treats it as mandatory; a lean profile is
a decision-only window. See
[benchmarks.md](benchmarks.md).

**A decision came back slow even though nothing else was running.**
The first call after a restart is cold (no prefix cache). Warm repeats are an
order of magnitude faster and are also served from pqnld's own LRU when the
request is byte-identical.

**What about questions with more than 26 options?**
They are relabelled with a tokenizer-verified single-token alphabet (`a-z` for
the first 26, then more ASCII/Greek/Cyrillic symbols), and the engine is asked
for those exact token IDs. Because the engine caps explicit IDs at 128 per
request, a prompt with more options is split across `ceil(n/128)` requests and
the logprobs are merged losslessly. The echo fallback remains only for models
without exact-token scoring.

**Does it support the `score` question type?**
No. pqnld supports `choice` (2–255 options) and `noul` (a yes-probability). An
unsupported type is refused with `422`, never approximated.

**Can it hallucinate?**
Not in the option sense: it only ever scores the options you supplied, so an
answer outside your criteria is impossible. It can still be *wrong* or
miscalibrated — that is what the probability distribution is for.

**Is it accurate / calibrated?**
Calibration is the point: use the distribution, not just the argmax. Report
calibration error and wrong-at-high-confidence on your own data; do not assume
the reference numbers apply to your model or prompts.

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
