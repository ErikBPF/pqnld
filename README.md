# pqnld — Parallel Query Node, Logit Decisions

Use your existing LLM for fast completions and typed decisions when needed.
pqnld reads model logprobs into a probability distribution over your options,
without hosting a second model.

pqnld is a **secondary decision capability for an LLM you already serve**, not a
replacement for it. Keep your existing model for chat and generation and get
typed, closed-set decisions out of the same endpoint, so you get more out of the
infrastructure you already run without replacing it.

Turn a compatible **vLLM / OpenAI-style endpoint** into a **typed decision engine** by
reading the model's own answer-slot logprobs. No extra weights, no training, no
fine-tuning.

Give it a `state` and a set of typed questions (`choice` over 2–255 options,
`noul` for a yes-probability) and it returns one answer per question **with a
full probability distribution** over exactly the options you supplied — no free
text and no options outside the supplied set. Calibration must be evaluated on
your own labeled data.

The 255-option limit is the wire ceiling, not guaranteed serving capacity. Exact
scoring also requires enough distinct single-token labels: 206 were measured on
the reference tokenizer, from a default candidate alphabet of 208 characters.

```
state: "The meeting is on Tuesday at 3pm in room B."
question: which day?  {monday, tuesday, friday}
->  {"choice": "tuesday", "probabilities": {"monday": 0.02, "tuesday": 0.95, "friday": 0.03}}
```

The lettered readout uses a one-token request per question on the engine you already serve, so
chat completions and decisions share one vLLM process and its continuous
batching — no second model to host.

**Serving status:** exact-token scoring is experimental. The tested vLLM MTP
path drops requested scores during concurrent chat; the proposed scorer passed a
GPU component check, but patched-server mixed-load verification remains pending.
MTP remains required for the primary completions workload.
See [current serving status](docs/readout-one-pager.md).

> The name is the point. **P**arallel **Q**uery **N**ode, **L**ogit **D**ecisions
> — and, if you tilt your head, *¿por qué no los dos?* Why not both: chat and
> decisions, on one engine. 🎲

---

## What it is / what it isn't

**It is:**

- A readout. Returned option scores are normalized over your options; they are
  not automatically calibrated correctness probabilities.
- A wrapper. No additional weights, no training, no fork of your engine.
- One engine. Chat and decisions share vLLM's continuous batch, because a decision
  is just a tiny chat request.

**It isn't:**

- A replacement for your chat or generation model. It does not write prose or
  answer open-ended questions; it answers the typed decision you frame.
- A reasoner. It does not chain thoughts, call tools, or emit prose. One pass
  decides — that is the whole trick, and the whole limit.
- A `score` type, or free-form. `choice` (2–255) and `noul` only; anything else is
  refused, never approximated.
- Magic calibration. Accuracy depends on model capacity, prompt formulation,
  readout and backend; this benchmark does not isolate their contributions.
- An engine-wide scheduler. It bounds its own scoring, not direct engine chat;
  the engine's admission policy and shared load still affect latency.

---

## Why

- **Closed-set outputs by construction.** The model never emits free text; pqnld reads
  the next-token distribution over the options you provided and softmaxes it.
  An answer outside your option set is impossible; an incorrect choice is not.
- **No new weights.** It is a wrapper over a compatible endpoint you already
  run. Logprobs alone do not establish adapter, tokenizer, or model compatibility;
  correctness and calibration still need labeled evaluation.
- **Explicit uncertainty.** The output is a distribution over allowed labels.
  Validate calibration before using it for routing, triage, or confidence thresholds.
- **One engine, two workloads.** A decision is just a `max_tokens=1` +
  `logprobs` chat request, so vLLM's continuous batching and chunked prefill
  serve chat and decisions concurrently.

---

## Quickstart

Grab the static binary from the
[latest release](https://github.com/ErikBPF/pqnld/releases/latest), or build it:

```sh
cargo build --release --manifest-path sidecar-rs/Cargo.toml
```

This quickstart targets the tested vLLM adapter. Other adapters require
backend-specific compatibility verification; hosted OpenAI is not currently a
verified drop-in target.

```sh
./sidecar-rs/target/release/pqnld-rs --vllm-url http://127.0.0.1:8000 \
  --model qwen38-27b-nvfp4 --models-dir sidecar-rs/models
```

Ask a decision:

```sh
curl -sS http://127.0.0.1:11560/v1/decide -H 'content-type: application/json' -d '{
  "state": "The meeting is on Tuesday at 3pm in room B.",
  "questions": {
    "day": {
      "type": "choice",
      "instructions": "Which day is the meeting?",
      "criteria": {"monday": "Monday", "tuesday": "Tuesday", "friday": "Friday"}
    },
    "is_urgent": {"type": "noul", "instructions": "Is the meeting urgent?"}
  }
}'
```

```json
{
  "model": "qwen38-27b-nvfp4",
  "answers": {
    "day": {"type": "choice", "choice": "tuesday",
            "probabilities": {"monday": 0.02, "tuesday": 0.95, "friday": 0.03}},
    "is_urgent": {"type": "noul", "noul": 0.11}
  },
  "usage": {"input_tokens": 99}
}
```

No GPU and no model are needed to exercise the HTTP surface: `cargo test`
drives it against a stub engine.

---

## How it works

1. pqnld renders each question with **single-letter option labels**
   (`a) Tuesday`, `b) Friday`, …).
2. It requests one answer token (`max_tokens=1`, `logprobs`) and reads the
   distribution over those labels at the answer slot. Exact-ID scoring may use
   multiple upstream requests for one question.
3. It maps single-letter tokens back to option keys, ignores words and other
   non-letter tokens, requires every label to be present, and softmaxes with a
   per-model temperature. Missing label scores are refused with HTTP 422.

Because a label is a single token, options are independent and multi-token
option keys do not require scoring their full text on the lettered path. With
`specific_token_scores: true` in the descriptor and a compatible adapter,
questions with **more than 26 options** use an extended, tokenizer-verified
single-token alphabet. Above the explicit-ID cap (128), the same rendered prompt
is sent in `ceil(n/128)` requests and absolute answer-slot logprobs are merged.
Merging requires comparable full-vocabulary scores from every request, not
per-chunk normalized scores. Exact-ID scoring is
opt-in; the shipped descriptor does not enable it. Without it, large questions
may use the chunked **echo** path.

Echo can exhaust prompt-logprob memory on long prompts. Validate backend capacity
before using it for large questions; the 255-option wire ceiling is not a
production-serving guarantee.

At startup pqnld **probes the model**: with `readout: auto` it keeps the lettered
path only when a single option letter really is the top token for a fixture, and
otherwise forces the echo readout. A descriptor that asks for echo without a
chat template is refused.

Questions in one request are read **sequentially by default**. `--workers N`
bounds sidecar scoring globally across clients, including
readout probes; at one worker each request retains stored question order. This
does not govern chat sent directly to the engine. Failed concurrent decisions
cancel and drain their remaining local tasks, but cannot unsend upstream requests.
Raising the worker bound allows opt-in question concurrency, and identical
repeated decisions are served from a bounded LRU cache keyed by model +
descriptor + readout + request. Shared engine load, restarts and cache state
prevent a universal repeatability guarantee.

### Reliability versus decision quality

Valid typed answers and genuine score evidence do not guarantee correct decisions.
Task accuracy depends on the model, prompt, readout and backend together; poor
accuracy alone does not identify a tool defect or establish a model-capacity limit.
The pinned 100-request sample completed without errors, with mixed native task
quality. See [the evaluation](docs/decision-index-quality.md)
for task metrics, tiny-sample limits and unproven calibration.

### The binary

`sidecar-rs/` is the whole implementation: one static binary serving the wire
contract, with no runtime dependency. Build it with
`cargo build --release --manifest-path sidecar-rs/Cargo.toml`; `cargo test`
covers the readout with no GPU (see
[sidecar-rs/README.md](sidecar-rs/README.md)).

See [docs/architecture.md](docs/architecture.md) for the details and the
capacity ceiling.

---

## Wire format

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/decide` | Decision Index request → typed answers |
| `POST` | `/v1/systemone` | Alias kept for stock Decision Index kit clients |
| `POST` | `/v1/chat/completions` | OpenAI chat shim: a Decision Index JSON user message is answered as a decision; anything else returns a notice |
| `GET` | `/healthz` | Readiness |

Full schemas and errors: [docs/wire.md](docs/wire.md).

---

## Model descriptors

Each model can be described by a `models/<name>.json` file (`--descriptor NAME`,
or `--model NAME` as the fallback). A shipped example is
[`sidecar-rs/models/qwen38-27b-nvfp4.json`](sidecar-rs/models/qwen38-27b-nvfp4.json).

```json
{
  "readout": "auto",
  "temperature": 1.0,
  "enable_thinking": false,
  "chat_template": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n thinking\n\n</think>\n\n",
  "engine_profile": "full"
}
```

Fields, resolution, and the auto-probe contract: [docs/descriptor.md](docs/descriptor.md).

---

## Running against a real engine

Server-side flags matter more than pqnld's:

- **`--max-num-seqs`** controls the engine's sequence budget. Tune it for the
  shared chat/decision workload and available memory.
- **`--enable-prefix-caching`**: makes agentic multi-turn loops cheap — only the
  new delta is prefilled.
- **Speculative decoding (`--speculative-config`, e.g. MTP)** benefits long
  completions more than one-token decisions. Chat is the primary workload: keep
  MTP enabled. pqnld does not configure the engine or apply `engine_profile`.

Current validation is in [docs/decision-index-quality.md](docs/decision-index-quality.md).
[Archived rig measurements](docs/benchmarks.md) remain available for reference.

---

## Compatibility

The `/v1/systemone` path matches the wire contract of the reference Decision
Index reproduction kit, so a stock kit HTTP engine can point at pqnld directly.
`examples/run-s2-kit.sh` and `examples/s2-rows.jsonl` show a kit run.

---

## Development

```sh
make test       # unit tests (no GPU, no network)
make build      # debug binary
make release    # static x86_64-unknown-linux-musl binary
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

Apache-2.0. See [LICENSE](LICENSE).

---

Thanks for reading this far. Now go point it at an endpoint you already run. 🎲
