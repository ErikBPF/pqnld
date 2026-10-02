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

```
state: "The meeting is on Tuesday at 3pm in room B."
question: which day?  {monday, tuesday, friday}
->  {"choice": "tuesday", "probabilities": {"monday": 0.02, "tuesday": 0.95, "friday": 0.03}}
```

The lettered readout uses a one-token request per question on the engine you already serve, so
chat completions and decisions share one vLLM process and its continuous
batching — no second model to host.

**Serving status:** exact-token scoring is experimental. The tested vLLM MTP
path drops requested scores during concurrent chat; the proposed engine fix is
not yet GPU-validated. MTP remains required for the primary completions workload.
See [benchmark evidence and repair status](docs/mixed-readout-diagnosis.md).

> The name is the point. **P**arallel **Q**uery **N**ode, **L**ogit **D**ecisions
> — and, if you tilt your head, *¿por qué no los dos?* Why not both: chat and
> decisions, on one engine. 🎲

---

## What it is / what it isn't

**It is:**

- A readout. Returned option scores are normalized over your options; they are
  not automatically calibrated correctness probabilities.
- A wrapper. No weights, no training, no fork of your engine.
- One engine. Chat and decisions share vLLM's continuous batch, because a decision
  is just a tiny chat request.

**It isn't:**

- A replacement for your chat or generation model. It does not write prose or
  answer open-ended questions; it answers the typed decision you frame.
- A reasoner. It does not chain thoughts, call tools, or emit prose. One pass
  decides — that is the whole trick, and the whole limit.
- A `score` type, or free-form. `choice` (2–255) and `noul` only; anything else is
  refused, never approximated.
- Magic calibration. It reports the model's belief faithfully. Whether that belief
  is *right* is your data's problem, not the wrapper's.
- A scheduler. Under load, your engine's admission ramp sets latency, not pqnld.

---

## Why

- **Closed-set outputs by construction.** The model never emits free text; pqnld reads
  the next-token distribution over the options you provided and softmaxes it.
  An answer outside your option set is impossible; an incorrect choice is not.
- **No new weights.** It is a wrapper over an endpoint you already run. If you
  serve Qwen, Llama, Mistral, or anything else that exposes `logprobs`, you
  already have the model for typed decisions; whether its decisions are accurate
  is your calibration problem, not a new model's.
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

Start it against a running vLLM (or any) OpenAI server:

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
2. It sends **one chat request** (`max_tokens=1`, `logprobs`, `top_logprobs ≥
   option count`) and reads the distribution over those letters at the answer
   slot.
3. It maps single-letter tokens back to option keys, ignores words and other
   non-letter tokens, requires every label to be present, and softmaxes with a
   per-model temperature. Missing label scores are refused with HTTP 422.

Because a label is a single token, options are independent and multi-token
option keys cost nothing extra. Questions with **more than 26 options** use an
extended, tokenizer-verified single-token alphabet; when the engine's
per-request explicit-token cap (128) is exceeded the shared prompt is split
across `ceil(n/128)` requests and the answer-slot scores are merged losslessly.
The older chunked **echo** readout remains only for models without
explicit-token scoring.

At startup pqnld **probes the model**: with `readout: auto` it keeps the lettered
path only when a single option letter really is the top token for a fixture, and
otherwise forces the echo readout. A descriptor that asks for echo without a
chat template is refused.

Questions in one request are read **sequentially by default** so repeated runs
are reproducible; a worker bound allows opt-in concurrency, and identical
repeated decisions are served from a bounded LRU cache keyed by model +
descriptor + readout + request.

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

- **`--max-num-seqs`**: the number of sequences batched per step. At `2` tiny
  `max_tokens=1` decisions admit slowly under load; `8` more than doubled
  mixed-load throughput in our measurements.
- **`--enable-prefix-caching`**: makes agentic multi-turn loops cheap — only the
  new delta is prefilled.
- **Speculative decoding (`--speculative-config`, e.g. MTP)**: great for long
  completions, near-useless for one-token decisions, and it shrinks the
  per-step token budget. A lean profile (no spec decode, no KV connector) is
  what pqnld's `engine_profile: lean` selects in the reference launcher. Use it
  only for a decision-only window: when chat is the primary workload, keep MTP —
  this project treats MTP as mandatory, not something to disable permanently.

See [docs/benchmarks.md](docs/benchmarks.md) for the measured trade-offs and the
raw benchmark receipts from a 2× RTX 5060 Ti rig.

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
