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
text and no options outside the supplied set.

```
state: "The meeting is on Tuesday at 3pm in room B."
question: which day?  {monday, tuesday, friday}
->  {"choice": "tuesday", "probabilities": {"monday": 0.02, "tuesday": 0.95, "friday": 0.03}}
```

The lettered readout uses a one-token request per question on the engine you already serve, so
chat completions and decisions share one vLLM process and its continuous
batching — no second model to host.

> The name is the point. **P**arallel **Q**uery **N**ode, **L**ogit **D**ecisions
> — and, if you tilt your head, *¿por qué no los dos?* Why not both: chat and
> decisions, on one engine. 🎲

---

## What it is

- A readout. Returned option scores are normalized over your options.
- A wrapper. No additional weights, no training, no fork of your engine.
- One engine. Chat and decisions share vLLM's continuous batch, because a decision
  is just a tiny chat request.

---

## Why

- **Closed-set outputs by construction.** pqnld reads the next-token distribution
  over the options you provided and softmaxes it. No invented option names.
- **No new weights.** A wrapper over a compatible endpoint you already run.
- **Explicit uncertainty.** The distribution, not just the winner.
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

This quickstart uses the live-tested vLLM adapter. Other backend support is
documented in [engine capabilities](docs/engine-capabilities.md).

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

Multi-token option keys cost no extra answer tokens on the lettered path: the
model scores labels, and pqnld maps them back to your keys. Large option sets can
use an extended tokenizer-verified alphabet with opt-in exact-ID scoring, or the
echo readout. [Architecture](docs/architecture.md) covers label capacity,
chunk merging and backend requirements.

At startup pqnld **probes the model**: with `readout: auto` it keeps the lettered
path only when a single option letter really is the top token for a fixture, and
otherwise forces the echo readout. A descriptor that asks for echo without a
chat template is refused.

`--workers N` sets the shared sidecar scoring budget; a bounded LRU cache serves
identical repeated decisions. Scheduling and cancellation details live in the
[runtime guide](sidecar-rs/README.md).

### Operating notes

- Scores are normalized model preferences, not guaranteed correctness. Evaluate
  accuracy and calibration on your labeled data.
- Exact-ID scoring is opt-in and experimental. Shared-chat/MTP validation is
  pending; check [serving status](docs/readout-one-pager.md) before enabling it.
- `choice` accepts 2–255 criteria keys; serving capacity depends on the readout,
  tokenizer and backend. See [architecture](docs/architecture.md).

### The binary

`sidecar-rs/` is the whole implementation: one static binary serving the wire
contract, with no runtime dependency. Build it with
`cargo build --release --manifest-path sidecar-rs/Cargo.toml`; `cargo test`
covers the readout with no GPU (see
[sidecar-rs/README.md](sidecar-rs/README.md)).

See [docs/architecture.md](docs/architecture.md) for the readout details.

---

## Wire format

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/decide` | State + typed questions → typed answers |
| `POST` | `/v1/systemone` | Compatibility alias for `/v1/decide` |
| `POST` | `/v1/chat/completions` | OpenAI chat shim: a decision-request JSON user message is answered as a decision; anything else returns a notice |
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
