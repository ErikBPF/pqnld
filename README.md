# pqnld — Parallel Query Node, Logit Decisions

First, the honest disclaimer: this is a **small, opinionated wrapper**, not a
framework. It does one thing — turn the model's own logprobs into a typed
probability distribution over *your* options — and it would rather do that well
than grow features. If you came for a general agent framework, this is not it.
If you want calibrated decisions out of the endpoint you already serve, keep
reading.

Turn any **vLLM / OpenAI-compatible endpoint** into a **typed decision engine** by
reading the model's own answer-slot logprobs. No extra weights, no training, no
fine-tuning.

Give it a `state` and a set of typed questions (`choice` over 2–255 options,
`noul` for a yes-probability) and it returns one answer per question **with a
full probability distribution** over exactly the options you supplied — no free
text, nothing invented, and a calibrated confidence.

```
state: "The meeting is on Tuesday at 3pm in room B."
question: which day?  {monday, tuesday, friday}
->  {"choice": "tuesday", "probabilities": {"monday": 0.02, "tuesday": 0.95, "friday": 0.03}}
```

A decision costs **one prefill** and runs on the engine you already serve, so
chat completions and decisions share one vLLM process and its continuous
batching — no second model to host.

> The name is the point. **P**arallel **Q**uery **N**ode, **L**ogit **D**ecisions
> — and, if you tilt your head, *¿por qué no los dos?* Why not both: chat and
> decisions, on one engine. 🎲

---

## What it is / what it isn't

**It is:**

- A readout. One prefill per question; the distribution is the model's own belief,
  softmaxed over your options.
- A wrapper. No weights, no training, no fork of your engine.
- One engine. Chat and decisions share vLLM's continuous batch, because a decision
  is just a tiny chat request.

**It isn't:**

- A reasoner. It does not chain thoughts, call tools, or emit prose. One pass
  decides — that is the whole trick, and the whole limit.
- A `score` type, or free-form. `choice` (2–255) and `noul` only; anything else is
  refused, never approximated.
- Magic calibration. It reports the model's belief faithfully. Whether that belief
  is *right* is your data's problem, not the wrapper's.
- A scheduler. Under load, your engine's admission ramp sets latency, not pqnld.

---

## Why

- **No hallucination by construction.** The model never emits text; pqnld reads
  the next-token distribution over the options you provided and softmaxes it.
  An answer outside your option set is impossible.
- **No new weights.** It is a wrapper over an endpoint you already run. If you
  serve Qwen, Llama, Mistral, or anything else, you already have a decision
  engine.
- **Honest uncertainty.** The output is a distribution, so you get calibration
  for free: use it for routing, triage, abstention, or confidence thresholds.
- **One engine, two workloads.** A decision is just a `max_tokens=1` +
  `logprobs` chat request, so vLLM's continuous batching and chunked prefill
  serve chat and decisions concurrently.

---

## Quickstart

```sh
pip install pqnld
```

Start it against a running vLLM (or any) OpenAI server:

```sh
pqnld --vllm-url http://127.0.0.1:8000 --model qwen38-27b-nvfp4
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

No GPU and no model are needed to try the HTTP surface: see
`python -m unittest discover -s tests` (a stub vLLM backs the tests).

---

## How it works

1. pqnld renders each question with **single-letter option labels**
   (`a) Tuesday`, `b) Friday`, …).
2. It sends **one chat request** (`max_tokens=1`, `logprobs`, `top_logprobs ≥
   option count`) and reads the distribution over those letters at the answer
   slot.
3. It maps letters back to option keys, ignores non-letter tokens, floors
   letters the model did not rank, and softmaxes with a per-model temperature.

Because a letter is a single token, options are independent and multi-token
option keys cost nothing extra. Questions with **more than 26 options** cannot
be labelled with one letter; those fall back to a chunked **echo** readout
(echo the context, then each `context+key`, and sum the key-token
log-probabilities).

At startup pqnld **probes the model**: with `readout: auto` it keeps the lettered
path only when a single option letter really is the top token for a fixture, and
otherwise forces the echo readout. A descriptor that asks for echo without a
chat template is refused.

Questions in one request are read in **parallel** (bounded by
`MAX_QUESTION_WORKERS`), and identical repeated decisions are served from a
bounded LRU cache keyed by model + descriptor + readout + request.

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
[`src/pqnld/models/qwen38-27b-nvfp4.json`](src/pqnld/models/qwen38-27b-nvfp4.json).

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
  what pqnld's `engine_profile: lean` selects in the reference launcher.

See [docs/benchmarks.md](docs/benchmarks.md) for the measured trade-offs, and
[`build_examples/`](build_examples/README.md) for the whole recipe on a real rig
(2× RTX 5060 Ti) with the raw benchmark receipts.

---

## Container

```sh
docker build -t pqnld:dev .
docker run --rm -p 11560:11560 pqnld:dev \
  --vllm-url http://host.docker.internal:8000 --host 0.0.0.0
```

---

## Tester (Swagger)

`pqnld-tester` serves a small OpenAPI UI for a decision endpoint: `POST /decide`
forwards a Decision Index request, `POST /complete` runs a plain chat completion.
Requires the `tester` extra:

```sh
pip install "pqnld[tester]"
PQNLD_BASE_URL=http://127.0.0.1:11560/v1 pqnld-tester   # http://127.0.0.1:8088/docs
```

---

## Compatibility

The `/v1/systemone` path matches the wire contract of the reference Decision
Index reproduction kit, so a stock kit HTTP engine can point at pqnld directly.
`examples/run-s2-kit.sh` and `examples/s2-rows.jsonl` show a kit run.

---

## Development

```sh
make install      # editable install with dev + tester extras
make test         # unit tests (no GPU)
make selftest     # benchmark self-tests (no GPU)
make test-all
make build        # sdist + wheel
make image
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

Apache-2.0. See [LICENSE](LICENSE).

---

Thanks for reading this far. Now go point it at an endpoint you already run. 🎲
