# Changelog

Behaviour of each release, stated as it is. The measurements behind these choices
are kept in the operator's private records rather than in this repository.

## 0.4.0

**llama.cpp**

- The lettered readout uses the chat surface (`/v1/chat/completions`) with the
  descriptor's `letter_system_prompt`. `lettered_surface: "native"` selects the raw
  `/completion` prompt instead.
- The generic top-k readout widens itself — 512, 2048, 4096 — while option labels
  are missing, logging each rung, and refuses only if the widest rung still cannot
  see every label. `top_k_floor` sets the starting width rather than being required.
- `--engine-kind auto` detects llama.cpp by probing `/props` before anything that
  could look OpenAI-compatible.

**Engines**

- An OpenAI-compatible surface honours `top_k_floor` instead of clamping
  `top_logprobs` at the API's default of 20.
- `GET /v1/capabilities` reports the engine surface: `tokenize`, `exact_ids`,
  `top_k`, `prompt_logprobs`, the starting read width and the escalation ladder, the
  lettered surface, and an `unsupported` block. The startup line repeats it.

**Decisions**

- A refusal names the option labels that were missing, the read widths tried, and the
  descriptor fields that would help.

**Added**

- `--label-permute-seed` and a per-question `label_seed`: seeded cyclic label
  binding, so a caller can reassign labels without changing option order. Without a
  seed, labels are bound in option order.
- `bias_probe`: two-factor (label and position) measurement over a fixture, a
  planted-bias self-test, an engine capability probe, and a cross-run summariser.
  Emits `flip_rate_mean`, `inversions_mean` and `accuracy_mean` alongside the
  per-question maxima.
- A live llama.cpp CI lane (`.github/workflows/llamacpp-live.yml`): pinned
  `llama-server` tag, hash-pinned small model, readiness gated on a real completion,
  and an end-to-end arm through `--engine-kind auto`.
