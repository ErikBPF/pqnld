# Label-bias measurement and llama.cpp validation results

**Revision:** results-1, 2026-10-13. **Status:** measured on two live engines;
one engine row (Strata) blocked. No push, release or deployment.

Human seed: the LocalLLaMA thread on pqnld asked whether the *letter* bound to an
option moves that option's score, and proposed re-running one decision with the
letters reassigned. Coordination records:
`homelab docs/proposals/2026-10-13-pqnld-label-bias-engine-validation.md` and
`homelab docs/plans/2026-10-13-pqnld-label-bias-llamacpp-strata.md`.
Accepted behavior: [label-bias.feature](behaviors/label-bias.feature).

## What was built

| Artifact | Purpose |
|---|---|
| `sidecar-rs/src/labels.rs` | Deterministic seeded cyclic label binding; `seed % k` shift, so `k` consecutive seeds put every label at every position exactly once |
| `--label-permute-seed N` (CLI/env) and a per-question `label_seed` field | Select the binding; default (`None`) renders byte-identical to before |
| `sidecar-rs/src/bin/bias_probe.rs` | Two-factor measurement (`label` and `position`), repeats, capability probe, self-test, cross-run summariser |
| `sidecar-rs/models/qwen38-27b-nvfp4-labels.json` | vLLM descriptor with `specific_token_scores: true` |
| `sidecar-rs/models/qwen4b-v8-trial.json` | llama.cpp trial descriptor with `top_k_floor: 4096` |
| `sidecar-rs/tests/fixtures/bias-fixture.jsonl` | 12 rows / 13 questions, SHA256 `df4d8b48391f0e1c11d53f3b8bc43b9b880bf8c9cce82be5591f0729f22b9b50` |

Design: factor **B** holds option order and shifts the label binding (`seed = s`,
`s` in `0..k`); factor **A** rotates option order and sets `seed = r` for rotation
`r`, which keeps each option's label fixed while its position moves. The two
factors are therefore separately estimable. Metrics: per-option probability
spread (p50/p95/max), top-1 flip rate, inversion rate against each fixture's
expected answer, per-label mean offset `δ_L`, per-position mean offset `δ_P`.
Pre-registered rules (fixed before measurement): rejected if `spread_p95 > 0.02`
and flips `> 2%`; supported if `spread_p95 ≤ 0.01`, flips `≤ 0.5%` and zero
inversions; otherwise unresolved; noise-limited when the floor exceeds `0.01`.
A label-specific effect requires `δ_L` to exceed `δ_P` on the same question.

Runnable checks: `cargo test` (53 unit tests; the seeded-binding and top-k-floor
tests failed on their assertions before the implementation, receipt in the
homelab audit) and `bias_probe --self-test` → `PQNLD_BIAS_CORE_OK` (planted bias
detected and named, null supported, a 0.5 floor suppresses the verdict).

## Arm 1 — vLLM, Apollo, exact-ID readout

Engine: live `qwen38-27b-nvfp4` on Apollo (`vllm-openai:v0.30.0`, MTP2 shared
serving), read over the OpenAI-compatible path with
`specific_token_scores: true`. Three sequential runs, 13 questions, 226 requests
per run, **0 refusals** (complete label coverage including a 25- and a 26-option
question), repeats identical, `PQNLD_ARM_ASSERTIONS_OK` each run.

| question | k | spread p95 | flips | inversions | max δ_L | max δ_P | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| s2-0002:q1 | 3 | 0.5785 | 0.333 | 0.333 | 0.1911 | 0.1417 | rejected |
| labels-k4:q1 | 4 | 0.2026 | 0.000 | 0.000 | 0.0527 | 0.0557 | unresolved |
| s2-0003:q1 | 3 | 0.1085 | 0.000 | 0.000 | 0.0133 | 0.0168 | unresolved |
| labels-k13:q1 | 13 | 0.0993 | 0.000 | 0.000 | 0.0186 | 0.0095 | unresolved |
| labels-k7:q1 | 7 | 0.0972 | 0.000 | 0.000 | 0.0534 | 0.0085 | unresolved |
| labels-k2:q1 | 2 | 0.0722 | 0.000 | 0.000 | 0.0217 | 0.0361 | unresolved |
| s2-0004:q2 | 3 | 0.0372 | 0.000 | 0.000 | 0.0124 | 0.0021 | unresolved |
| labels-k26:q1 | 26 | 0.0281 | 0.000 | 0.000 | 0.0032 | 0.0051 | unresolved |
| s2-0004:q1 | 3 | 0.0184 | 0.000 | 0.000 | 0.0014 | 0.0061 | unresolved |
| s2-0006:q1 | 3 | 0.0135 | 0.000 | 0.000 | 0.0055 | 0.0021 | unresolved |
| s2-0007:q1 | 3 | 0.0066 | 0.000 | 0.000 | 0.0022 | 0.0023 | supported |
| s2-0001:q1 | 3 | 0.0041 | 0.000 | 0.000 | 0.0010 | 0.0013 | supported |
| s2-0005:q1 | 25 | 0.0028 | 0.000 | 0.000 | 0.0003 | 0.0002 | supported |

Verdict counts: **supported 3, unresolved 9, rejected 1**. Cross-run floor 0
(three runs inside one sidecar/engine process, so this is a lower bound, not the
restart-separated floor; restarting the production vLLM was out of scope).

Reading:

- **No decision moved for a label-specific reason.** The single flipping
  question (`s2-0002`, "which option is the fruit?" with `apple`/`banana`/`chair`)
  flips at exactly the same rate under both factors (0.333 label, 0.333
  position) — the fixture is ambiguous (banana is a fruit), so the flip is model
  ambiguity, not a letter preference. No other question changed its top choice
  under either factor.
- **Label-conditional score movement is real but small.** `δ_L` exceeds `δ_P` by
  more than 0.01 on 3 of 13 questions (max difference 0.049, on `s2-0002`), while
  `δ_P` dominates on 1. So letters do move scores by up to a few probability
  points, without changing an answer in these fixtures.
- The strict pre-registered support band (`≤ 0.01`) is met on 3 of 13 questions,
  which is why the count is dominated by "unresolved" rather than "supported".
  Reporting only the aggregate would have called the whole arm "rejected" on the
  strength of one ambiguous fixture; the per-question table is the honest form.
- Wide questions are the most label-stable here (`k = 25` spread 0.0028), so
  "more letters" did not mean more bias on this model.

## Arm 2 — llama.cpp, Orion trial, native top-k readout

Engine: pinned `ghcr.io/ggml-org/llama.cpp@sha256:33b2abe…` (the same image
digest as Orion production), trial container on Orion, loopback `:8082`, CPU-only
(`-ngl 0`, 16 threads, `-c 4096`, one slot), model
`qwen4b-v8.q5_k_m.gguf` SHA256 `75a9d6391f9f572e9bd33d372f02304e8856b3292436e96327870f482d70b9bb`
(a private `ha-harness` fine-tune, reused from `/projects` rather than downloaded).
Production `llama-chat` on `:8080` was never touched, stopped or reconfigured;
the trial ran CPU-only for a measured ~3.1 GiB of free GPU memory at ~13.2 GiB
used.

Three runs, in this order:

1. **Default read width (`max(options, 20)`) — refused.** 235 refused samples of
   235: `422 answer-slot logprobs missing option labels: b, c`. Fail-closed
   behavior was already correct; the read was simply too narrow.
2. **`top_k_floor: 512` — partially covered.** k ≤ 7 questions complete
   (0 refusals); the 13-option question returned 1 of 28 samples and the 25- and
   26-option questions refused completely (135 refusals of 235). The measured
   requirement above 512 is a property of *this* model's answer-slot ranking.
3. **`top_k_floor: 4096` — complete.** 13 questions, 0 refusals, repeats
   identical, `PQNLD_ARM_ASSERTIONS_OK`, wall time 61 s for 226 requests.

Engine-side measurements against the trial (and read-only probes of production
`llama-chat`): `/tokenize` returns one id for `b` (65); `/completion` accepts
`n_probs` of 8, 20, 64, 256 and returns exactly that many `top_logprobs` entries
(no hard cap — the previously unresolved "`n_probs` behaviour at very large k");
`/props`, `/health` present; the capability probe reports
`native-top-k-surface-present`. Label tokens can carry a leading space (`" a"`),
which the existing `letter_of` already trims.

**Arm 2's bias numbers are not usable as an H1 result**: this 4B tool-calling
fine-tune does not perform the letter-labelled task (inversion rate 0.98, flip
rate 0.96), so its permutations measure model incoherence, not label bias. The
arm's value is the adapter verdict below.

## Verdicts

- **H2 (llama.cpp adapter): live-verified for the native top-k surface**, with
  three qualifications: (a) exact-ID scoring is unavailable and refused honestly;
  (b) >26 options route to the echo path and are refused, as documented;
  (c) label coverage is model-dependent, so a `top_k_floor` is a prerequisite for
  models whose answer slot does not rank option letters — on this trial model the
  threshold lay between 512 and 4096 for questions with 13+ options.
- **H1 (label invariance): unresolved by the pre-registered band on vLLM, with no
  label-specific decision change.** Scores move by up to ~0.05 under label
  reassignment; no answer changed for a label-specific reason. A model that
  performs the task would be needed to extend this to llama.cpp, and a
  restart-separated floor to tighten it.
- **H1 extended (2026-10-14, Arms 3/5/6): rejected on every llama.cpp arm,
  including 27B on GPU.** 3B, 9B and 27B are all label-driven (13/13, 9/13 and 8/13
  questions rejected; `δ_L` up to 0.61 against `δ_P` ≤ 0.06), while the same 27B
  family on vLLM is not. The dominant factor is the engine/readout path, not model
  size; the 27B path control was inconclusive and a chat/system-prompt variant of
  the llama.cpp lettered path is the next decisive test.
- **H3 (Strata parity): gap, measured live.** Strata's server at the pinned commit
  serves no `logprobs`, no `/tokenize` and no `/completion`, so a pqnld readout
  cannot run against it; the earlier "no binary on Orion" blocker is superseded by
  this capability result (Arm 4).

## Limits

- One model per engine, 13 questions, one prompt format; results are not a
  universal claim about letters.
- `δ_L` and `δ_P` are first-order mean offsets; interaction terms are not
  reported.
- The cross-run floor is zero only because all vLLM runs shared one engine
  process; restart-separated reproducibility remains unmeasured (the same
  qualification the earlier evaluation recorded).
- The llama.cpp arm ran CPU-only with a private 4B model; a GPU arm with a
  task-competent model is the natural next step.
- Scores remain normalized over the supplied options and are not calibrated
  correctness probabilities.

## Arm 3 — llama.cpp, Apollo CUDA build, task-competent small model

Engine: `localhost/apollo-llamacpp:b10853-cuda13` (llama.cpp **b10853** CUDA build,
different from the b10398 image used on Orion), trial container on Apollo, loopback
`:8090`, CPU-only (`-ngl 0`, 24 threads, `-c 4096`), model
`Qwen2.5-3B-Instruct-Q4_K_M` SHA256
`29716154290ca491e8b4e3ca18a5c8b4ef99abf7c5ddec140a423aa6b92ab36f`.
Descriptor `models/qwen25-3b-trial.json`. Production vLLM on the same host was not
restarted, reconfigured or stopped; its GPU memory was untouched (CPU-only trial).

Read width, same ladder as Orion: default `max(options, 20)` completed questions up
to 7 options and refused 143 of 235 samples at 13/25/26 options; `top_k_floor: 2048`
completed all 13 questions with 0 refusals and identical repeats
(`PQNLD_ARM_ASSERTIONS_OK`, 92 s for 226 requests). Three runs, bit-identical.

Per-question summary (three runs, floor 0 within one engine process):

| question | k | spread p95 | flips (label) | flips (position) | inversions | max δ_L | max δ_P | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| s2-0001:q1 | 3 | 0.9855 | 0.667 | 0.000 | 0.333 | 0.5524 | 0.0001 | rejected |
| s2-0002:q1 | 3 | 0.9509 | 0.667 | 0.000 | 0.333 | 0.5465 | 0.0000 | rejected |
| s2-0003:q1 | 3 | 0.9215 | 0.000 | 0.333 | 0.833 | 0.0001 | 0.3072 | rejected |
| s2-0004:q1 | 3 | 0.8638 | 0.000 | 0.333 | 0.167 | 0.0001 | 0.2879 | rejected |
| labels-k7:q1 | 7 | 0.8044 | 0.143 | 0.143 | 0.143 | 0.1137 | 0.1148 | rejected |
| s2-0007:q1 | 3 | 0.7803 | 0.333 | 0.000 | 0.167 | 0.2601 | 0.1223 | rejected |
| s2-0006:q1 | 3 | 0.7305 | 0.667 | 0.000 | 0.333 | 0.4668 | 0.0039 | rejected |
| labels-k13:q1 | 13 | 0.6417 | 0.154 | 0.000 | 0.077 | 0.0744 | 0.0011 | rejected |
| labels-k26:q1 | 26 | 0.9901 | 0.808 | 0.038 | 0.423 | 0.3389 | 0.0370 | rejected |
| s2-0004:q2 | 3 | 0.1827 | 0.000 | 0.000 | 0.000 | 0.0023 | 0.0609 | unresolved |
| s2-0005:q1 | 25 | 0.1421 | 0.000 | 0.000 | 0.000 | 0.0201 | 0.0067 | unresolved |
| labels-k2:q1 | 2 | 0.1188 | 0.000 | 0.000 | 0.000 | 0.0371 | 0.0594 | unresolved |
| labels-k4:q1 | 4 | 0.0555 | 0.000 | 0.000 | 0.000 | 0.0179 | 0.0000 | unresolved |

Verdict counts: **rejected 9, unresolved 4, supported 0**. Mean label flip rate
0.264 against a mean position flip rate 0.065; `δ_L` reaches 0.55 while `δ_P` stays
at or below 0.31.

Reading, and the difference from Arm 1:

- **On this engine and model the answer follows the letter, not the option.** Moving
  a label to a different option moves the decision (label flips 7 of 13 questions;
  position flips 4), while `δ_L` (up to 0.55) dwarfs `δ_P`.
- A matched control (same question, three label bindings, expected option `red`)
  makes the mechanism explicit and rules out the raw-prompt format as the cause:
  with labels `b) red c) blue a) green` the answer is `a` — the option that now
  carries the model's preferred letter — on **both** the native `/completion` path
  and the chat path with `letter_system_prompt`, at logprob ~-0.1 versus -2.4 for
  the correct letter. The prepared llama.cpp adapter sends only the raw prompt, so
  the two paths differ in principle; here they agree, so the missing system prompt
  is **not** the cause of the flip.
- Where the model is competent and the option set is wide or the answer obvious
  (`s2-0005`, `labels-k2`/`k4`), the readout is stable and label-insensitive
  (`δ_L` 0.02-0.04), matching Arm 1. The bias therefore appears **where the model
  cannot answer from content** — the letter prior decides.
- This is the thread's hypothesis confirmed on a small model, and it is the
  opposite of Arm 1's 27B result on the same fixtures. The practical implication
  for a decision engine: single-token letter readout needs a model large enough to
  answer from content, or a label-binding randomisation (now available) so that a
  letter prior shows up as disagreement instead of a silently wrong decision.

## Arm 4 — Strata on Apollo: capability gap, not parity

Strata's own server was run on Apollo from the pinned source
(`Niko1221/Strata@fb58e0d`, Python `serve.server` with the scripted mock engine so
that no AMD-only binary is required) and surveyed live. The engine boundary in that
server is `Engine.generate(prompt_ids, ...) -> iterator of token ids`: it never asks
an engine for logits, so the served routes are the whole surface any backend can
offer.

| Route | Result |
|---|---|
| `GET /health`, `/v1/models`, `/props`, `/slots` | 200 |
| `POST /v1/chat/completions` | 200, and **no `logprobs` in the response even when `logprobs: true` is requested** |
| `POST /tokenize` | **404** |
| `POST /completion` | **404** |

`bias_probe --capability-probe` against it prints `PQNLD_CAPABILITY_GAP`. The
lettered readout needs either exact-token-id scoring or a top-k logprob read, and
Strata serves neither; the echo path needs prompt logprobs, which it also does not
serve. **Strata cannot serve a pqnld decision readout at all** without a new
engine-side surface, so no parity measurement exists to make. No adapter was
invented and no engine patch was written.

## Arm 5 — llama.cpp, Apollo GPU, 9B (Qwen3.5-9B-Q4_K_M)

Full offload (`-ngl 99`) on the freed Apollo cards, production stopped for the
window and restored after. Two runs, 13 questions, **0 refusals**, bit-identical:
`spread_p95 0.673`, `flip_rate 0.769`, `inversions 0.625`, **rejected 13 of 13**.
Label flips exceed position flips on 10 of 13 questions and `δ_L` reaches 0.36
against `δ_P` ≤ 0.22. GPU rather than CPU changes nothing about the verdict and
improves coverage: the same fixture took ~3 minutes per run on CPU and ~20 seconds
on GPU.

## Arm 6 — llama.cpp, Apollo GPU, 27B (Qwen3.8-27B-UD-Q4_K_M)

Same shape, 16.46 GB across both cards, two runs, 0 refusals, bit-identical:
`spread_p95 0.758`, `flip_rate 0.750`, `inversions 0.375`, **rejected 8 of 13**.

| question | k | p95 | flips (label) | flips (position) | inversions | max δ_L | max δ_P | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| labels-k4:q1 | 4 | 0.899 | 0.750 | 0.000 | 0.375 | 0.609 | 0.061 | rejected |
| s2-0002:q1 | 3 | 0.758 | 0.667 | 0.000 | 0.333 | 0.496 | 0.025 | rejected |
| labels-k7:q1 | 7 | 0.733 | 0.571 | 0.143 | 0.357 | 0.370 | 0.045 | rejected |
| labels-k13:q1 | 13 | 0.693 | 0.154 | 0.231 | 0.077 | 0.117 | 0.032 | rejected |
| labels-k26:q1 | 26 | 0.674 | 0.038 | 0.500 | 0.269 | 0.057 | 0.025 | rejected |
| s2-0006:q1 | 3 | 0.597 | 0.333 | 0.000 | 0.167 | 0.285 | 0.004 | rejected |
| s2-0003:q1 | 3 | 0.560 | 0.333 | 0.000 | 0.167 | 0.259 | 0.018 | rejected |
| s2-0004:q1 | 3 | 0.553 | 0.000 | 0.333 | 0.167 | 0.021 | 0.185 | rejected |
| s2-0001:q1 | 3 | 0.448 | 0.000 | 0.000 | 0.000 | 0.163 | 0.006 | unresolved |
| labels-k2:q1 | 2 | 0.412 | 0.000 | 0.000 | 0.000 | 0.206 | 0.146 | unresolved |
| s2-0004:q2 | 3 | 0.397 | 0.000 | 0.000 | 0.000 | 0.064 | 0.136 | unresolved |
| s2-0005:q1 | 25 | 0.307 | 0.000 | 0.000 | 0.000 | 0.029 | 0.014 | unresolved |
| s2-0007:q1 | 3 | 0.111 | 0.000 | 0.000 | 0.000 | 0.038 | 0.036 | unresolved |

A CPU run of the same model (single run, interrupted by the GPU stage, 34
refusals) gives the same shape: p95 0.780, flips 0.750, inversions 0.375.

## Arm 10 — read-width ladder (no descriptor tuning needed)

The generic top-k path now escalates instead of refusing: it starts at
`max(options, 20)` (or the descriptor's `top_k_floor`) and retries at 512, 2048 and
4096 while option labels are missing, logging each rung, and refuses only if the
widest rung still cannot see every label. Validated on the same 27B trial with
descriptors that declare **no** `top_k_floor`:

| surface | refusals | escalations | accuracy | label flips | p95 |
|---|---:|---:|---:|---:|---:|
| native `/completion` raw prompt | 0 | 214 | 0.84 | 0.22 | 0.758 |
| chat `/v1/chat/completions` (default) | 0 | 104 | 0.99 | 0.03 | 0.084 |

Both match the fixed-2048 arms exactly, so the ladder changes coverage, not
results, and `top_k_floor` is now an override for the starting rung rather than a
requirement. A first attempt at this ladder was wrong in a way worth recording: its
loop compared the current rung against itself, so it never escalated while its
refusal message claimed four widths had been tried. A direct engine probe (same
prompt, widths 512/2048/2048/4096/4096) showed the server returning every label
each time, which located the fault in the client, not the engine.

## Cross-engine comparison, and the corrected reading

| Engine / readout | Model | p95 | label flips | position flips | verdicts |
|---|---|---:|---:|---:|---|
| vLLM, exact IDs + chat/system prompt | 27B NVFP4 | 0.203 | 0.264 | 0.065 | supported 3 / unresolved 9 / rejected 1 |
| llama.cpp, native top-k, raw prompt | 27B UD-Q4_K_M | 0.758 | 0.750 | 0.060 | rejected 8 / unresolved 5 |
| llama.cpp, native top-k, raw prompt | 9B Q4_K_M | 0.673 | 0.769 | 0.050 | rejected 13 / unresolved 0 |
| llama.cpp, native top-k, raw prompt | 3B Q4_K_M | 0.701 | 0.885 | 0.065 | rejected 9 / unresolved 4 |

**The earlier conclusion needs correcting.** After the 3B result this file read the
bias as conditional on model competence. With the 27B measured on the same engine
family as the 3B and the 9B, the pattern is better explained by the **engine and
readout path**: every llama.cpp arm is label-driven regardless of size, and the one
arm that is not (vLLM) differs in three ways at once — exact-token-ID scoring
instead of a top-k read, a chat request with `letter_system_prompt` instead of the
raw completion prompt, and NVFP4 rather than a GGUF quant. Model competence shows
up in the inversion rate, not in whether the letter decides.

A control for the 27B (three label bindings of one question whose answer is `red`)
did **not** resolve which of the three causes dominates: with a narrow top-k the
letters are below the top tokens on that model, so both the raw and the chat
request answer a newline. The decisive experiment is to give the llama.cpp lettered
path a chat/system-prompt variant and re-run this fixture; that is now a tracked slice in the homelab coordination repo's official-support
plan (`docs/proposals/2026-10-14-pqnld-llamacpp-official-support.md`) rather than
an assumption. Until then, the honest statement is: on llama.cpp, the letter decides.

## Arm 7 — per-configuration sweep (size and quant), Apollo GPUs

One arm pair per model/quant configuration, llama.cpp `b10853` on Apollo with full
offload (`-ngl 99`), same 13-question fixture and the same read width
(`top_k_floor: 2048`) for every row, **two runs per configuration, bit-identical,
zero refusals everywhere**. The point is to read size and quant effects off
separately, and to keep two questions apart: how far the model's answers diverge
from the fixture's expected answer (`accuracy = 1 - inversions`), and how much the
model's own answer moves under label and order permutation (the variance that the
readout introduces).

| configuration | accuracy vs expected | label flips | order flips | p95 spread | max δ_L | max δ_P | supported / unresolved / rejected |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-3B Q4_K_M | 0.83 | 0.18 | 0.07 | 0.99 | 0.47 | 0.30 | 1 / 4 / 8 |
| Qwen3.5-9B Q4_K_M | 0.66 | 0.46 | 0.12 | 0.67 | 0.38 | 0.21 | 0 / 0 / 13 |
| Qwen3.5-9B Q8_0 | 0.66 | 0.49 | 0.14 | 0.67 | 0.35 | 0.21 | 0 / 0 / 13 |
| Qwen2.5-14B Q4_K_M | 0.97 | 0.06 | 0.01 | 0.54 | 0.27 | 0.08 | 3 / 8 / 2 |
| Qwen3.8-27B UD-IQ4_XS | 0.82 | 0.24 | 0.11 | 0.70 | 0.48 | 0.20 | 0 / 5 / 8 |
| Qwen3.8-27B UD-Q4_K_M | 0.84 | 0.22 | 0.09 | 0.76 | 0.61 | 0.18 | 0 / 5 / 8 |
| Qwen3.8-27B UD-Q4_K_XL | 0.86 | 0.20 | 0.07 | 0.75 | 0.50 | 0.19 | 0 / 5 / 8 |

Reading, kept to what the numbers support:

- **The letter lever dominates in every configuration.** Label flips exceed order
  flips in all seven rows (0.06-0.49 against 0.01-0.14) and `δ_L` exceeds `δ_P`
  everywhere (0.27-0.61 against 0.08-0.30). No size or quant removes it, and no row
  is label-invariant on this engine.
- **Quant barely matters.** At 9B, Q4_K_M and Q8_0 are within noise of each other
  (accuracy 0.66/0.66, label flips 0.46/0.49, p95 identical). At 27B the three
  quants span 0.82-0.86 accuracy and 0.20-0.24 label flips, i.e. the quant axis is
  a few points wide while the label effect is a factor of 2-10 over the order
  effect. The concern that quant divergence explains the earlier cross-engine gap
  is therefore not supported: quant divergence is *smaller* than the label effect
  in every matched pair measured here.
- **Accuracy is not monotone in size on this fixture, and the gaps are small.**
  14B scores highest (0.97) and 9B lowest (0.66), with 27B between (0.82-0.86).
  Thirteen questions give a granularity of 0.077, so the 14B-to-27B gap is under
  two questions; the ladder is too coarse to claim a size effect on accuracy.
  14B is also the only row from a different family (Qwen2.5 vs Qwen3.5/3.8), so its
  outlier status (least label-sensitive) is confounded with family and prompt
  format rather than attributable to size.
- **Within-run variance is zero** (identical repeats and identical runs), so every
  variance figure above is between-permutation variance — the bias — not
  measurement noise.

Follow-up this generates: the 14B row is the only configuration whose decisions are
mostly stable (6% label flips). Whether that is family, prompt format or its task
competence is not separable with one model per size; a same-family 14B or a second
Qwen2.5 size would settle it.

## Arm 8 — the path question answered: llama.cpp chat surface vs native raw prompt

Same model (Qwen3.8-27B-UD-Q4_K_M), same 13-question fixture, same read width
(`top_k_floor: 2048`), GPU, two identical runs, zero refusals — the only change is
the surface: **llama.cpp's OpenAI-compatible `/v1/chat/completions` with the letter
system prompt** instead of the native `/completion` raw prompt.

| surface (same model, fixture, width) | accuracy | label flips | order flips | p95 | max δ_L | max δ_P | sup/unres/rej |
|---|---:|---:|---:|---:|---:|---:|---|
| native `/completion`, raw prompt | 0.84 | 0.22 | 0.09 | 0.76 | 0.61 | 0.18 | 0 / 5 / 8 |
| chat `/v1/chat/completions` + letter system prompt | **0.99** | **0.03** | 0.00 | **0.08** | 0.16 | 0.06 | 7 / 5 / 1 |

The raw prompt was the cause. Moving the same weights to the chat surface removes
almost all of it: label flips fall 0.22 → 0.03, the p95 spread 0.76 → 0.08, and
accuracy rises 0.84 → 0.99 with the fixture's expected answers unchanged. It also
matches the vLLM arm's shape (which used this surface), so the earlier
"letter wins on llama.cpp" result is a property of the *native raw-completion
readout*, not of llama.cpp, not of quantisation and not of model size.

Reaching the letters on the chat surface needed one fix: the OpenAI adapter
clamped `top_logprobs` to the OpenAI API's 20, and declared that as its ceiling.
It now honours the descriptor's `top_k_floor` (`openai_topk_size`, unit-tested), so
an OpenAI-compatible server that accepts wider reads — llama.cpp maps the field to
`n_probs` — gets read at the declared width.

A residue remains and is not zero: δ_L 0.16 against δ_P 0.06, one rejected question
in thirteen. Label sensitivity is reduced, not eliminated.

## Arm 9 — 50-question fixture from the Decision Index corpus

The kit's `sample-le26` rows (≤26 options, official expected answers) converted to
the probe's fixture format: **50 questions**, k = {2:19, 3:21, 4:4, 6:1, 9:1, 11:4},
SHA256 `5c9eb15bc45c94e293961d23b7b39d844d8058b2df8e45a75fab109c1ae9d5e5`, one run
per configuration after the two-run identity established everywhere else.

| configuration (llama.cpp native, GPU, floor 2048) | accuracy | label flips | order flips | p95 | max δ_L | max δ_P | sup/unres/rej |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3.8-27B UD-Q4_K_M | 0.68 | 0.20 | 0.12 | 0.97 | 0.74 | 0.34 | 0 / 24 / 26 |
| Qwen2.5-14B Q4_K_M | 0.59 | 0.15 | 0.18 | 0.88 | 0.37 | 0.44 | 1 / 23 / 26 |
| Qwen3.5-9B Q4_K_M | 0.47 | 0.46 | 0.16 | 0.61 | 0.32 | 0.24 | 0 / 1 / 49 |

What the finer fixture changes and what it does not:

- **Accuracy resolution.** 50 questions give 0.02 granularity, and accuracy is now
  ordered by size (9B 0.47 < 14B 0.59 < 27B 0.68). Arm 7's reading — 14B ahead of
  27B — does not survive the bigger fixture, which is exactly why the 13-question
  figure was flagged as too coarse.
- **Accuracy is dominated by option count, not by size.** On the 27B arm: k=2 0.87,
  k=3 0.69, k=9 0.94, but k=4 0.25 and k=11 0.07. The fixture is a tool-selection
  corpus whose hard rows are the wide ones; a single accuracy number hides that.
- **Variance survives.** Every configuration still labels-flips at 0.15-0.46 and
  δ_L ≥ δ_P in the 9B and 27B rows (14B is the exception, δ_P 0.44 > δ_L 0.37).
- The arm-level `flip_rate`/`inversions` fields are the *worst* question; the probe
  now also emits `flip_rate_mean`, `inversions_mean` and `accuracy_mean`, and every
  figure in this section is a mean over questions.

## Arm 11 — the sweep re-run under the shipping default (chat surface)

Same seven configurations, the same descriptor files, the same fixtures as Arm 7/9 —
only the surface changed, from the raw `/completion` prompt to the chat surface that
is now the default. One run per configuration (two-run identity was established in
seven configurations already).

13-question fixture:

| configuration | accuracy | label flips | order flips | p95 | max δ_L | max δ_P | sup/unres/rej |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-3B Q4_K_M | 0.81 | 0.16 | 0.07 | 1.00 | 0.48 | 0.14 | 2 / 5 / 6 |
| Qwen3.5-9B Q4_K_M | 0.99 | 0.02 | 0.00 | 0.38 | 0.19 | 0.10 | 3 / 9 / 1 |
| Qwen3.5-9B Q8_0 | 0.99 | 0.03 | 0.00 | 0.35 | 0.23 | 0.06 | 1 / 11 / 1 |
| Qwen2.5-14B Q4_K_M | 0.99 | 0.00 | 0.02 | 0.04 | 0.01 | 0.08 | 10 / 3 / 0 |
| Qwen3.8-27B UD-IQ4_XS | 1.00 | 0.00 | 0.00 | 0.07 | 0.11 | 0.03 | 6 / 7 / 0 |
| Qwen3.8-27B UD-Q4_K_M | 0.99 | 0.03 | 0.00 | 0.08 | 0.16 | 0.06 | 7 / 5 / 1 |
| Qwen3.8-27B UD-Q4_K_XL | 1.00 | 0.00 | 0.00 | 0.12 | 0.11 | 0.06 | 7 / 6 / 0 |

50-question Decision Index fixture (0.02 accuracy granularity):

| configuration | accuracy | label flips | order flips | p95 | max δ_L | max δ_P | sup/unres/rej |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5-9B Q4_K_M | 0.69 | 0.10 | 0.06 | 0.51 | 0.24 | 0.22 | 1 / 31 / 18 |
| Qwen2.5-14B Q4_K_M | 0.65 | 0.07 | 0.11 | 1.00 | 0.32 | 0.49 | 26 / 8 / 16 |
| Qwen3.8-27B UD-Q4_K_M | 0.75 | 0.05 | 0.05 | 0.34 | 0.10 | 0.15 | 15 / 27 / 8 |

What the default changes, stated against Arm 7/9 (same configurations, raw prompt):

- **Label-driven decisions are gone for every configuration at 9B and above**:
  label flips fall from 0.15-0.46 to 0.00-0.03 on the 13-question fixture and to
  0.05-0.10 on the harder 50-question one. The 3B model remains the outlier
  (accuracy 0.81, label flips 0.16, p95 1.00) — below some size, the answer slot is
  not stable enough for a single-token readout.
- **Accuracy becomes the differentiator, and 27B leads**: 0.75 on the 50-question
  corpus against 0.69 (9B) and 0.65 (14B), with 27B also complete on the 13-question
  fixture (1.00 for two of its three quants). 9B and 14B swap places between
  surfaces, and both are within about two questions of each other, so only the 27B
  lead is worth stating.
- **Order effects do not disappear**: 14B on the hard corpus is order-dominated
  (δ_P 0.49 against δ_L 0.32, p95 1.00) while its 13-question behaviour is the
  calmest of the set. Position sensitivity is a separate axis from the letter
  effect and needs its own fixture attention.
- Quantisation still does not matter at 27B (0.99/1.00 accuracy across IQ4_XS,
  Q4_K_M, Q4_K_XL; zero label flips in two of the three).

The earlier tables in this file stay as the record of what the raw prompt did; this
section is what the shipped configuration does.
