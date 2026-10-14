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
- **H1 extended (2026-10-14, Arm 3): rejected on a small model.** llama.cpp on
  Apollo with `Qwen2.5-3B-Instruct-Q4_K_M` shows label-driven answers (9 of 13
  questions rejected, `δ_L` up to 0.55 against `δ_P` ≤ 0.31) where the 27B vLLM arm
  showed none — the letter prior decides when the model cannot answer from
  content, and a matched control rules out the raw-prompt format.
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
