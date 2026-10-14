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
- **H3 (Strata parity): blocked.** No Strata binary or image exists on Orion
  (`localhost/llama-dflash2:bfe65e1` and the pinned llama.cpp image are the only
  related images; the `qwen38-flash-next` IQ3/UD-IQ4 packs were freed). Blocked
  on a rebuild from `Niko1221/Strata@fb58e0d` plus model re-download inside an
  authorized window; no adapter was invented and no Strata binary was built.

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
