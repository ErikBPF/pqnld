# Decision Index validation

## Current evaluation

The pinned Rust release completed **100/100 sampled requests, 760 questions,
zero errors or refusals** against `qwen38-27b-nvfp4`. The
[quality report](decision-index-quality.md) records task metrics, denominators,
timing and comparison limits.

The official suite has **150,317 eligible requests**. Current official output is
`decision_index=0.0`, `raw_index=0.11`, weighted coverage **0.31%**, unweighted
completion **0.0665%**, and `complete=false`. The sample is complete; the suite
is not. A coverage-adjusted zero index is not zero sample accuracy.

This evaluates **model + prompt + readout + backend**, not PQNLD in isolation.
Accuracy causes are not isolated; normalized option probabilities are not
calibrated correctness probabilities. Tiny per-task samples establish neither
broad superiority nor a decision-accuracy gain.

The [official kit](https://github.com/apolinario/decision-index) is the grader.
The `multimodalart/jev-decision-index` Space is a static leaderboard, not an
inference endpoint.

## Reproducibility contract

- Scorer: `decision-index==0.2.1`, clean kit commit
  `87d4650b42b377c0291a89c1f1a879f9b31082bf`; suite corpus edition `release-v2.1`.
- [Execution receipt](quality-evidence/fresh-result-20261002.json) pins the source
  archive, running release binary, task descriptor, sample, suite inputs and
  scorer files with full hashes. Model API identity is recorded, but a
  weight-content hash and stable engine-epoch attestation are not established.
- Original structured JSON instructions and descriptions are preserved. All
  100 requests passed input validation; no input transformation, dropped rows,
  tuning or training occurred.
- [Suite verification](quality-evidence/offline-score-20261002.json) confirms
  uncompressed corpus content, additions, exclusions and subsets. The selected
  gzip hash differs from the manifest's compressed hash; matching content is
  accepted by the official verifier. The input is not rewritten to hide that.
- The task-copy descriptor opts into `specific_token_scores: true` and forces
  `readout: lettered`. The run used **no echo, engine/MTP changes or restart**;
  the shipped descriptor remains unchanged.
- [Runtime scheduling](../sidecar-rs/README.md#scheduling) owns worker ordering,
  concurrency and cancellation. The run used one worker and a blank sidecar
  cache; shared engine cache/load and direct chat were not isolated.
- [Answer comparison](quality-evidence/fresh-comparison-20261002.json) matches all
  100 payloads and 760 answers. Observed parity is not a serving-determinism
  guarantee. Repeatability claims need controlled engine epoch, batch/load and
  cache state as well as pinned inputs and binaries.

## Readout and current capacity

Exact-ID scoring is opt-in. The reference tokenizer yielded **206 distinct
single-token labels**; this is not the **255-key wire ceiling**. Above the
**128-ID request cap**, the [readout](../sidecar-rs/src/main.rs) sends chunks of
the **same rendered prompt** and merges absolute scores. Correct merging requires
comparable full-vocabulary logprobs at the same conditioned answer slot, not
independently normalized chunk distributions.

Context and backend memory remain real limits. [Architecture capacity](architecture.md#capacity-ceiling)
and [echo memory requirements](architecture.md#the-echo-fallback) own those
boundaries. This explicit-ID sample does **not** establish shipped fallback
capacity or [shared-chat/MTP readiness](mixed-readout-diagnosis.md).

## Evidence locations and privacy

Apollo root `/mnt/data/ai/validation/decision-index-2026-10-01` is `/work` in
`dibuild`: sample `/work/sample-100.jsonl.gz`, suite `/work/suite`, kit `/work/kit`,
interpreter `/work/venv/bin/python` (Python 3.12.3). Pinned source/binary paths are
in the execution receipt; fresh output is `/work/runs/http-repaired-sample100-20261002`.
The [executed command receipt](quality-evidence/fresh-command-receipt-20261002.txt)
records the completed authorized evaluation, not permission for another run.

Raw rows, responses, labels and logs stay Apollo-private: directories `0700`,
files `0600`. Published evidence contains aggregate metrics, counts and hashes
only. Evaluation is not training; the kit's MIT license grants no constituent
corpus redistribution rights. No dataset is copied locally.
