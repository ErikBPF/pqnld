# Decision approach comparison

## First live experiment: 2026-09-27

Ran the historical Python `benchmarks/compare.py` on Apollo's `qwen38-27b-nvfp4`
via a temporary SSH tunnel. Eight synthetic selection cases: 2/3/10/26 options,
each in both orders. Four arms ran sequentially in seeded shuffled order per case. Result
caching was disabled; engine prefix-cache state was uncontrolled. These are
smoke results, not held-out accuracy or calibration evidence.

| Arm | Answered | Correct | Median latency | Mean NLL | Mean Brier |
|---|---|---|---|---|---|
| Exact label IDs | 8/8 | 8/8 | 194 ms | 0.000557 | 0.000000929 |
| Constrained JSON choice | 8/8 | 8/8 | 472 ms | Not available | Not available |
| Bounded analysis then exact IDs | 8/8 | 8/8 | 1,035 ms | 0.0000713 | 0.0000000103 |
| Existing echo scoring | 8/8 | 6/8 | 1,761 ms | 0.173525 | 0.125081 |

Echo used batch size one to bound prompt-logprob memory; these timings are not
an optimized throughput comparison. Both 26-option echo cases chose `item-1`
instead of `item-13`. Unterminated continuation likelihoods reward the shorter
prefix event; this arm is a diagnostic baseline, not a recommended probability
estimator. A future terminated-continuation arm must verify the model's actual
answer-ending token and score that ending too, rather than length-normalizing
arbitrarily. Current echo scoring refuses prefix-overlapping keys before inference;
length and tokenizer-boundary sensitivity remain operating constraints.

The reasoning arm generated at most 128 analysis tokens with thinking disabled,
then passed that analysis alongside the original state to a separate exact-label
request. It is a two-pass explicit-analysis experiment, not native hidden
reasoning or an RL-trained method. Finish reasons are recorded. Higher confidence
on eight easy examples does not establish better calibration.

Exact-ID shared-chat/MTP acceptance is pending; see
[current serving status](readout-one-pager.md). This experiment does not establish
mixed-load readiness for any arm.

## Historical experiment inputs

The Python comparison harness was removed in the Rust-only port; these were its
historical commands:

```sh
PYTHONPATH=src python3 benchmarks/compare.py --url http://localhost:21542
PYTHONPATH=src python3 benchmarks/compare.py --url http://localhost:21542 --rows private-cases.jsonl
```

Input is one object per line: `id`, `state`, `question`, `expected`. A question
has `type: "choice"`, `instructions`, and a `criteria` mapping. Expected is one
criterion key. This deliberately small format is not the raw Decision Index
row format and did not support noul. Do not commit private case content.

Stdout is JSONL: dataset hash/configuration manifest, per-case answers/errors,
and arm summaries. Failures count against all-case accuracy. Probability metrics
are explicitly conditional on successfully scored cases, with their counts.
JSON choices have no probability metrics. Zero probability on truth records
infinite NLL rather than silently clipping. Preserve stdout with the source
revision, engine image/weight digest, hardware and tokenizer identity for a
reproducible campaign; those identities are not yet captured automatically.

## Official benchmark prerequisite

Pin [Decision Index](https://github.com/apolinario/decision-index) to
`87d4650b42b377c0291a89c1f1a879f9b31082bf`, edition **0.2.1**. Its documented
suite rebuild requires approximately 7 GB downloads, 17 GB working space and
accepted HLE access. The suite is not redistributed by the kit. No gated data
or credentials were acquired in this experiment.

Use the pinned kit's `suite rebuild`, `suite import` (hash verification), and
`suite sample` before running its official engine/scoring interfaces. Do not
feed arbitrary kit rows to this custom choice runner or label its aggregate
accuracy as a Decision Index score. A JSON-only choice arm is not a complete
typed-probability engine for that suite.

The official suite is verified and the pinned Rust sample is complete.
[Current evaluation](decision-index-quality.md) reports its native metrics,
coverage and calibration limits. Evaluation data remains evaluation-only.
