# Decision Index quality: pinned sample100

**Outcome / revision:** Q1 / 3, 2026-10-02. The pinned release completed **100/100 sampled requests, zero errors/refusals**, covering 760 questions. Native quality is mixed; valid response count is not correctness. Source, binary and evaluation inputs are pinned in the execution receipt.

## Evidence and limits

Apollo root `/mnt/data/ai/validation/decision-index-2026-10-01` is mounted at `/work` in existing `dibuild`. Verified interpreter `/work/venv/bin/python` is Python 3.12.3; installed `decision-index==0.2.1` is editable from clean kit commit `87d4650b42b377c0291a89c1f1a879f9b31082bf`. Host venv is not executable outside the container. SSH identity and existing pinned host key were verified against owning fleet/config before access.

The suite verifier reported matching content, additions, exclusions and subsets. Selected gzip differs from manifest's compressed hash; official verifier accepts its matching uncompressed content hash. Suite has 119,898 base scoreable requests plus 30,419 added requests: **150,317 eligible requests**, 44 benchmarks, 38 index benchmarks. Physical selected-file rows are a different denominator: 124,971 before edition/exclusion filtering.

Fresh output: `/work/runs/http-repaired-sample100-20261002`. Official `decision_index=0.0`, `raw_index=0.11`, weighted index coverage `0.0031`, `complete=false`: only 100 requests answered/scored, 150,217 pending. Unweighted completion is 0.0665%, distinct from weighted index coverage 0.31%. Coverage-adjusted, chance-corrected index penalizes unanswered requests; **zero index is not zero sample accuracy**. Historical offline scores remain under `http-uds-v7-offline-20261002` and `http-uds-seq1-offline-20261002`.

## Execution

Structured JSON instructions and descriptions are supported. All original 100
requests passed input validation: 746 choice and 14 noul questions. No coercion,
filtering, tuning or training occurred.

[Execution receipt](quality-evidence/fresh-result-20261002.json) pins source archive `866f0c04…9d72`, release `ded05dca…d87a`, and task descriptor `2a9f6df5…4aa5`; full hashes are retained. Immutable archive and running executable hashes matched approved Apollo GREEN artifacts. Task-copy descriptor forced lettered/exact scoring; no echo fallback, model switch, engine/MTP change or restart. New sidecar cache began blank; shared engine cache/load was not reset or isolated.

One pipeline included two stock synthetic warmups outside the scored sample denominator. Official 100-request run: **222.5s**, excluding warmup/scoring. Supervised pipeline: **270.1s**, including imports, warmup, scoring and up to ~5s polling lag; sidecar through cleanup 270.3s, within 900s budget. Successful HTTP request latency on 100 requests: median 321.1ms, p95 14284.2ms, mean 2225.1ms; not per-question or cache-free latency.

Workers=1/private loopback11574 socket ownership verified. Owned sidecar37726/pipeline37755 reaped; port released. Engine health before/after/after-cleanup HTTP200, 59 successful checks, no failures. Direct engine chat remains outside sidecar admission; this is not mixed-MTP readiness proof.

## Native task quality

### What these scores attribute

This benchmark measures **model + prompt formulation + pqnld readout + inference
backend**, not the tool in isolation. Incorrect answers can reflect model knowledge
or reasoning limits, question formulation, or readout/backend behavior. The current
experiment does not isolate those causes: low accuracy is neither proof of a tool
defect nor proof that model capacity is the cause.

Reliability checks establish refusal and response-contract behavior. They do not
guarantee correct decisions, and normalized option probabilities are not calibrated
correctness probabilities. No decision-accuracy gain is demonstrated. To
investigate attribution, compare the
same model and examples using ordinary generation and closed-set readout, keeping
information and scoring comparable. That comparison has not been run.

All 44 benchmarks are represented, usually by two requests. [Fresh native metrics](quality-evidence/fresh-native-metrics-20261002.json) preserve task names, scores and denominators; [historical metrics](quality-evidence/native-metrics-20261002.json) remain unchanged. Native metrics apply to complete supported sampled case groups; request count is not universally a field/group denominator.

| Task | Sample requests | Fresh official native metric |
|---|---:|---|
| BFCL | 3 | Case exact accuracy 1.000 |
| API-Bank | 3 | Accuracy 0.333 |
| ContractNLI | 2 | Macro-F1 0.434 |
| Home appliance | 2 | Case exact accuracy 0; field accuracy 0.917 |
| ACOS | 7 | Per-review F1 0.333; field accuracy 0.988 |
| ToolRet | 3 | nDCG@10 0.673; supplied BM25 baseline 0.552 |
| BRIGHT | 2 | nDCG@10 0.675; equal to supplied BM25 baseline |
| ForecastBench | 2 | Brier 0.0848; lower is better |
| RAGTruth | 2 | Hallucinated-class F1 0.667 |

High field accuracy can hide failed whole-case decisions and class imbalance; ACOS and appliance results illustrate this. ACOS's seven requests span only two stored groups with 402 requested questions; they are not seven independent reviews or 402 independent cases. Overall sample has 91 stored groups by catalog, not 760 independent observations. Requested-question counts are not universally gold-scored field denominators. Ranking nDCG, Brier, macro-F1 and exact accuracy cannot be pooled. Tiny denominators prohibit broad superiority claims. Native iSarcasm summary has no pooled scalar; none invented.

## Comparison limits

[The baseline join](quality-evidence/fresh-comparison-20261002.json) matches all
100 IDs/payloads and 760 answers, with zero probability or native-primary metric
changes. This is observed parity, not causal accuracy improvement or serving
determinism: engine epoch, shared cache and chat load were uncontrolled. Model API
identity remained unchanged, but no weight-content hash was established.
[Earlier run comparisons](quality-evidence/reproducibility-20261002.json) remain
available as evidence, not a repeatability guarantee.

## Commands, privacy and continuing work

[Fresh executed command](quality-evidence/fresh-command-receipt-20261002.txt) records one attempt, exit0, fresh-path guards and owned-PID cleanup. Earlier [offline commands](quality-evidence/commands-20261002.txt) and [receipt](quality-evidence/offline-score-20261002.json) retain their distinct scope. Offline flags/Python `-B` remained set; inference used only the authorized existing engine. No builds, downloads, installs, source edits or engine reconfiguration occurred in this subtask.

Raw records/logs remain Apollo-private: directories0700/files0600. Reports contain aggregate metrics/counts/hashes only, no rows, responses, labels, credentials or full environment. Kit MIT license does not license constituent corpora; no redistribution permission inferred or corpus copied locally.

The sample is complete; the suite is not. No general calibration or full-suite
quality claim is justified by these tiny per-task samples.
