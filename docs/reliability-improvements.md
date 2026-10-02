# Reliability improvements — PL / IP

**Revision:** plan-3, 2026-10-02. **Scope:** reliability repairs and Decision Index evaluation using the user's existing jev-index dataset. Q-PQNLD-5 explicitly preserves JSON-valued instructions and criterion descriptions, aligning the shared validator and MCP schema with the existing renderer and official corpus. Required fields, question types, criteria objects/cardinality, and fail-closed scoring remain enforced. All builds/tests/processes/data stay on Apollo. Q-PQNLD-4 authorizes one fresh 100-row sample on the existing Apollo model after remote tests pass, capped at fifteen minutes with a temporary loopback sidecar and fresh outputs. No engine restart/configuration change, deployment, or publishing is authorized.

## Human basis

Original seed: `/home/erik/Documents/erik/homelab/pqnld-one-pager.md`, SHA256 `4790883dbdbb788307c6e0f673337b94e1f1bb01b330bafbfd9b783fca225d02`; a verbatim copy is preserved in the homelab task audit. Request: evaluate via `/pl /ip /rv` and propose improvements. User selected reliability and decision quality, then said “lets do improvements”. For evaluation data: “we have jev-index dataset that was used for benches infirst version. Lets use it”.

Reviewed source baseline: `6651289f53d5e4cc9b9e13cfd71825c57822faa1` (`v0.3.0`). This plan does not rewrite the original seed or turn sample completion into a quality score.

## PL: supported outcome and decision map

The existing typed-decision capability is worth repairing before extending adapters: successful responses must come from complete, usable model evidence, and malformed requests must not crash the service. The [behavior contract](behaviors/reliability.feature) operationalizes the seed's existing fail-closed claims; it does not introduce calibration, global determinism, or support for another provider.

Source confirms missing/empty choice criteria can reach `keys[best]` (`src/main.rs:645–690`), echo parsing can erase missing scores (`src/engine.rs:764–781`), and an empty continuation sums to zero (`src/main.rs:542–560`), yielding valid-looking uniform probabilities. Temperature startup only tests `<= 0` (`src/main.rs:1076–1085`).

**Destination:** eliminate these failure modes without changing valid response envelopes. **Actors:** clients get explicit refusal instead of invented confidence; operators keep a functioning service; evaluators distinguish data quality and coverage from decision correctness. **Owner:** pqnld. **Dependencies:** existing Rust tests and engine fixture seams; actual jev-index data and official scoring for quality evaluation.

**Known limits retained:** the exact-ID path is opt-in; 255 is a wire ceiling, not guaranteed tokenizer capacity; historical sample parity is neither accuracy nor calibration. Q-PQNLD-3 subsequently accepted global sidecar admission, implemented in R4; ordinary direct engine chat remains outside that budget. General adapter expansion and an engine-side mixed-MTP repair remain out of this slice.

### Bounded party / grill

**Product:** “A completed sample is not a useful decision. Use the original labeled benchmark.” **Tester:** “First stop missing scores masquerading as 50/50.” **Architect:** “One validator in the shared path, not guards in every interface.” **Security:** “Typed output is not permission to act; keep loopback/gateway requirements.” **Pragmatist:** “Reuse Rust tests and the old benchmark, not a new evaluation framework.”

Round two resolves sequencing: reliability regressions first; locate jev-index independently; run only offline scoring of existing outputs until an inference target is explicitly authorized. Unresolved choices—global worker admission semantics, strict resource budgets, 255-label expansion, and provider support—must not be silently decided by this repair.

## IP: vertical delivery and verification

One useful green change, not a speculative PR stack. Preserve assertion-failing RED logs before each corresponding implementation change. Per the user's environment correction, all Rust build/test/format commands run on Apollo in a dedicated snapshot build workspace with its existing toolchain; this machine only edits source and reads receipts. Benchmark processes and dataset remain on Apollo too.

| Slice | Observable result / RED seam | Minimum GREEN | Verification / marker |
|---|---|---|---|
| R1 | Malformed/empty/under- or over-sized choices refuse before engine access; regression through shared readout and applicable transports | One shared input validator called before probing/scoring; retain existing supported error mapping | `make test` exits zero, then `printf 'R1_OK\n'` |
| R2 | Missing/misaligned/non-finite echo evidence and wrong batch cardinality refuse, while legitimate first unscored prompt token remains supported | Fallible evidence parsing plus continuation/cardinality checks at shared echo seam | `make test` exits zero, then `printf 'R2_OK\n'` |
| R3 | Non-finite/non-positive temperatures refuse; valid finite temperature still works | Finite-positive startup validation | `make test` exits zero, then `printf 'R3_OK\n'` |
| Q1 | Distinguish benchmark correctness, coverage, refusals, and calibration from parity | Locate and reuse jev-index's official labels/scorer and saved outputs; no dataset or results invented | Exact owner recipe to be recorded after discovery; explicit handoff if labels/outputs/runner are missing |

The markers are printed only after the test command succeeds, using `if make test; then printf 'R1_OK\n'; else exit 1; fi` (substitute the relevant marker). Current tests are expected to pass at baseline; new tests must fail by assertion, not compilation or infrastructure errors. Bind each feature scenario to named Rust tests in the implementation receipt. Run full tests and formatting after each green slice and after final review.

**Rollout:** review local green diff, publish only on request, and deploy only after separate target authorization. **Rollback:** restore the previous released artifact through the owner deployment recipe; no host hand-edits. During development, undo only this task's experiment, never reset unrelated work. Stop after three repairs of the same failure.

## Quality gate

Preflight found 11 original sample requests / 166 questions with structured JSON
instructions or criterion descriptions. The initial string-only R1 assumption,
derived from the narrower MCP schema, would reject legitimate official inputs.
PL was reopened; Q-PQNLD-5 chose preserving supplied JSON through the existing
renderer. A new remote assertion RED and minimal validator/schema correction were
observed before final GREEN and inference. Earlier string-only test expectations
are explicitly superseded by this human decision; the corpus is not transformed,
and no scoring evidence or acceptance is silently rewritten.

Decision Index (called jev-index by the user) discovery must record dataset version, official labels/scoring, split selection, saved output provenance, refusal/coverage denominators, and whether scores are actually available for calibration. Record component licenses; keep restricted rows private, honor evaluation-only restrictions, and exclude benchmark evaluation rows from training. Compare a declared simple baseline using the same examples; do not fit or select on the reported evaluation split. If live inference is needed, ask for target and bounded budget. A missing corpus or scorer is a blocker, not permission to fabricate a replacement.

## Current evidence / status

R1–R4 and the Q-PQNLD-5 JSON amendment are implemented; final Apollo debug and release each passed 46 unit tests plus one CLI integration test. [Implementation receipts](reliability-evidence/receipt.md) retain genuine RED/GREEN, pre-correction local evidence, remote runner, mutation control and feature-to-test bindings. Official saved-output scoring and the single authorized repaired-binary sample are complete in [the quality report](decision-index-quality.md): 100/100 requests, no errors/refusals, all 760 answers matching historical v7. No measured accuracy improvement is claimed. Independent review and bounded conformance re-review found no blocking production defect after numerical/concurrency coverage and wire/worker/schema reconciliation. Formatter/Clippy are unavailable in the existing Apollo toolchain; no components installed. The homelab full gate has fourteen inherited broken links plus remote execution prerequisites; no unrelated repair is included.
