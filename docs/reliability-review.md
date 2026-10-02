# Reliability: independent review and revision

**Stage / revision:** final RV + local integration / review-4, 2026-10-02. **Status:** final source/conformance review clear; Apollo rerun passed 47 tests in each debug/release profile with unchanged artifact hashes. Fresh sample completed 100/100. User-authorized local merges follow scoped doc/package verification. Formatter/Clippy and full homelab gate remain blocked; no all-gates-green claim.

## Outcome and scope

R1–R4 preserve the human seed's existing closed-set capability while repairing malformed-input handling, missing echo evidence, invalid temperatures, and the explicitly accepted global worker budget. Independent reviewers found no production defect in that bounded patch. They did not infer live provider support, calibration, engine determinism, publication, or deployment authority from green unit tests.

Human feedback clarified attribution: poor benchmark accuracy can reflect model
capacity rather than a tool defect. The documented result measures the combined
model/prompt/readout/backend system; no causal component diagnosis was performed.
This final revision makes that boundary explicit without changing code, accepted
behavior, the corpus, or recorded measurements.

Reviewed first-draft source snapshot was `293f51030953c6d0ee9811980f80e062b4b0165c27b6e78b6f93a17cf9304e37`. Its source delta is preserved in [the first-draft production patch](reliability-evidence/first-draft-production.patch); [RED/GREEN receipts](reliability-evidence/receipt.md) retain discovery, infrastructure failures, and successful checks. Earlier local evidence predates the user's Apollo-only correction. All later Rust and benchmark processes run on Apollo.

## Independent perspectives and dispositions

Fresh risk reviewer session `ses_f03018720ffeoLgmdyfxHG7dUm` independently checked security, reliability, tests, compatibility, performance/operations, adversarial behavior, and simplicity. Fresh architect session `ses_f0301871fffeaOgzgOdXDtF7Ny` checked human-seed integrity and contract conformance. Both read source/callers; neither edited code, invoked nested reviews, or ran Rust locally.

| Verified finding | Revision / disposition |
|---|---|
| No validator bypass, semaphore deadlock, or failed-task drain defect found | Keep minimal shared validator, existing semaphore and `JoinSet`; no speculative production rewrite. |
| Failure test waits before watchdog and leaves recovery unbounded | Bound the complete scenario so scheduler regressions fail rather than hang. |
| Concurrent result-order reconstruction lacks direct reverse-completion regression | Add deterministic gated two-worker success case; no sleeps. |
| Split-merge test only checks count/chunk shape | Pin independent per-key probabilities, second-chunk winner and non-unit temperature; use a controlled mutant to prove discrimination, not fabricated bugfix RED. |
| Structured JSON state compatibility needs a transport example | Extend valid transport fixture without changing its acceptance. |
| README/architecture still promise per-request scheduling and byte-identical results | Correct global sidecar budget and engine/cancellation exclusions; remove unconditional repeatability promise. |
| HTTP previously tolerated omitted instructions/arbitrary descriptions | Initial string-only correction conflicted with actual corpus. Reopened PL: user Q-PQNLD-5 preserves JSON values and aligns MCP schema; required fields and cardinality remain. Four remote assertion failures then GREEN verify the amendment; no corpus transformation. |
| Stage plans still show unimplemented R4 | Link observed implementation receipt and accepted later benchmark authorization. |

Editorial structure/prose review and the earlier two-round grill are recorded in homelab's task audit. Their factual corrections include actual Decision Index denominator/index, latency-percentage arithmetic, absolute difference versus ULP, exact-ID opt-in, tokenizer capacity, and unverified provider status.

## Evidence and limits

First reviewed Apollo gate passed 41 unit tests and one CLI integration test;
the latter runs 29 startup cases. After coverage and accepted JSON compatibility
revisions, final debug and release each passed **46 unit + 1 integration tests**.
Final source snapshot is `866f0c04f189c7039b07585d06c337d33e9cc296fab2fa05bb4fafe844bf9d72`;
the pinned release SHA256 is `ded05dca02e73849c8f9a88da981e717bb20211086796c2477e53d12caa6d87a`.
Formatter and Clippy components are absent on Apollo, and formatting already
failed at the untouched baseline; no component installation or blanket formatting
is part of this delivery.

Final bounded independent conformance re-review, session
`ses_f0301871fffeaOgzgOdXDtF7Ny`, found no blocking or critical finding. It verified
Q-PQNLD-5's four genuine RED assertions, unchanged renderer, universal-JSON MCP
schema, preserved required fields/cardinality, whole-case cancellation watchdog,
reverse-completion order, and the 151-key numerical/chunk mutation control. The
control is not claimed as bugfix RED. Earlier string-only constraints were
explicitly superseded by human approval, not changed secretly to improve scores.

Official historical dataset verification and exactly two offline scores passed.
The single Q-PQNLD-4 fresh pipeline then completed 100/100 requests and all 760
answers without errors/refusals against the pinned repaired release. Supervised
pipeline time was 270.1 seconds, within the fifteen-minute budget; two stock
warmups are excluded from the scored 100-request denominator. Owned processes
were reaped, private port released, and 59 engine health checks returned 200.
Existing engine/MTP configuration remained unchanged.

[Fresh quality evidence](decision-index-quality.md) shows all answers matching
historical v7, not an accuracy gain. API-Bank accuracy was 0.333 on three requests;
Appliance case accuracy was zero on two. The index remains incomplete at 0.0
with weighted coverage 0.31%, not zero sampled accuracy. Shared engine/cache/load
was uncontrolled; tiny task samples do not establish calibration or superiority.

## Decisions, continuing work, and recovery

### Final landing review

Fresh production reviewer `ses_f028c8942ffe1S5a9xxWosLuFE` found no introduced
blocker against R1–R4/Q5. Fresh conformance/editorial reviewer
`ses_f028c3bc1ffeLMpCHMycDzvowI` requested five verified doc reconciliations:
DR-14 now requires a worker budget above one and remains explicitly unautomated;
README no longer blames accuracy solely on data; architecture/descriptor docs
qualify exact-ID opt-in and echo fallback; probe coverage distinguishes exact
from generic mode; current handovers record completed evaluation and local-merge
authorization. No production code, corpus or scoring acceptance changed.

The root log exclusion would have omitted RED/GREEN anchors from Git. A narrow
`/docs/reliability-evidence/*.log` exception makes those sanitized receipts durable;
privacy review found no credentials or restricted corpus/answer exports in the
reviewed logs. Build artifacts, other logs and secret-file exclusions stay intact.
Staged verification identified 64 whitespace diagnostics in historical logs and
the frozen patch, none in source or Markdown. A narrow `.gitattributes` exception
preserves those artifact bytes; source, scripts, contracts and docs retain normal
Git whitespace checks. The final control must show source whitespace still fails.
Final Apollo debug/release checks and scoped doc/package verification are the
landing evidence; neither clears unavailable formatter/lint or inherited homelab
failures. Original seed and first-draft evidence remain preserved.

[Final Apollo merge verification](reliability-evidence/apollo-merge-validation.txt)
records Rust/Cargo 1.99.0, zero exits for both owner test commands, 46 unit plus
one integration test in each profile, and matching source/debug/release hashes.
The final independent reconciliation found all five doc findings fixed, with no
new material contradiction. No further production changes or inference were needed.

No accepted behavior is weakened. The user now authorizes final review, docs
updates, commits and local merges of both task branches to `main`; no push,
deployment or additional inference is authorized. Final docs and test checks run
on Apollo. The prior scoped docs checks passed, but full homelab validation remains
blocked by fourteen inherited broken links and remote execution prerequisites.
Unrelated dirty work in canonical homelab stays outside the task commits and merges.
Raw benchmark rows/answers stay on Apollo; source and summary receipts are the
delivery surface.

Worker admission does not limit inbound body memory, direct engine chat, or upstream work already received when a local future is cancelled. UDS path safety, usage accounting, provider scaffolds, and mixed-MTP readiness remain follow-ups. Recovery preserves original benchmark outputs and uses only the task-owned source-snapshot workspace; no unrelated source or running-host configuration is overwritten. A later rollback should revert only task commits through the owner workflow, never reset unrelated `main` work.
