# Global workers — accepted planning extension

**Stage / revision:** PL + IP + GREEN + RV / workers-3, 2026-10-02. **Status:** R4 implemented with Apollo assertion RED/GREEN; final coverage and conformance review clear; complete debug/release suites pass. Basis: reliability priority and Q-PQNLD-3, answered “Global bound (Recommended)”. This extends R1–R3 in [the reliability plan](reliability-improvements.md); it does not supersede their acceptance.

## Outcome and boundary

`--workers N` should bound active sidecar scoring across client requests, including startup/readout scoring probes. Preserve stored question order at `N=1`; on a failed concurrent decision, cancel and drain its remaining tasks instead of detaching them. Ordinary chat sent directly to the engine is outside this bound. Never turn this scheduling guarantee into engine determinism or mixed-MTP readiness.

The [contract](behaviors/global-workers.feature) is approved and operationalized by named Rust tests in [the implementation receipt](reliability-evidence/receipt.md). Existing semaphore and Tokio task management replace no scheduler/service. Deployment is not authorized; Q-PQNLD-4 separately authorizes one bounded fresh sample after remote tests, recorded in [the reliability plan](reliability-improvements.md).

## Grounding and grill

Baseline `6651289`: `sidecar-rs/src/main.rs:784–809` skips the shared semaphore when workers=1 and returns early on a task error; HTTP independently spawns connections at `:999–1005`. Readout probing can occur before question scheduling, so moving a permit only into a question task is insufficient. Permit ownership must avoid recursive acquisition/deadlocks and cover all actual sidecar scoring entry points, not merely the named client path. Cancellation cannot unsend an already accepted upstream HTTP request; it can stop/drop local futures and reclaim permits.

The reliability boundary is explicit: admission control reduces sidecar concurrency, not direct model traffic. Keep per-request sequential ordering rather than replacing it with unordered spawning. Compare task-drain behavior on error and panic; do not allow cleanup to mask the original refusal. Avoid an unbounded queue rewrite: this change bounds active work, not inbound request/body memory.

## Delivery slice R4

**Owner:** pqnld. **Dependency:** R1–R3 green so code ownership remains serial. **RED:** existing test engine fixture holds scoring requests open while two real readout clients run; assert maximum active sidecar scoring ≤N, include first auto-mode probe, verify stored question order at N=1. A failing multi-question fixture proves outstanding local tasks terminate and permits become available after refusal.

**Minimum GREEN:** enforce the existing global budget consistently, reuse task cancellation/draining, preserve response format and question order. Do not change temperature, score parsing, labels, or cache policy in this slice.

**Verification:** run `make test` from a dedicated pqnld source-snapshot workspace **on Apollo only**. No local Rust builds/tests/formatting after the user's environment correction. `if make test; then printf 'R4_OK\n'; else exit 1; fi` prints marker only after all assertions pass. Preserve actual remote RED and GREEN receipts. No new framework or dependency. Documentation must describe the revised global bound only after observed tests pass.

**Rollout / rollback:** local reviewed diff only; publishing and deployment require separate authorization. Revert only this task's scheduler patch if verification fails, without discarding R1–R3 or unrelated work. Three repairs of one failure is the limit. Final RV covers correctness, deadlocks, cancellation, throughput implications, compatibility, and simplicity.

## Decisions and continuing work

Q-PQNLD-3 is answered; no further scheduling-policy approval is needed. R1–R4 and
JSON compatibility are green in both Apollo debug/release profiles; corpus
verification, two official offline scores and the one authorized fresh sample
are complete. The user requests final review, docs updates and local task merges
to `main`; pushing, deployment and further inference remain unauthorized.
External chat/MTP and generic resource limits stay outside the accepted scope.
Local cancellation cannot unsend upstream work already accepted.
