# Reliability evidence — R1–R5

Baseline: `6651289f53d5e4cc9b9e13cfd71825c57822faa1`, branch
`fix/reliability-20261002`. Main-thread planning/quality documents are preserved.
No live inference, deployment, publishing, new dependencies, or quality-evaluator
changes. R4 worker changes were added only after explicit Q-PQNLD-3 approval.
Main fetched locked dependencies explicitly after the first infrastructure
blocker; the committed lockfile remains unchanged.

## Fresh pre-code critique

Material refinements after source/tests/security/reliability inspection:

- Validate every question at the shared `readout_call` boundary before cache,
  probe, or scoring; validate direct `score_question` callers too. A malformed
  later question must not allow an earlier valid question to reach the engine.
- Use existing `Unsupported` → HTTP 422 / MCP tool-error 422 mapping for question
  contract violations; preserve transport-envelope 400 / JSON-RPC -32602 errors.
  MCP's schema requires string instructions and string/null criterion values.
  This initial schema-based assumption was explicitly superseded by the user's
  Q-PQNLD-5 “Preserve JSON” decision below; earlier receipts remain historical.
- Preserve initial unscored prompt token as explicit `Option<f64>`, not by
  deleting its token or inventing zero. Reject malformed/alignment-breaking
  evidence as `Other` (500). Permit boundary retokenization: sum scored suffix
  after longest common token prefix, but require nonempty finite continuation.
- Check batch cardinality before consuming any option scores. No new provider
  features; tighten existing echo parsers and shared echo seam only.
- Exercise temperature via actual binary startup in existing Cargo harness,
  descriptor forcing echo + MCP EOF so valid startup needs no engine requests.
- R4 refinement before coding: put admission in probe and question scoring,
  acquire question permits only after mode resolution to avoid recursive
  semaphore deadlock, and serialize initial mode detection. Use `JoinSet` to
  observe failures regardless of stored task order; abort and drain siblings
  without replacing the original refusal. Restore stored answer order afterward.

## Commands and results

**Current outcome: R1–R5 implemented; final Apollo debug and CI release gates
pass 47 tests each (46 unit + 1 CLI integration).** Q-PQNLD-5 explicitly amends
the common validator and MCP schema to preserve supplied JSON. The preceding
RV test-quality follow-up changed tests only; a deliberate mutation-control
failed the new probability oracle and was completely restored. No benchmark
started. Genuine local
R1/continuation RED/GREEN preceded the
Apollo-only restriction. All subsequent parser, cardinality, finite-score,
temperature, global-budget and drain RED/GREEN ran on Apollo. No local Rust,
build, test, formatter, or dependency process ran after that restriction.
Main's planning/quality documents remain untouched by this task.

Commands run from `/home/erik/Documents/erik/pqnld/worktrees/reliability-20261002`.

| Stage / hypothesis | Command / result | Receipt / disposition |
|---|---|---|
| Baseline | `CARGO_NET_OFFLINE=true make test` → exit 2, `make: cargo: No such file or directory` | [baseline.log](baseline.log); installed toolchain inspected |
| Existing toolchain | `PATH=/home/erik/.rustup/toolchains/1.94.1-x86_64-unknown-linux-gnu/bin:$PATH CARGO_NET_OFFLINE=true make test` → exit 2; `sccache` server startup timeout | [baseline-toolchain.log](baseline-toolchain.log); bypass wrapper for task only |
| Offline dependency availability | same command with `RUSTC_WRAPPER=` → exit 2; locked `smallvec 1.16.2` unavailable in local registry index | [baseline-offline.log](baseline-offline.log); stop rather than install or rewrite lockfile |
| First blocked handoff full gate | same command with `RUSTC_WRAPPER=` and `CARGO_TERM_COLOR=never` → exit 2, same dependency-resolution failure | [final-test.log](final-test.log); no tests executed at this point |
| First blocked handoff formatting | same environment, `cargo fmt --manifest-path sidecar-rs/Cargo.toml --check -- --color never` → exit 1, existing formatting differences in all four Rust source files | [final-fmt.log](final-fmt.log); baseline formatting left unchanged |

Toolchain discovery: `/home/erik/.rustup/toolchains/1.94.1-x86_64-unknown-linux-gnu`
exists. A bounded local search (`rtk find /home/erik -maxdepth 7 -type d -name
'smallvec-1.16.2'`) found no alternate extracted cache. This does not establish
that `smallvec` was the only missing dependency; Cargo stopped at first failure.

### Recovery and genuine local RED/GREEN (before restriction)

Main explicitly fetched pinned dependencies, then authorized offline resumption.
All subsequent local test commands used installed Rust 1.94.1, `RUSTC_WRAPPER=`,
`CARGO_NET_OFFLINE=true`, and `CARGO_TERM_COLOR=never`.

| Stage | Command / result | Receipt / keep decision |
|---|---|---|
| Cache recovered, linker broken | `make test` → exit 2; installed Rust's bundled LLD wrapper references a missing Nix path | [baseline-recovered.log](baseline-recovered.log); infrastructure failure, not RED |
| Runnable baseline | add `RUSTFLAGS='-C link-arg=-fuse-ld=bfd'`, run `make test` → exit 0, 22 passed | [baseline-bfd.log](baseline-bfd.log); existing system linker used without installation |
| R1 assertion RED | same environment and `make test` → exit 2, 22 passed / 1 failed; expected engine requests 0, observed 4 for one-choice request | [r1-red.log](r1-red.log); shared prevalidation required |
| R1 GREEN | same command → exit 0, 23 passed | [r1-green.log](r1-green.log); keep shared validator at readout and direct scoring boundaries |
| R1 transport/valid compatibility | same command → exit 0, 26 passed | [r1-transports-green.log](r1-transports-green.log); HTTP/chat/MCP refusals and valid envelopes observed |
| R2 continuation assertion RED | same command → exit 2, 26 passed / 1 failed; empty suffix accepted as `Ok(([-0.0, -0.0], 0))` | [r2-continuation-red.log](r2-continuation-red.log); empty evidence must not normalize |
| R2 continuation GREEN | same command → exit 0, 27 passed | [r2-continuation-green.log](r2-continuation-green.log); keep nonempty context and continuation checks |

After that GREEN, two echo-adapter regression tests were added but not run yet.
The user then prohibited local Rust execution and required all builds/tests/runs
on Apollo. Parser implementation resumed only after actual Apollo assertion
RED. R1 and continuation receipts remain local pre-restriction evidence, not
retroactively relabeled Apollo results.

### Apollo assertion RED/GREEN

Remote commands are reproducible with [apollo-test.sh](apollo-test.sh):
`bash docs/reliability-evidence/apollo-test.sh > RECEIPT.log 2>&1`. The script
streams only task source, validates the task marker, prints deterministic
snapshot SHA256, refreshes extracted source mtimes to prevent stale Cargo
artifacts, then runs the owner's exact `make test` inside `dibuild`.
Every succeeding snapshot is recorded in its test log. First remote snapshot:
[apollo-snapshot-01.log](apollo-snapshot-01.log), used by initial parser RED.

| Tracer | Actual assertion RED | GREEN / disposition |
|---|---|---|
| Echo parser/alignment | [apollo-r2-parser-red.log](apollo-r2-parser-red.log): malformed partial scores accepted; initial null token dropped; 27 passed / 2 failed | [apollo-r2-parser-green-2.log](apollo-r2-parser-green-2.log): 29 passed; explicit `Option<f64>` preserves null and fallible parsers reject corruption |
| Batch/context cardinality | [apollo-r2-cardinality-red.log](apollo-r2-cardinality-red.log): zero batch results accepted as `Ok([])`; 29 passed / 2 failed | [apollo-r2-cardinality-green.log](apollo-r2-cardinality-green.log): 31 passed; verify shared batch and adapter counts before consuming scores |
| Choice index alignment | [apollo-r2-index-red.log](apollo-r2-index-red.log): reversed indices accepted; 31 passed / 1 failed | [apollo-r2-index-green.log](apollo-r2-index-green.log): 32 passed; refuse mismatched/duplicate/malformed/partial indices when supplied |
| Nonfinite/overflow scores | [apollo-r2-finite-red.log](apollo-r2-finite-red.log): NaN context accepted with finite-looking scores; 32 passed / 1 failed | [apollo-r2-finite-green.log](apollo-r2-finite-green.log): 33 passed; reject nonfinite context/suffix and overflowing sum |
| Startup temperature | [apollo-r3-red.log](apollo-r3-red.log): actual CLI accepted NaN; 36 unit passed / 1 integration failed | [apollo-r3-green.log](apollo-r3-green.log): 36 unit + 1 integration passed; finite-positive startup guard |
| Global workers/probes | [apollo-r4-budget-red.log](apollo-r4-budget-red.log): two simultaneous scoring futures with workers=1; 36 passed / 1 failed | [apollo-r4-budget-green.log](apollo-r4-budget-green.log): 37 unit + 1 integration passed; shared semaphore covers scoring and probes |
| Failure cancellation/drain | [apollo-r4-drain-red.log](apollo-r4-drain-red.log): two scoring operations survived refusal; 37 passed / 1 failed | [apollo-r4-drain-green.log](apollo-r4-drain-green.log): 38 unit + 1 integration passed; cancel and drain `JoinSet` on error/panic |
| Additional conformance | no new production behavior; added order/probe overlap, parser omissions, valid MCP, descriptor startup coverage | [apollo-r4-coverage-green.log](apollo-r4-coverage-green.log), then final gate below |

One tooling repair: [apollo-r2-parser-green.log](apollo-r2-parser-green.log) is a
**failed candidate run**, not GREEN. Deterministic tar mtimes were older than
Cargo artifacts, so old tests reran without recompilation. Source was inspected,
runner changed to refresh task source mtimes, and recompiling rerun passed.
No acceptance weakened; no production repair exceeded one attempt per failure.

**First implementation handoff:** [apollo-final-test.log](apollo-final-test.log) recompiles that
source and reports 41 unit + 1 integration passed, no ignored/failed tests.
Snapshot SHA256:
`293f51030953c6d0ee9811980f80e062b4b0165c27b6e78b6f93a17cf9304e37`.

### Independent RV test-quality follow-up

User supplied verified review findings; no blocking production defect was
reported. Changes are coverage of already-correct behavior, **not bugfix RED**:

- `failed_concurrent_readout_cancels_and_drains_remaining_scoring`: move the
  existing two-second watchdog around each entire setup/failure/recovery case,
  including waiting for two entries and the subsequent readout. No unbounded
  async phase remains in those four cases.
- `over_128_options_split_and_merge_losslessly`: own 151 Unicode labels and
  token IDs 5000–5150, assert exact 128/23 ID chunks and identical conditioned
  prompt. Fixture scores are `2*ln((i+1)/151)`, temperature 2; independent
  closed-form expectation is `(i+1)/11476` for every supplied key, tolerance
  `1e-12`, and winner `k150` in the second chunk. Oracle reads neither production
  labels/normalization nor returned scores/probabilities to derive expectations.
- New `workers_two_preserve_answer_order_after_reverse_completion`: wait for
  both operations to enter, release second stored question, observe its finished
  notification, then release first. Assert reversed completion but returned
  answer keys `z,a`, both complete, maximum activity 2 and final activity 0.
  No sleeps; whole case has a two-second watchdog.
- HTTP/chat valid fixture passes nested objects/arrays, numeric, boolean, null
  and Unicode state values; captured engine prompts prove structured state
  forwarding and chat cache reuse. MCP valid fixture passes an array containing
  primitives and nested object. Production handlers are unchanged.

| Follow-up command / control | Actual outcome / receipt |
|---|---|
| Existing Apollo runner, strengthened tests | [apollo-rv-test-quality-green.log](apollo-rv-test-quality-green.log): 42 unit + 1 integration passed; source SHA256 `dbaa9184f0eae7f7cb89716d85c15723a58db3867c88e4d45b7b22847c2e7ab5` |
| Deliberate mutation-control: subtract 4 from only second-chunk fixture scores (first ID 5128), keeping oracle unchanged | [apollo-rv-chunk-mutation-control.log](apollo-rv-chunk-mutation-control.log): 41 passed / 1 assertion failed, `k000` expected `0.0000871383757406762`, got `0.00011505123744933279`; mutant source SHA256 `da7ea82fa17be69e1f5a29a99068205b76cd30566c2f2e388e1521e2d9810578` |
| Remove only the deliberate fixture mutation, rerun exact `make test` | [apollo-rv-final-debug.log](apollo-rv-final-debug.log): 42 unit + 1 integration passed; source hash returns exactly to `dbaa9184f0eae7f7cb89716d85c15723a58db3867c88e4d45b7b22847c2e7ab5` |
| Existing CI release command `cargo test --release --manifest-path sidecar-rs/Cargo.toml`, same Apollo environment plus `CARGO_BUILD_JOBS=2` | [apollo-rv-final-release.log](apollo-rv-final-release.log): 42 unit + 1 integration passed. [Resource check](apollo-rv-release-resources.log) confirmed existing disk/memory capacity; no installs or dependency changes |
| Production source preservation | [before](rv-production-before.log) / [after](rv-production-after.log): full `main.rs`/`engine.rs` and non-test `mcp.rs` prefix hashes identical; comparison passed |

Pre-Q-PQNLD-5 snapshot and copied debug/release binaries are preserved entirely inside
the owned Apollo workspace under
`artifacts/dbaa9184f0eae7f7cb89716d85c15723a58db3867c88e4d45b7b22847c2e7ab5/`.
`source.tar` is the deterministic source snapshot; archive content was compared
against every corresponding remote source file before preservation.
[Artifact receipt](apollo-rv-artifacts.log) and remote `SHA256SUMS` pin:

- Debug binary SHA256:
  `6c0d23ad3368e76006e52bf50eb6870512585c313446d3f47c5cbf6696c6a77a`.
- Release binary SHA256:
  `59833fc4cb5cfe0277e4f4b24a15a1b9b5974567cacfe7fc0767ba9183c22059`.

These artifacts are handoff inputs, **not benchmark/model-quality results**.
Main owns subsequent benchmark authorization/execution and documentation fixes.
No production mutation was needed or retained for that coverage slice; only the test fixture was
deliberately corrupted for one discriminating control run. No repair loop needed.

### Q-PQNLD-5: explicit Preserve JSON behavior amendment

Human approval: **“Preserve JSON”**. Main reported preflight of the original
official sample: 11 requests / 166 questions, including 160 object instructions
and 6 structured criterion descriptions. That corpus preflight was not rerun or
transformed by this task. The initial string-only schema assumption conflicted
with the human seed/corpus; this is an explicit accepted-contract amendment,
not removing refusals to improve a benchmark score.

Minimum implementation: common validator requires an object question and a
**present** `instructions` field, without restricting its JSON value type;
criterion descriptions may be any JSON value. Question type remains supported
string `choice`/`noul`; choice criteria remain a required object with 2–255 keys.
MCP instructions use empty JSON schemas and criterion `additionalProperties`
is `true`; required fields, object shapes, min/max cardinality remain intact.
Existing Python-compatible `text` rendering is reused unchanged. Null criterion
descriptions still print the key, whereas null instructions render `null`.

New bindings:

- `structured_question_json_preserves_shared_readout_and_legacy_rendering`:
  object/array/integer/float/boolean/null/string instructions and descriptions,
  choice+noul envelopes, and independently pinned rendered prompt strings,
  including Python-compatible `1e-05` and null-description key fallback.
- `structured_question_json_succeeds_over_real_http_and_chat`: actual loopback
  HTTP connections traverse the real router/handlers for nested JSON fields;
  exact prompt fragments and chat cache reuse are checked. Whole test bounded
  by a two-second watchdog; no live engine/model request.
- `decide_tool_accepts_present_json_instructions_and_descriptions`: MCP tool
  runs the same seven-value choice+noul matrix successfully.
- `decide_schema_allows_json_values_but_preserves_required_shape`: direct schema
  assertions prove universal JSON value schemas while pinning required fields,
  question type constants, criteria object and 2–255 bounds. No schema library
  or new dependency was added.

Only three earlier refusal fixtures were superseded: present numeric choice
instructions, numeric criterion description, and present null noul instructions.
They are now covered by positive matrices. Added missing noul instructions,
missing type and non-string type refusal cases. Existing missing/empty/nonobject
criteria, 1/256 choices, missing choice instructions, unsupported/nonobject
questions, empty questions and no-engine-on-malformed-input assertions remain.
All score evidence, temperature and global-worker checks remain in the gate.
Earlier string-guard RED/GREEN logs and the pre-Q5 source archive are preserved.

| Q5 command / result | Actual receipt |
|---|---|
| Add accepted-behavior regressions, then existing Apollo `make test` | [apollo-r5-json-red.log](apollo-r5-json-red.log): 42 passed / 4 assertion failures; real HTTP 422 versus expected 200, shared/MCP `question requires string instructions`, schema string versus universal JSON. Source SHA256 `d46de6817fc3c8c948cddaf1cb59b3f447d627a94b819036583f12b5358335a8` |
| Minimal validator/schema amendment plus explicitly superseded refusal fixtures | [apollo-r5-json-green.log](apollo-r5-json-green.log): 46 unit + 1 integration passed on first candidate |
| Final exact `make test` and CI release command using sanctioned Apollo environment | [apollo-r5-final-debug.log](apollo-r5-final-debug.log) and [apollo-r5-final-release.log](apollo-r5-final-release.log): each 46 unit + 1 integration passed, no ignored/failed tests |
| Scope comparison against preserved pre-Q5 source archive | [apollo-r5-production-diff.log](apollo-r5-production-diff.log): PASS; only validator and MCP schema changed in production. Renderer, engine and remaining production code byte-identical |

**Final Q5 source snapshot SHA256:**
`866f0c04f189c7039b07585d06c337d33e9cc296fab2fa05bb4fafe844bf9d72`.
[Artifact receipt](apollo-r5-artifacts.log) pins preserved archive and executable
copies in the owned container workspace under
`artifacts/866f0c04f189c7039b07585d06c337d33e9cc296fab2fa05bb4fafe844bf9d72/`:

- `source.tar`: deterministic snapshot, every file compared against remote source.
- `pqnld-rs-debug` SHA256:
  `cbcfe14cb4489acb8ae56a5a1f4a027f375bd649197b42e9bff35fb2e13dad56`.
- `pqnld-rs-release` SHA256:
  `ded05dca02e73849c8f9a88da981e717bb20211086796c2477e53d12caa6d87a`.

The prior chunk-offset mutation-control remains coverage evidence, not Q5
behavior RED. Q5 RED is a genuine failure against the newly human-approved
contract. No repair retries, new dependency, formatter installation, dataset
rewrite or benchmark/model call. Main owns human-facing documentation and fresh
benchmark decisions; this task stops at green pinned handoff artifacts.

## Feature bindings and limits

Bindings for `../behaviors/reliability.feature`:

- Malformed choices / choice cardinality: observed
  `malformed_questions_refuse_before_any_engine_request`;
  `http_and_chat_share_input_refusal_without_engine_access`;
  `mcp::tests::malformed_decide_tool_questions_refuse_without_engine_access`.
  Covers missing/empty/nonobject criteria, 1/256 choices, malformed question,
   unsupported/non-string/missing type, required present instructions,
  empty questions, and malformed later question before earlier scoring.
- Missing echo evidence: observed `echo_refuses_empty_or_unextended_evidence`,
  `echo_refuses_nonfinite_scores_and_sum_overflow`, and
  `unusable_echo_evidence_returns_http_error_not_probabilities`.
  Adapter bindings:
  `echo_adapters_refuse_malformed_or_misaligned_scores` and
  `echo_adapters_preserve_initial_unscored_token_alignment`. Their fixture is
  synthetic loopback HTTP using existing libraries, not a model/inference call.
- Incomplete echo batches: `echo_refuses_wrong_batch_cardinality`,
  `vllm_echo_refuses_wrong_response_cardinality`, and
  `vllm_echo_refuses_misaligned_choice_indices`.
- Invalid startup temperature: `cli_temperature_requires_finite_positive_value`
  executes 29 actual binary startups across CLI, primary/legacy env and finite
  JSON descriptor settings. NaN/infinity cannot be represented in JSON.
- Supported decisions: observed `valid_http_and_chat_retain_typed_envelopes`,
  `specific_scores_produce_a_distribution`, `noul_reports_true_probability`,
  `over_128_options_split_and_merge_losslessly`, `validate_rejects_bad_distributions`,
  `valid_decide_tool_retains_structured_response`,
  `echo_preserves_null_prefix_and_boundary_retokenization`, and existing cache
  tests. `input_validator_accepts_255_choices_without_promising_tokenizer_capacity`
  checks wire acceptance only; the existing 151-option test exercises scoring.

Bindings for `../behaviors/global-workers.feature`:

- Independent clients / initial probes:
  `workers_one_bounds_independent_clients_and_initial_probes`;
  `workers_one_shares_budget_between_probe_and_question_scoring`.
  Both explicitly poll gated real readout/probe/scoring futures; no sleeps.
- Sequential order: `workers_one_preserves_stored_question_and_answer_order`.
  Concurrent success/order: `workers_two_preserve_answer_order_after_reverse_completion`.
- Failure cancellation/drain:
  `failed_concurrent_readout_cancels_and_drains_remaining_scoring` exercises
  error and panic, both before and behind blocked work, with notifications and
  a whole-case two-second deadlock watchdog. Confirms zero surviving fixture operations,
  all permits reclaimed and a complete subsequent decision.
- Direct engine chat: architectural boundary, not an executed live-chat test.

All named tests above pass remotely. `.feature` files remain behavior contracts,
not a Gherkin runner. Synthetic adapters establish parse compatibility, not
live provider/model correctness, calibration, or mixed-MTP readiness.

## Sanctioned Apollo runner / recovery

Q-RUNNER-PQNLD resolved by main's explicit named dev-target authorization.
Host workspace:
`/mnt/data/ai/validation/decision-index-2026-10-01/pqnld-reliability-20261002`;
container workspace: `/work/pqnld-reliability-20261002`. Path was checked absent
before exclusive creation at mode `0700`; marker
`.task-id` contains `pqnld-reliability-20261002-7342` (mode `0600`).
[apollo-workspace.log](apollo-workspace.log) confirms current ownership marker,
mode and installed compiler/Cargo versions. Rust is existing stable 1.99.0;
no toolchain installation or host configuration change.

Read-only SSH established documented owner flow:

- `docs/decision-index-validation.md` names host workspace
  `/mnt/data/ai/validation/decision-index-2026-10-01`, `pm.sh`, and container
  `dibuild` with `/work` mount.
- Host `pm.sh` wraps existing privileged podman binary with dedicated root and
  runroot. `pm.sh inspect dibuild` confirms running container and read-write bind
  of that host workspace to `/work`.
- `pm.sh exec -i dibuild python3 -` performed directory listings only:
  `/work/pqnld-rs` has existing `Cargo.toml`, `Cargo.lock`, `src`;
  `/work/rustup/toolchains/stable-x86_64-unknown-linux-gnu` and `/work/cargo/bin`
  exist; `/work/cargo-target` has prior release/musl artifacts.
- `sidecar-rs/build.sh` points to these toolchain/cache paths but hardcodes
  `/work/pqnld-rs` and auto-installs if missing. **Do not execute it unchanged**:
  preserve existing source/data, reject installation, isolate task target dir.

Runner environment:

```sh
W=/mnt/data/ai/validation/decision-index-2026-10-01
$W/pm.sh exec -w /work/pqnld-reliability-20261002 dibuild env \
  CARGO_HOME=/work/cargo RUSTUP_HOME=/work/rustup \
  PATH=/work/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  CARGO_TARGET_DIR=/work/pqnld-reliability-20261002/target \
  CARGO_NET_OFFLINE=true CARGO_TERM_COLOR=never RUSTC_WRAPPER= make test
```

Only `Makefile`, `sidecar-rs/{Cargo.toml,Cargo.lock,src/,models/,tests/}` transfer.
No secrets, raw datasets, `.git`, local targets, old workspace edits or original
artifact changes. Existing compiler and dependency cache are reused; task build
artifacts are isolated. No live model requests; loopback fixtures only.

## Limits and handoff

- Full Apollo debug and release test gates green; local `git diff --check` green. Remote
  [cargo fmt](apollo-final-fmt.log) and [Clippy](apollo-clippy.log) cannot run:
  existing toolchain lacks `cargo-fmt` / `cargo-clippy`. No components installed.
  Earlier pre-restriction formatting check also showed baseline differences in
  all four Rust source files. No blanket formatting or unrelated fixes applied;
  changed-code formatter/lint verification remains unproven.
- Worker budget bounds active local sidecar scoring, including probes; it does
  not bound direct engine clients, inbound/body memory or already accepted
  upstream requests after local cancellation. No global determinism claim.
- 255 choices pass input validation, not a tokenizer-capacity guarantee.
  Echo remains length-sensitive; no calibration or quality result.
- Main owns final independent review, planning/wire/worker documentation updates,
  and any authorized publication/deployment. This task invoked no nested agent
  or RV workflow, committed/pushed nothing, and left task workspace intact.
