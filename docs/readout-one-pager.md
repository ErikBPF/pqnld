# PQNLD readout improvements: implementation and live verification

**Stage / revision:** RV / 5  
**Status:** local correctness improvements complete; shared-chat readiness blocked  
**Owner / date:** PQNLD / 2026-09-28 UTC  

Latest review: two independent passes corrected patch-generation atomicity and
CPU regression gaps. Local suite: 48 passing tests; both benchmark self-tests
pass. Revised candidate routing also passes inside the running image using CPU
tensors and in-memory source transformation. Added GPU numerical check is syntax
checked but unexecuted. No candidate image, GPU numerical result, distributed
gather result or repaired mixed-serving result exists yet. Those gates remain
blocked on GPU capacity/maintenance; no new serving performance claim is made.

**Basis:** improve Jev-like typed decisions and normal chat using one resident LLM;
"continue. Do all necessary improvements and show me a one pager in the end".
Publication candidate: branch `test/readout-baseline`, based on `f541536`.
The original MTP engine and cache are restored after authorized diagnostic trials.

Latest: coordinated no-MTP + cache800 testing passed all six concurrent score
checks and ten isolated option/permutation cases. Counting chat was about 5.6x
slower. Original MTP + cache864 containers are restored and healthy. See the
[successful experiment](mixed-readout-diagnosis.md#successful-coordinated-experiment-2026-09-28).

## Outcome and why it matters

**Human correction:** MTP is mandatory. Fast completions are the primary value;
decisions are a lower-priority capability of the same resident model, used when
needed. A separate model deployment and permanently disabling MTP are excluded.
The no-MTP experiment is diagnostic evidence only, not a rollout candidate.

The deployed MTP engine fails simultaneous exact-token scoring and chat. A temporary
no-MTP engine with compatible LMCache geometry passed bounded concurrent checks,
but slowed the measured counting chat substantially. Repair speculative scoring
before adapters or reinforcement learning. Sustained-load acceptance remains open.

## What changed

The readout accepts only exact letter-shaped tokens instead of interpreting words
as labels. Missing option scores now cause refusal instead of invented low scores
or uniform confidence. Cache keys preserve option order, matching the rendered
prompt. Generic auto-probing still falls back to echo when top-k coverage fails;
echo remains a different, length-sensitive scoring method.

An experimental descriptor flag, `specific_token_scores`, uses the live engine's
tokenizer and requests the exact label IDs through `logprob_token_ids`. It validates
single-token, distinct mappings and requires complete returned coverage. It does
not bias logits or constrain decoding. The flag stays **off by default**, including
the Qwen descriptor, because mixed-chat validation failed. A repeatable synthetic
benchmark and regression tests cover the new path. Independent review identified
an auto-probe fallback regression; that was reproduced and corrected.

## Evidence and limits

| Check | Result | Limit |
|---|---|---|
| Existing strict top-k live smoke | 5/10 answered; all five correct | Missing labels at 10/20 options; 26 exceeded engine top-k cap |
| Exact-ID live smoke | 10/10 answered and correct; median 267 ms | Ten synthetic examples only; includes token-ID discovery; no result cache |
| Option permutations | Both orders passed at 2, 3, 10, 20, 26 options | Not unseen-domain quality evidence |
| Shared chat + exact-ID decision | Chat succeeded; decision HTTP 500, reproduced | Engine log: `IndexError` in `_create_chat_logprobs`; also fails with top-k 1 |
| Controlled overlap diagnostic | Isolated: complete; chat without scores: HTTP 500; chat with top-k: HTTP 200 but wrong score set | Both chats active at decision send and return; see [engine diagnosis](mixed-readout-diagnosis.md) |
| Unit regressions | 45 tests passed; both benchmark self-tests passed; diff check clean | Canned scores establish plumbing, not calibration; existing socket/resource and dependency warnings remain |

Engine identity was observed through `/version`: vLLM `0.30.0`; response fingerprint
`vllm-0.30.0-tp2-20ea93de`, served model `qwen38-27b-nvfp4` on Apollo. The
weight digest remains unpinned; the active image ID is recorded in the
[engine diagnosis](mixed-readout-diagnosis.md). The isolated smoke command is:

```sh
PYTHONPATH=src python3 benchmarks/bench_readout.py --url http://localhost:21542 --specific-token-scores
```

## What continues / what waits

Next, fix explicit-token logprobs in the engine's speculative rejection sampler
in its owning deployment/image source, then rerun concurrent chat/decision checks
and sustained latency measurements. A sidecar-only lock cannot coordinate ordinary
chat sent directly to the engine, so it is not a sound fix. The candidate patch
generator and CPU control-flow checks are included under `build_examples/2x-rtx5060ti/`;
GPU numerical correctness and patched mixed-serving performance remain unverified.

Publication rerun (2026-09-28): exact-ID smoke 10/10, median 266 ms. In the
eight-case sequential comparison, exact, constrained JSON and reason-then-score
each answered 8/8 correctly, with medians 190 ms, 373 ms and 1,030 ms respectively.
Echo answered 6/8 correctly and refused two prefix-overlapping cases; its answered
median was 491 ms. Concurrent MTP checks still returned HTTP 500 without chat
logprobs and incomplete scores with chat top-k. Both chats succeeded. These are
synthetic smoke results, with uncontrolled engine-cache state, not calibration
or sustained-load evidence. No deployment was changed for this rerun.

After serving passes, pin the Decision Index suite and evaluate held-out coverage,
NLL/Brier, reliability, and risk/coverage before claiming calibrated decisions.
Use supervised decision LoRA only if a measured quality gap justifies it; evaluate
RL later. The published RLCD recipe is not available for a faithful reproduction.

## Risks and recovery

Follow-up correctness revision: prefix-overlapping echo keys now produce a refusal
before inference. The >26-option regression fixture uses prefix-free padded keys
to continue exercising the supported fallback. A dedicated regression covers
the observed `item-1` / `item-13` failure.

Read-only source inspection and controlled overlap now localize the defect to
speculative decoding: the API clears ordinary `logprobs` for explicit-ID requests,
but the rejection sampler uses ordinary logprob settings and omits explicit-ID
gathering. Another request's top-k setting changes the decision from missing rows
to the wrong score set. Apollo runs MTP with six speculative tokens. Padding the
serializer or enabling chat logprobs would hide the defect, not fix it. No engine
patch was applied. Startup logs confirm the V2 runner; the per-batch path still
needs instrumented verification.

The supported interpretation is separate decision/chat requests sharing one base
deployment. The PQNLD chat shim itself is still not ordinary chat. More than 26
options use the existing echo path, outside this live test. Restore a descriptor's
flag to false to leave experimental scoring; do not restore fabricated missing
scores. Keep deployment gated on mixed-load success and quality claims gated on
held-out evidence. See [baseline receipts](readout-baseline.md) for the earlier run.
