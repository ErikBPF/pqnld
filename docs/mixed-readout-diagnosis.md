# Mixed readout: speculative logprob incompatibility

Observed 2026-09-27 on Apollo, vLLM `0.30.0`, fingerprint
`vllm-0.30.0-tp2-20ea93de`, model `qwen38-27b-nvfp4`.
Runtime startup logs show MTP speculative decoding with six speculative tokens,
tensor parallelism two, and a maximum of eight sequences. Image/weight digests
were initially unpinned. Follow-up identified the active image ID as
`dce379422d6420af4006eacb78c677d2e864446858ab6b31b28aad2e86c6a7bf`.
Startup logs explicitly confirm `Using V2 Model Runner`.

## Reproducer and observed results

```sh
PYTHONPATH=src python3 benchmarks/mixed_readout.py --url http://localhost:21542
```

Use an otherwise idle engine. The script discovers three label IDs, runs one
isolated decision, then sends the same decision after receiving streamed chat
content. Each chat is bounded to 128 tokens. It records whether chat is still
active at decision send and return, preserves HTTP error bodies and raw responses,
and checks complete finite scores rather than treating HTTP 200 as success.
Client overlap does not itself prove batch membership; engine instrumentation
remains necessary to verify the exact runtime branch.

| Condition | Decision result | Latency | Chat |
|---|---|---:|---|
| Isolated | All requested IDs 64, 65, 66 returned | 136 ms | Not run |
| Chat without logprobs | HTTP 500: `list index out of range` | 338 ms | Successful; active at send and return |
| Chat with top-k 3 | HTTP 200; IDs 65, 33, 292 instead of 64, 65, 66 | 337 ms | Successful; active at send and return |

These are three diagnostic observations, not throughput, reliability, or quality
estimates. The existing strict readout rejects the incomplete third response.

## Source-level causal chain

Paths below are relative to the installed `vllm` package read through the running
process root. They describe this installed snapshot, not every upstream release.

1. `entrypoints/openai/chat_completion/protocol.py:736` sets
   `SamplingParams.logprobs=None` when `logprob_token_ids` is present, regardless
   of the requested `top_logprobs`. This explains why top-k 1 did not help.
2. `v1/worker/gpu/sample/states.py:56` records ordinary `sampling_params.logprobs`,
   so an explicit-ID request contributes `NO_LOGPROBS` to that state.
3. `v1/worker/gpu/model_runner.py:1552` selects ordinary sampling when no draft
   tokens exist and rejection sampling otherwise. Ordinary sampling gathers
   explicit IDs (`v1/worker/gpu/sample/sampler.py:163`).
4. `v1/worker/gpu/spec_decode/rejection_sampler.py:107` returns no logprob tensors
   when ordinary logprobs are absent. When another request enables them, it calls
   `compute_topk_scores` without explicit-ID state. The model runner also explicitly
   excludes token IDs in its speculative gathering path at line 1579.
5. The scheduler expects scores because `SamplingParams.num_logprobs` recognizes
   explicit IDs (`sampling_params.py:803`). Missing rows eventually reach
   `entrypoints/openai/chat_completion/serving.py:1259`, which indexes them and fails.

The older runner's rejection sampler also gathers generic top-k only. The active
runner is now confirmed as V2; per-batch instrumentation remains outstanding. Both
the source mismatch and the controlled change from absent scores to generic
top-k scores point to speculative explicit-ID handling, not a PQNLD cache bug.

## Repair gate

Implement explicit-ID gathering for all generated positions in the owning
engine/image source, preserving per-request boundaries, raw score semantics,
and speculative acceptance/rejection mapping. Add engine regressions for isolated
and mixed requests with/without ordinary logprobs, including multi-token outputs.
Then run this diagnostic, the 26-label smoke, and sustained mixed-load checks.

An alternative experiment is a restart with speculative decoding off;
that tests a simpler serving configuration but may reduce chat throughput.
Do not fix this by padding missing rows, forcing chat logprobs, biasing labels,
or locking only the sidecar. Keep `specific_token_scores` opt-in until the gate passes.

## Authorized no-MTP experiment

The user authorized a temporary restart and restoration. A temporary container was
created from the original container's recorded creation command, replacing the
image tag with its exact image ID and removing only `--speculative-config` and
its value. The original container remained intact and stopped during the trial.

The trial failed during initialization: workers reported only 10.83/10.81 GiB free
on 15.48 GiB devices, below the configured 90% requirement. No inference checks ran;
this does not establish whether disabling MTP resolves scoring. The memory deficit
may have been transient, but its cause was not established. No other process was
stopped and no memory limits were changed to force the experiment through.

The original container was restarted successfully and `/health` passed. Its startup
then reported 14.92/14.90 GiB free. The temporary container was removed and the SSH
tunnel closed. The retry recommendation from this initial trial was superseded
by the coordinated cache experiment below.

Follow-up retry on 2026-09-28 waited three minutes for GPU memory to stabilize;
it did not, so no trial container was started. Restoration was also blocked by
the free-memory check. `nvidia-smi` identified the remaining allocation as the
existing LMCache server (PID 844249), holding approximately 4.4 GiB on each GPU.
This establishes the allocation owner, not the internal reason it retained memory.

With separate user authorization, the LMCache container was restarted, clearing
volatile cache state. Free GPU memory rose to 15697/15832 MiB. The original engine
then restarted successfully with its original image and MTP configuration.
Both cache and engine health checks passed; a chat request returned `OK` with
the original fingerprint. The experiment is stopped. No no-MTP inference evidence
was obtained. Future restarts must account for LMCache's retained GPU allocations
and explicitly include any necessary cache restart in their recovery procedure.

Runtime owner: custom Podman store `/mnt/microvms/ai/cache/podman-root`, runroot
`/run/apollo-ftw-containers`; original container
`apollo-qwen38-gittensor-lmcache`. Build assets live at
`/mnt/data/ai/validation/20260922-qwen38-repro`. Its `run-lmcache.sh` differs from
the active container's arguments, so it must not be used blindly for restoration.

## Successful coordinated experiment (2026-09-28)

After authorization to proceed with serving gate 1, coordinated cache restarts
resolved retained allocations. A no-MTP startup then exposed a second requirement:
LMCache's 864-token chunk size must be a multiple of the no-MTP engine's 800-token
block size. A temporary cache container with `--chunk-size 800`, together with a
temporary engine with speculative configuration removed, started successfully.
Both were derived from recorded creation commands with exact image IDs. Original
containers were preserved, stopped during testing, and restored afterward.

Results:

| Check | Observed result |
|---|---|
| Three repetitions of isolated three-label scoring | 3/3 complete finite score sets |
| Scoring during chat without logprobs | 3/3 complete; 433-438 ms |
| Scoring during chat with top-k logprobs | 3/3 complete; 432-489 ms |
| Chat overlap | All six chats successful and active at decision send and return |
| 2/3/10/20/26 options, both orders, isolated | 10/10 correct and complete; median 252 ms |
| 128-token counting chat, original MTP | Three calls: 1.102, 1.124, 1.095 seconds |
| Same chat, no MTP + cache800 | Three calls: 6.213, 6.165, 6.236 seconds |

The median end-to-end counting response was about 5.6 times slower without MTP.
These timings include SSH/HTTP overhead and prefill, not isolated decode TPS or
TTFT. Repeated counting is favorable to speculative decoding; do not generalize
the ratio to typical chat. The cache configuration also changed, and runs were
sequential rather than randomized. This is bounded diagnostic evidence, not a
sustained-load acceptance test. Concurrent 26-label/permutation coverage remains
to be measured (the concurrent diagnostic used three labels).

Conclusion: no-MTP plus compatible cache geometry isolates the scoring defect,
but is excluded from deployment. Preserve MTP by repairing explicit-ID gathering
in the V2 rejection sampler before sustained acceptance tests. The deployed engine
still uses original MTP and cache864; experimental scoring remains opt-in. Both
original health checks passed, temporary containers were removed, and the tunnel
was closed. No engine patch or permanent configuration change was made.

## MTP-preserving patch preparation

The human requires MTP for fast completions; decisions are a lower-priority
capability on the same resident model. No-MTP is excluded from deployment.

`build_examples/2x-rtx5060ti/prepare_mtp_patch.py` emits a source-checked unified
diff for the inspected snapshot. It wires existing explicit-ID scoring into the
V2 rejection sampler, passes expanded request mappings through each verification
chunk, and includes explicit-ID dimensions in sharded output gathering. A global
batch maximum keeps logprob column widths compatible across verification chunks.
Ordinary requests without logprobs preserve the existing early exit. This is a
candidate patch generator, not an installed engine change.

`check_mtp_scores.py` extracts the installed scoring method and exercises routing
with CPU tensors, stubbed flattening and a recording scoring function. It failed
against installed source with `speculative scorer lacks explicit-ID request
mapping`, then passed against the proposed transformation in memory. Both modified
modules compile. No installed files were changed and no GPU allocations were used
by this check. This verifies argument routing and boundaries, not numerical GPU
correctness, full chunk orchestration, adaptive verification, or distributed gather.

Before deployment: apply the diff in a pinned candidate image, verify raw numerical
scores and request mappings across chunked/multi-token/adaptive cases, run repeated
mixed 26-option/permutation checks, and compare completions TTFT/throughput/tail
latency with the unpatched MTP baseline. Decision concurrency must be bounded;
engine-level priority guarantees and an acceptable performance budget remain open.

Follow-up CPU checks execute the installed chunk orchestration with synthetic
noncontiguous request slots (7, 2, 9), multiple logits per request, and two
verification chunks. They confirm per-chunk expanded mappings, raw-logit slices,
rebased boundaries, common output widths, and concatenated request offsets.
An adaptive-boundary check confirms device offsets are preserved. Verification
and GPU scoring remain stubbed: these checks do not establish numerical kernel
correctness or end-to-end adaptive verification behavior.

## Review revision 2

Independent review found and corrected partial-patch output on source drift,
missing explicit-ID-only CPU coverage, and a constant-width fake that could hide
incorrect request-slot selection. The generator now buffers both transformations
before output and explicitly enables ID gathering in the model runner. A focused
test covers that transformation; distributed gathering remains unverified.

`check_mtp_gpu.py` adds a numerical candidate-image check using real scorer kernels
and a PyTorch log-softmax reference, including sampled columns, mixed ordinary and
explicit-ID requests, and unequal accepted lengths. It has **not been executed**:
the serving GPUs had only 424-438 MiB free. CPU routing passed with the candidate
transformed in memory; the live engine is unchanged. A spare GPU or a coordinated
maintenance window is required for GPU checks, image validation and serving tests.
Real chunk concatenation, adaptive numerical checks and distributed gathering
remain required before a deployment claim.
