# Mixed readout: explicit-ID serving gate

## Current explicit-ID serving status

**MTP stays enabled. Exact-ID shared-chat serving remains opt-in and unaccepted.**
The deployed engine uses its original MTP configuration and cache864; no engine
patch is deployed. The no-MTP/cache800 configuration is diagnostic evidence,
not a normal-deployment candidate.

The unresolved engine defect is specific: mixed chat and explicit-ID requests
can return **incomplete scores, wrong token IDs or HTTP 500**. The strict
[Rust readout](../sidecar-rs/src/main.rs) refuses missing label evidence; it does
not turn an HTTP 200 into invented probabilities.

One resident model serves chat and decisions; fast completions remain primary.
The [sidecar worker budget](../sidecar-rs/README.md#scheduling) bounds scoring and
probes, not chat sent directly to the engine. A sidecar lock cannot establish
engine-level mixed-serving correctness.

## Evidence

The inspected engine is vLLM `0.30.0`, fingerprint `vllm-0.30.0-tp2-20ea93de`,
model `qwen38-27b-nvfp4`. Startup logs confirm **V2 Model Runner**, MTP with six
speculative tokens, tensor parallelism two and a maximum of eight sequences.
The identified image ID is
`dce379422d6420af4006eacb78c677d2e864446858ab6b31b28aad2e86c6a7bf`.
These findings belong to that inspected snapshot, not every vLLM release.

| Explicit-ID condition | Observed result |
|---|---|
| Isolated, three labels | All requested IDs 64, 65, 66 returned |
| Overlapping chat without logprobs | HTTP 500: `list index out of range` |
| Overlapping chat with top-k 3 | HTTP 200; IDs 65, 33, 292 instead of 64, 65, 66 |

The overlapping chats succeeded and remained active at decision send and return.
Client overlap is evidence of shared load, not proof of batch membership;
per-batch engine instrumentation remains outstanding.

### Owning source mismatch

Paths are relative to the installed `vllm` package inspected through the running
process root:

1. `entrypoints/openai/chat_completion/protocol.py:736` sets ordinary
   `SamplingParams.logprobs=None` for `logprob_token_ids` requests.
2. `v1/worker/gpu/sample/states.py:56` records ordinary logprobs, so an
   explicit-ID request contributes `NO_LOGPROBS` to that state.
3. `v1/worker/gpu/model_runner.py:1552` selects rejection sampling when draft
   tokens exist. Ordinary sampling gathers explicit IDs in
   `v1/worker/gpu/sample/sampler.py:163`.
4. `v1/worker/gpu/spec_decode/rejection_sampler.py:107` returns no score tensors
   without ordinary logprobs; when another request enables them, it gathers
   generic top-k without explicit-ID state. The model runner excludes explicit
   IDs from speculative gathering at line 1579.
5. The scheduler still expects explicit-ID scores (`sampling_params.py:803`).
   Missing rows reach `entrypoints/openai/chat_completion/serving.py:1259`,
   which indexes them and fails.

The absent-score/generic-top-k split matches the observed responses. The defect
belongs to speculative explicit-ID handling, not the PQNLD result cache.

### Candidate evidence, not deployment proof

CPU component checks cover request mappings, raw-logit slices, rebased boundaries,
common output widths and concatenated offsets across verification chunks.
Adaptive-boundary checks preserve device offsets; verification and scoring in
those CPU checks are stubbed.

The candidate scorer's **GPU numerical check passed** against PyTorch
`log_softmax` on two devices. For `topk` in `(-1, 3)`, sampled and explicit-ID
columns, `-inf` padding and `cu_num_generated_tokens=[0,2,3,6]` matched the
reference for mixed ordinary/explicit-ID requests with unequal accepted lengths.
This checks real scoring kernels, not a patched engine's full serving path.

The [pinned Decision Index sample](decision-index-validation.md) supplies separate
end-to-end evaluation evidence. Its 100/100 completion is not shared-MTP
acceptance. [Quality receipts](decision-index-quality.md#evidence-and-limits)
retain evaluation provenance and its narrower scope.

## Remaining acceptance checks

- Complete finite explicit-ID evidence under mixed chat with and without ordinary
  logprobs, across generated positions and speculative acceptance/rejection maps.
- GPU chunk concatenation with **unequal chunk widths**, adaptive-verification
  numerics and **TP2 sharded gathering**.
- Repeated mixed 26-option/permutation coverage and sustained chat/decision load
  on one pinned candidate engine; full patched shared serving remains unverified.
- Completion TTFT, throughput and tail latency against the unpatched MTP baseline,
  with a defined decision-concurrency/performance budget and engine priority policy.

No padding missing rows, forcing chat logprobs or biasing labels to make checks
green. The gate is complete, comparable score evidence without sacrificing the
primary chat workload—not a friendlier error message.

Runtime owner: `apollo-qwen38-gittensor-lmcache`, custom Podman store
`/mnt/microvms/ai/cache/podman-root`, runroot `/run/apollo-ftw-containers`.
Build provenance lives at `/mnt/data/ai/validation/20260922-qwen38-repro`;
its `run-lmcache.sh` differs from active arguments and is not a restoration recipe.
