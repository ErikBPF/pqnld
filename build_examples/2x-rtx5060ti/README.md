# 2× RTX 5060 Ti — one engine, chat and decisions

The rig we developed `pqnld` on. Two consumer 16 GB cards, no datacenter GPU,
one vLLM engine serving chat completions and decisions at the same time. Every
number below has a raw receipt in [`results/`](results/).

## Hardware

| | |
|---|---|
| GPU | 2× NVIDIA RTX 5060 Ti, 16311 MiB each (32 GB total) |
| Parallelism | tensor-parallel 2 |
| Host | Linux, NVIDIA driver with FP4 support (Blackwell) |

## Model

`Qwen3.8-27B-NVFP4` (Gittensor `modelopt_fp4` snapshot), served as
`qwen38-27b-nvfp4`. It is a thinking/chat model — `pqnld` suppresses the
thinking block through the descriptor, which is the whole trick.

## Serve the engine

[`serve-engine.sh`](serve-engine.sh) is the profile the receipts were measured
with. It needs a model directory:

```sh
MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4 ./serve-engine.sh
```

`docs/benchmarks.md` lists the flags that matter. The short version: FP8 KV
cache, `--max-num-seqs 8`, MTP speculative decoding with 6 draft tokens, and
PIECEWISE CUDA graphs. At 200k context the KV cache and the CUDA-graph
activations compete for the same 32 GB, which is why the token budget is pinned
to 1727 (see "What did not work" below).

## Run pqnld against it

```sh
pqnld \
  --vllm-url http://127.0.0.1:11542 \
  --model qwen38-27b-nvfp4 \
  --descriptor qwen38-27b-nvfp4 \
  --host 0.0.0.0 --port 11560
```

`qwen38-27b-nvfp4.json` ships in the package
([`src/pqnld/models/`](../../src/pqnld/models/qwen38-27b-nvfp4.json)); the
startup probe confirms the lettered readout works on this model and would fall
back to the echo path otherwise.

## Speeds

### A single decision, warm (one at a time)

Receipt: `results/decide-latency.json` (5 runs each).

| Questions per decision | p50 | best |
|---|---|---|
| 1 | 0.119 s | 0.118 s |
| 2 | 0.229 s | 0.229 s |

A decision is one prefill of the ~99-token prompt and one output token. The
sidecar adds ~nothing; a single-question `/v1/decide` costs the same as calling
vLLM directly. Two questions cost two scheduler steps because that is how the
engine admits these tiny requests (each step is ~0.12 s), not because the
sidecar reads them serially.

### Decisions and chat together

Half decisions, half chat completions, fired concurrently.
Receipts: `results/parallel.json`, `results/parallel-numseq8-each4.json`,
`results/parallel-numseq8-each8.json`.

| `--max-num-seqs` | each | in flight | decision p50 | decision p95 | chat p50 | wall | req/s | parallelism |
|---|---|---|---|---|---|---|---|---|
| 2 | 4 | 8 | 2.86 s | 3.10 s | 1.66 s | 3.10 s | 2.58 | 5.59× |
| 8 | 4 | 8 | 1.39 s | 1.39 s | 1.10 s | 1.39 s | 5.74 | 6.37× |
| 8 | 8 | 16 | 2.55 s | 3.08 s | 1.62 s | 3.08 s | 5.19 | 10.49× |

Zero failures in all three. `parallelism` is total latency over wall time — at
8 requests in flight it is >1, so they are genuinely overlapping rather than
serialized. Raising `--max-num-seqs` from 2 to 8 cut the wall by 55% and doubled
throughput. The p50s are queueing under load, not per-call latency; a lone
decision is still 0.12 s.

### Chat throughput with decisions interleaved

Long conversations (95k-token context, 1024-token answers) plus 100 decisions
at 16 workers. Receipt: `results/mixed.json`.

| Phase | Decode throughput | Note |
|---|---|---|
| Single conversation | 62.4 tok/s | prefill 1205 tok/s, TTFT 78.9 s |
| Two conversations | 22.1 tok/s aggregate | second TTFT 159 s — prefills do not overlap |
| Two conversations + 100 decisions | 12.9 tok/s aggregate | chat retained 58%; decisions 100/100 ok, p50 11.1 s, p95 79.3 s |
| Two conversations, after decisions | 22.0 tok/s | 99.6% of the pair baseline |

The 11 s decision p50 here is queueing behind 180 s of prefill, not decision
cost. Under that pressure the engine still ran all 100 decisions in parallel
(27.5×) and gave the chat throughput back afterwards.

### Kit acceptance

`results/s2-kit-results.jsonl` is the authoritative Decision Index kit's `http`
engine run against the sidecar: 7/7 rows `ok`, correct argmax, distributions
summing to one. Synthetic sample, not the frozen suite — wire compatibility,
not a Decision Index score.

## What did not work

- `--max-num-batched-tokens` above 1727: at 200k context and
  `--gpu_memory_utilization 0.90` the KV cache needs ~3.59 GiB and there is
  almost nothing left. Both 4096 and 8192 starved the cache and the engine
  refused to start. Raising it means lowering `--max-model-len` or raising GPU
  utilization (which OOMs).
- `--enforce-eager` and `--max-num-seqs 1`: slower (~0.27 s per decision), so
  the CUDA graphs and batching stay.
- Forced-continuation scoring (`prompt_logprobs`): CUDA-OOMs at 94% GPU
  utilization. The lettered answer-slot readout is the route.

Lower `--num-speculative-tokens` (the `NUM_SPECULATIVE_TOKENS` knob in the
serve script) is untried and would free token budget at a decode-speed cost.
