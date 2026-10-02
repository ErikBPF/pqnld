# Server-side vLLM optimizations for the DECISION readout

**Status:** plan only — not executed. Nothing here has been applied, and this
task did not restart, stop, or modify the engine or cache containers.
**Target:** Apollo engine `apollo-qwen38-gittensor-lmcache` (vLLM 0.30.0,
served id `qwen38-27b-nvfp4`, TP2, MTP mandatory).
**Workload:** many short `max_tokens=1` exact-token logprob requests that share
a long state prefix; measured per-question median ~258 ms; prefill-bound.
**Measurement:** per-question median + p90 on the existing 100-row Decision
Index sample; engine `/health` after; answer parity via `agree.py`.

## Baseline facts (grounded in the running engine)

- **Local prefix cache is effectively dead; LMCache carries ~25-50%.**
  Live `/metrics` during both concurrent and sequential decision runs:
  `Prefix cache hit rate: 0.0-2.1%`, `External prefix cache hit rate: ~20-50%`.
  Sequential run at 23:55: local 0.0%, KV usage only 8.0%, one running request.
- **Physical KV block is 864 tokens.** Startup logs:
  `Setting attention block size to 864 tokens to ensure that attention page size is >= mamba page size`;
  `kv cache group sizes [864, 864, 864, 864]`; `kv lcm block sizes 864`;
  LMCache `scheduler_block_size=864`.
- The Python sidecar already records the consequence: *"this hybrid model's
  unified block size (864) exceeds a decision prompt, so nothing is
  prefix-cached"* (`src/pqnld/decision.py:11`).
- **Cache geometry is inert for fine-grained hits:** `hash_block_size ==
  block_size == 864`, so `Scheduler.mamba_partial_cache_hit` is `False`
  (`vllm/v1/core/sched/scheduler.py:361-364`), which forces
  `mamba_fine_grained_prefix_cache = False` (`scheduler.py:369-372`).
- **vLLM warns about the step budget:** *"max_num_scheduled_tokens is set to
  1727 based on the speculative decoding settings. This may lead to suboptimal
  performance. Consider increasing max_num_batched_tokens..."*
- **vLLM recommends a higher memory fraction:** *"0.9000 is equivalent to
  0.8634 without CUDA graph memory profiling. To maintain the same effective KV
  cache size, increase --gpu-memory-utilization to 0.9366."*
- **FlashInfer autotune is explicitly off:**
  *"Skipping FlashInfer autotune because it is disabled."*
- **Async scheduling is already on by default.** `async_scheduling` is `None`,
  `"mtp"` is in `EagleModelTypes`, `disable_padded_drafter_batch=False`
  (default), and the MP executor reports `supports_async_scheduling() -> True`
  (`vllm/v1/executor/multiproc_executor.py:558`). No change needed; confirm.
- **TP2 all-reduce runs on PYNCCL.** Backends in dispatch order were
  `['FLASHINFER_PCIE_IPC','FLASHINFER','NCCL_SYMM_MEM','QUICK_REDUCE',
  'AITER_CUSTOM','CUSTOM','SYMM_MEM','PYNCCL']`; FlashInfer all-reduce is
  disabled for `world_size=2`; custom all-reduce is disabled by flag and the
  env sets `NCCL_P2P_DISABLE=1`.

## Ranked candidates

Ranked by expected gain vs cost/risk for a prefill-bound, shared-prefix,
short exact-ID workload. All are config-only unless noted; none needs a
rebuild. Every candidate is A/B'd one flag at a time on the original container's
`Config.CreateCommand`, then combined.

### 1. Finer prefix-cache granularity + Mamba junction checkpoints — HIGH gain, MEDIUM risk

**Change (both flags together):**
- add `--enable-mamba-fine-grained-prefix-cache`
- add `--prefix-match-unit 16` (largest divisor of 864 below it that also
  divides each spec's `tokens_per_state`; if the engine rejects it, read the
  `Got alignments=` list from the error and pick a multiple of those, still
  `< 864`)

**Why it should help:** the shared state prefix is only cacheable at 864-token
granularity, which exceeds most decision prompts and does not land on the
shared-prefix junction. `--prefix-match-unit` computes cache keys inside a
physical block (`vllm/config/cache.py:91-102`), so attention KV for the shared
state can be reused at 16-token boundaries. The Mamba flag additionally
registers a GDN/Mamba recurrent-state checkpoint at the *shared-prefix
junction where an EAGLE/MTP sibling resumes* (`vllm/config/cache.py:187`,
`scheduler.py:358-372`) — exactly the "N questions sharing one state under MTP"
shape. Without it each sibling re-prefills/replays the shared prefix; this is
the measured 0% local-hit / 258 ms behaviour.

**Risk:** correctness and nondeterminism — the cached recurrent state plus fp8
KV may differ from a fresh replay and can flip near-ties; this is the same
class of effect already documented for batched decode. Extra cache entries cost
VRAM. Chat is unaffected in principle (the flag is designed for MTP and MTP
stays). Verify the mechanism actually engaged: no
`Disabling fine-grained prefix-cache hits ... require block-aligned lookups`
warning, and the local `Prefix cache hit rate` rises above ~0%.

**Measurement:** per-question median + p90 vs baseline; `agree.py` must report
100/100 identical on a sequential run; repeat the run twice to check
sequential determinism; `/health` 200.

### 2. Raise `--max-num-batched-tokens` — HIGH gain, HIGH risk

**Change:** `1727 -> 4096`, then `8192` only if the engine starts and `/health`
passes.

**Why it should help:** prefill-bound workload; larger chunks amortize the
fixed per-step cost over more prompt tokens per iteration and clear the
engine's own "<8192 suboptimal" warning (`vllm/config/vllm.py:2135`). Directly
attacks the dominant cost (state prefill), not decode.

**Risk:** activation/VRAM peak grows; the repo already recorded that 4096/8192
failed to start at 200k context and 0.90 in this rig (`docs/benchmarks.md`,
"Raising --max-num-seqs"). Combine with candidate 3 to free KV headroom, and
treat a failed start as a rejected candidate, not a bug. Config-only.

**Measurement:** engine reaches `/health`; per-question median + p90; `agree.py`;
`/metrics` KV usage and preemption/deferred counts.

### 3. Raise `--gpu-memory-utilization` 0.90 -> 0.9366 — MEDIUM/HIGH gain, MEDIUM risk

**Change:** set the existing `--gpu-memory-utilization` value to `0.9366`.

**Why it should help:** the engine states 0.90 only yields the KV pool of a
0.8634 run after CUDA-graph memory profiling. Restoring the pool raises KV
capacity, keeps shared prefixes resident and reduces the 96% KV-pressure and
`Deferred`/`Waiting` queueing observed under concurrency. Also makes candidate 2
viable.

**Risk:** engine and LMCache server share each 15.48 GiB device; a higher
fraction can fail startup or OOM under load. The prior incident showed a
retained LMCache allocation blocking restarts; a cache restart may be needed in
the recovery procedure. Config-only.

**Measurement:** `/health`; `/metrics` `GPU KV cache usage` ceiling and
`Prefix cache hit rate`; per-question median + p90; `agree.py`.

### 4. Cudagraph mode `PIECEWISE` -> `FULL_AND_PIECEWISE` — MEDIUM gain, LOW/MEDIUM risk

**Change:** `--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE"}`
(the v1 default; `FULL_DECODE_ONLY` is the fallback if mixed capture is
rejected).

**Why it should help:** every decision is a single-token decode; PIECEWISE
leaves decode attention eager while FULL captures the whole decode batch and
removes per-step launch overhead (`vllm/config/compilation.py:618-641`).
Upstream default is `FULL_AND_PIECEWISE`.

**Risk:** longer capture and more graph memory; the mode is only valid when the
attention backend reports full-cudagraph support (FlashInfer does; the engine
falls back with a warning otherwise). Low chat impact. Config-only.

**Measurement:** `/health`; per-question median + p90; `agree.py`; startup log
must not show a cudagraph-mode downgrade.

### 5. `--enable-flashinfer-autotune` — MEDIUM gain, MEDIUM risk

**Change:** remove `--no-enable-flashinfer-autotune`.

**Why it should help:** the engine currently skips autotune entirely
(`kernel_warmup.py:255`). Autotune selects FlashInfer attention/GDN kernels for
this exact sm120 shape, which is the model-side prefill cost.

**Risk:** multi-minute startup, and autotune may exceed the current
`VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864`; raise that only as part of the
same trial if tuning fails. Potential run-to-run kernel-selection variance, but
selection is fixed per boot. Config-only.

**Measurement:** `/health`; per-question median + p90; `agree.py`.

### 6. `--scheduling-policy priority` + client priority — MEDIUM gain on mixed tails, LOW/MEDIUM risk

**Change:** add `--scheduling-policy priority`; have the sidecar send a higher
`priority` for decision requests.

**Why it should help:** `SchedulerPolicy = Literal["fcfs","priority"]`
(`vllm/config/scheduler.py`). Under simultaneous chat, decisions currently
queue behind ~95k-token prefills (p50 ~11 s). Priority lets decision requests
skip the chat backlog. No gain when the engine is idle.

**Risk:** requires a client change; mis-set priorities can starve chat.
Config-only server side.

**Measurement:** run the mixed chat+decision benchmark, not the 100-row sample
alone; report decision p50/p90 and chat recovery.

### 7. Confirm `--async-scheduling` stays on — NO action

Already enabled by default (see baseline). Only if a trial shows it was
downgraded would `--async-scheduling` be added explicitly. Config-only.

### 8. LMCache geometry and cross-run reuse — LOW gain for a single run

**Change:** keep server `--chunk-size 864` (it must match the 864 scheduler
block). Cross-run reuse helps only repeated runs of the same rows; do not
expect a single-run median gain. The 800 chunk is a no-MTP-only geometry
(`docs/mixed-readout-diagnosis.md`); MTP must stay.

**Diagnostic value:** an A/B *without* `--kv-transfer-config` isolates whether
the 0% local prefix hit is caused by the connector path or solely by block
granularity. Run it only after candidate 1, to confirm the mechanism.
Config-only but needs engine + cache restart and the established recovery
procedure.

### 9. Custom all-reduce / `NCCL_P2P_DISABLE` — NOT RECOMMENDED

Remove `--disable-custom-all-reduce` (or `NCCL_P2P_DISABLE=1`) only as a
diagnostic. On 2x consumer PCIe with P2P disabled and FlashInfer all-reduce
unavailable at `world_size=2`, custom all-reduce is likely to fail init or fall
back; the all-reduce itself is small relative to prefill, so upside is low and
nondeterminism/correctness risk is real. Leave disabled.

### 10. `kv-cache-dtype`, `--load-format` — NO GAIN

- Keep `--kv-cache-dtype fp8`: fp8_e4m3 doubles KV capacity and speeds
  attention; bf16 is a determinsm tradeoff, not a win.
- `--load-format ipc_cache` only affects startup weight mapping; no
  steady-state effect.

## Measurement protocol (exact)

1. **Baseline.** Original container running; `curl -fsS /health` returns 200.
   Run the 100-row sample sequentially through the existing canonical client
   (`--workers 1`):
   ```sh
   $W/venv/bin/python -m decision_index pipeline --engine uds_engine:UdsSystemOne \
     --option uds=/tmp/pqnld.sock --option model=qwen38-27b-nvfp4 \
     --rows /work/sample-100.jsonl.gz --suite-dir /work/suite --edition 0.2.1 \
     --out /work/runs/<baseline-tag>
   ```
2. **Per-question latency.** For each row in `results.jsonl`,
   `model_request_wall_ms / len(payload.questions)`; report median and p90 over
   all questions (this isolates per-question cost from row fan-out). Also
   report total wall from the event log.
3. **Parity.** `_evaluation`/`raw_output` compared with the baseline via
   `agree.py`; a sequential candidate run must be 100/100 identical.
4. **Determinism.** Repeat the candidate run; two sequential runs must agree.
5. **Engine health.** `curl -fsS /health` 200; capture `/metrics`
   (`Prefix cache hit rate`, `External prefix cache hit rate`, `GPU KV cache
   usage`, `Deferred`/`Waiting`) before and after.
6. **A/B procedure.** Reuse the established safe procedure: capture
   `$W/pm.sh inspect apollo-qwen38-gittensor-lmcache` -> `Config.CreateCommand`;
   generate a temp command (rename `--name`, change only the tested flags);
   `pm.sh stop apollo-qwen38-gittensor-lmcache`; wait for both GPUs >=15000 MiB
   free (restart the LMCache server if it retains memory); `pm.sh run` the temp
   engine; measure; `pm.sh stop` + `pm.sh rm -f` the temp container;
   `pm.sh start apollo-qwen38-gittensor-lmcache` to restore the exact original.
   Never leave a temp container running.

## Top 5 (summary)

| # | Change | Expected gain | Risk |
|---|---|---|---|
| 1 | `--prefix-match-unit 16` + `--enable-mamba-fine-grained-prefix-cache` | High — removes re-prefill of the shared state; attacks 0% local prefix hit | Correctness/nondeterminism, VRAM |
| 2 | `--max-num-batched-tokens` 1727 -> 4096/8192 | High — larger prefill chunks; engine's own warning | Startup/OOM at 200k |
| 3 | `--gpu-memory-utilization` 0.90 -> 0.9366 | Medium/High — restores KV pool, retains prefixes | Engine/LMCache VRAM contention |
| 4 | `cudagraph_mode` -> `FULL_AND_PIECEWISE` | Medium — captures 1-token decode graphs | Capture time/memory |
| 5 | Remove `--no-enable-flashinfer-autotune` | Medium — better FlashInfer kernels for sm120 | Startup time, workspace size |

Verify in order 1, 3, 2, 4, 5: #1 is the only candidate that addresses the
measured mechanism directly; #3 is the cheapest enabler for #2.
