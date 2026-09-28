# Strict readout smoke baseline

Measured 2026-09-27 against Apollo's existing engine at its tailnet-bound
port 11542 through a temporary SSH tunnel. Model request ID:
`qwen38-27b-nvfp4`. Engine revision and checkpoint digest were not independently
captured; this is a diagnostic smoke run, not a reproducible quality benchmark.

The local working tree used strict letter recognition, complete-label coverage,
and order-sensitive caching on top of `f541536`. No deployment or engine flags
were changed. Result caching was disabled; engine prefix-cache state was not
controlled. Timings include tunnel overhead.

Command after opening a local tunnel:

```sh
PYTHONPATH=src python3 benchmarks/bench_readout.py --url http://127.0.0.1:21542
```

The fixture selects an explicitly named item from an enumeration, once in each
option order. It deliberately avoids the ambiguous fruit question in the older
wire sample. It is not held-out calibration data or a Decision Index suite.

| Options | Original order | Reversed order |
|---|---|---|
| 2 | Correct, 165 ms | Correct, 127 ms |
| 3 | Correct, 126 ms | Correct, 127 ms |
| 10 | Correct, 272 ms | Refused: missing h, j; 275 ms |
| 20 | Refused: missing b, n, o, p, s, t; 268 ms | Refused: missing k, m, n, o, p, q, r, t; 267 ms |
| 26 | HTTP 400, 10 ms | HTTP 400, 11 ms |

Coverage: 5/10. Accuracy over answered cases: 5/5. Accuracy counting refusals
and errors as wrong: 5/10. All-case latency median: 146 ms, including fast
rejections; it must not be presented as successful-decision latency.

A separate request confirmed the 26-option error:

> Requested sample logprobs of 26, which is greater than max allowed: 20 (parameter=logprobs, value=26)

The test process completed and printed all ten rows. The enclosing shell later
timed out while managing the SSH tunnel; a subsequent socket check confirmed
the local tunnel listener was gone.

## Next gate

Inspect the running engine's supported mechanisms for obtaining specified-token
scores. Increasing top-k alone cannot guarantee complete label coverage. Compare
an exact candidate-score path or explicitly identified echo fallback before
resuming mixed-load measurements. Do not claim calibration from these ten easy
synthetic cases, and do not silently change logits through constrained sampling
without recording the resulting probability semantics.
