# Build examples

Reference rigs we actually built, served, and measured. Each directory is a
recipe: the hardware, the engine build and serve flags, the `pqnld` command,
and the raw receipts behind every number.

| Example | Hardware | VRAM | Engine profile | Headline |
|---|---|---|---|---|
| [`2x-rtx5060ti`](2x-rtx5060ti/README.md) | 2× RTX 5060 Ti | 32 GB | vLLM TP=2, MTP-6, FP8 KV, 200k ctx | one decision ~0.12 s; decisions and chat batch together on one engine; 62 tok/s single-stream decode at 95k ctx |

These are not benchmarks rigged for a headline. They are the runs cited in
[`docs/benchmarks.md`](../docs/benchmarks.md), kept raw so you can line them up
against your own rig. Hardware is whatever it is; a result on two mid-range
cards is more useful to most people than one on a datacenter part.

## Adding your own

Drop a `build_examples/<name>/` with a `README.md` and a `results/` directory
holding the JSON your run produced. Say what the hardware was, the exact engine
flags, and which benchmark script produced each file. Numbers without the
receipt are a rumour.
