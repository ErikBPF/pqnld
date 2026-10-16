# Running pqnld against llama.cpp

llama.cpp is a supported engine. This is the recipe that has been exercised live
(2026-10-13 to 2026-10-16, CPU and GPU, 3B to 27B, three quantisations of 27B).

## 1. Start `llama-server`

```bash
llama-server -m /models/model.gguf --host 127.0.0.1 --port 8080 \
  -ngl 99 -c 8192 --parallel 1 -t 8
```

CPU-only works (`-ngl 0`) and is the right choice when a GPU is busy; the readout
is identical, just slower. On a container, the CUDA driver must be visible — a
CUDA build with no `libcuda` **silently falls back to CPU**, so verify with a
`/props` call or by watching VRAM. Where the host does not inject the driver, mount
it explicitly:

```bash
-v /path/to/nvidia-driver/lib:/usr/local/nvidia/lib64:ro \
-e LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64
```

## 2. Wait for readiness properly

`/health` answers 200 **while the model is still loading** and real requests get
`503 Loading model`. Gate on a real completion:

```bash
until [ "$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' -d '{"prompt":"x","n_predict":1}')" = 200 ]; do sleep 2; done
```

## 3. Descriptor

`models/<name>.json` — the defaults are appropriate for llama.cpp:

```json
{
  "readout": "lettered",
  "letters": "abcdefghijklmnopqrstuvwxyz",
  "specific_token_scores": false
}
```

- `specific_token_scores` must stay false: llama.cpp has no exact-token-id scoring.
- The lettered readout uses the **chat surface** by default; `"lettered_surface":
  "native"` would keep the raw `/completion` prompt, which measured ~8x more
  label-driven flips on identical weights.
- `top_k_floor` is optional. Leave it out and the readout widens itself
  (20 → 512 → 2048 → 4096) until every label is visible; set it only to start
  wider and skip the narrower rungs.

## 4. Start pqnld and check what it thinks it can do

```bash
pqnld-rs --engine-kind llamacpp --vllm-url http://127.0.0.1:8080 \
  --model my-model --descriptor my-model --models-dir models \
  --host 127.0.0.1 --port 11560
curl -s localhost:11560/v1/capabilities | jq .
curl -s localhost:11560/healthz
```

`--engine-kind auto` also detects llama.cpp (it probes `/props`, which carries the
llama.cpp shape, before anything that could look OpenAI-compatible).

`/v1/capabilities` reports truthfully, for example:

```json
{"engine":"chat-completions","tokenize":true,"exact_ids":false,"top_k":true,
 "prompt_logprobs":false,"max_top_k":null,"read_width_start":20,
 "read_width_ladder":[20,512,2048,4096],"lettered_surface":"chat",
 "readout":"lettered","letters":26,
 "unsupported":{"exact_token_ids":true,"prompt_logprob_echo":true}}
```

## 5. Refusals you may see

- `answer-slot logprobs missing option labels: …` — no read width saw every label.
  The message names the widths tried. Almost always a model whose answer slot is
  not a letter (see the chat/native note above) rather than a width problem.
- More than 26 options: the echo path is required and llama.cpp has no prompt
  logprobs, so the question is refused by design.

## 6. What to expect

From the recorded arms: llama.cpp is stable question-to-question and across
repeats, but on identical weights the **raw prompt** produced label-driven
decisions (label flips 0.22, p95 spread 0.76) while the **chat surface** did not
(0.03, 0.08). Prefer the chat surface and check `/v1/capabilities` before trusting
an unchanged deployment.
