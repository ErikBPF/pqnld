# Engine capabilities for the pqnld readout

**Status:** the engine abstraction is implemented as the `Engine` trait in
[`sidecar-rs/src/engine.rs`](../sidecar-rs/src/engine.rs); the vLLM adapter is
exercised live, the others are compile- and parse-unit-tested. Backend API
research below is not a promise of live adapter compatibility.
**Owner / date:** PQNLD / 2026-10-01 UTC.

## Method and evidence

- **vLLM 0.30.0 (Apollo, `qwen38-27b-nvfp4`)** was read live, read-only: the
  running server's `GET http://100.77.14.27:11542/openapi.json` and the installed
  package under `apollo:/usr/local/lib/python3.12/dist-packages/vllm/`. The
  container command carries no `--max-logprobs`, so the default 20 applies.
  Public reference: <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>.
- **llama.cpp** `tools/server/README.md` and `tools/server/server-common.cpp`,
  `server-task.cpp` on `master` (fetched 2026-10-01).
- **SGLang** docs (`basic_usage/native_api`, `basic_usage/sampling_params`) and
  `python/sglang/srt/entrypoints/openai/protocol.py`,
  `python/sglang/srt/managers/io_struct.py`, `http_server.py` on `main`.
- **OpenAI** API reference pages for chat and legacy completions.

## Capability matrix

| | vLLM (OpenAI server) | llama.cpp (`llama-server`) | SGLang server | OpenAI API |
|---|---|---|---|---|
| **(a) token logprobs?** | yes | yes | yes | yes, on supported models |
| **(b) exact token IDs?** | **yes** `logprob_token_ids` | no — top-k only | **yes**, but native only (`/generate` `token_ids_logprob`, `/v1/score` `label_token_ids`); OpenAI chat has no per-id field | no |
| **(c) tokenize endpoint?** | yes `/tokenize` (+ `/detokenize`) | yes `/tokenize` (+ `/detokenize`) | yes `/tokenize` (+ `/detokenize`) | **no** (client-side `tiktoken` only) |
| **(d) answer-slot surface** | `/v1/chat/completions` `choices[0].logprobs.content[0]`; echo via `/v1/completions` `echo:true` | `/v1/chat/completions` `choices[0].logprobs.content[]`; native `/completion` `completion_probabilities` | OpenAI: `choices[0].logprobs.content[]`; native `/generate` `meta_info.output_token_logprobs` | chat: `choices[0].logprobs.content[]` (+ `refusal[]`); echo via legacy `/v1/completions` `echo:true` |
| **(e) limits** | explicit ids **128**; top-k **20** (default, `--max-logprobs`; `-1` = vocab) | no documented top-k cap; `top_logprobs` defaults to **20** when `logprobs:true` | no documented top-k cap on native; `token_ids_logprob` is a list per item | chat `top_logprobs` **0–20**; legacy `logprobs` **max 5** |

### (e) exact request/response fields

**vLLM 0.30.0** (verified in the live OpenAPI and installed source):

- Request (`ChatCompletionRequest`): `logprobs: bool=true`,
  `top_logprobs: int` (default 0), `logprob_token_ids: [int]`,
  `return_tokens_as_token_ids: bool`.
  - `protocol.py:288` defines `logprob_token_ids`; `protocol.py:408` defines
    `return_tokens_as_token_ids`; `protocol.py:821` requires `logprobs:true`
    with `logprob_token_ids`.
  - `protocol.py:738` sets `SamplingParams.logprobs=None` whenever
    `logprob_token_ids` is set — the root cause documented in
    [mixed-readout-diagnosis.md](mixed-readout-diagnosis.md).
- `/tokenize` request: `{model, prompt, add_special_tokens, return_token_strs}`;
  response: `{count, max_model_len, tokens:[int], token_strs?}`.
- Limits: `MAX_LOGPROB_TOKEN_IDS = 128` (`sampling_params.py:30`, enforced at
  `sampling_params.py:858`); out-of-vocab ids rejected; `max_logprobs` default
  **20** (`config/model.py:254`), `-1` = vocab.
- Exact-ID response: `choices[0].logprobs.content[0]` carries the sampled token
  plus a `top_logprobs` array; with `return_tokens_as_token_ids` every token is
  the string `"token_id:<n>"`, so the readout matches ids exactly.
- Echo: `/v1/completions` with `echo:true, max_tokens:0, logprobs:1` returns
  `choices[0].logprobs.{tokens, token_logprobs, top_logprobs}` for the prompt.
- Also present: `/generative_scoring`, `/v1/responses`, `/detokenize`.

**llama.cpp** `master`:

- OpenAI endpoints accept `logprobs: bool` and `top_logprobs: int`; the server
  maps them to the native `n_probs` (`server-common.cpp:1431-1437`):
  ```
  if logprobs: n_probs = top_logprobs (default 20)
  ```
  `top_logprobs` without `logprobs:true` is a 400.
- Native `/completion` `n_probs` returns `completion_probabilities` — an array
  per generated token, each item `{id, token, bytes, logprob, top_logprobs:[...]}`
  (README "Response format").
- OpenAI response: `choices[0].logprobs.content[]` with the same per-token
  `{token, bytes, logprob, top_logprobs}` shape (`server-task.cpp:434`).
- `/tokenize` request `{content, add_special, parse_special, with_pieces}` →
  `{"tokens":[ids]}` or `{id,piece}` objects; `/detokenize` exists.
- **No exact-ID scoring.** `logit_bias` only biases logits; it does not return
  chosen-id scores. No `echo` on the native endpoint, so the vLLM prompt-logprob
  echo trick has no direct equivalent here — top-k is the usable path.
- No documented maximum on `n_probs`.

**SGLang** `main`:

- OpenAI chat (`protocol.py:868-869`): `logprobs: bool`, `top_logprobs: int`,
  plus `sampling_logprobs_mode`. No per-id field. Response uses
  `ChoiceLogprobs.content: [ChatCompletionTokenLogprob]` (same token/logprob/
  `top_logprobs` shape as OpenAI).
- Native `/generate` (`io_struct.py:229-236`): `return_logprob`,
  `logprob_start_len` (`-1` = output only, `0` = include prompt),
  `top_logprobs_num`, and **`token_ids_logprob: [int]`** (exact ids). Response
  `meta_info` exposes `input_token_logprobs`, `output_token_logprobs`,
  `input_top_logprobs`, `output_top_logprobs`, and the paired
  `input_token_ids_logprobs` / `output_token_ids_logprobs` (val + idx).
- **`/v1/score`** (`protocol.py:1401-1413`): `query`, `items`,
  `label_token_ids: [int]|[int][]`, `apply_softmax`, `temperature`,
  `return_token_logprobs`, `item_first`, `score_extraction_token`. Response
  `scores` is one probability list per item, in `label_token_ids` order — the
  cleanest exact-ID surface of any engine.
- `/v1/decisions`: native typed `choice`/`score`/`yes_no` decisions scored via
  the `/v1/score` path (a full readout already implemented server-side).
- `/tokenize` and `/detokenize`: yes (native API doc).

**OpenAI API** (see caveat: parameter support is model-dependent):

- Chat: `logprobs: bool`, `top_logprobs: 0–20`. Response
  `choices[0].logprobs.content: [ChatCompletionTokenLogprob]` with
  `{token, bytes, logprob, top_logprobs:[{token,bytes,logprob}]}` and a parallel
  `refusal[]`. A token outside the top 20 gets `logprob = -9999.0`.
- Legacy `/v1/completions`: `logprobs: 0–5`, `echo: true` returns
  `{text_offset, token_logprobs, tokens, top_logprobs}` for the prompt — the one
  echo-compatible surface.
- No `/tokenize`; token ids exist only client-side (`tiktoken`) and are
  model-specific. No exact-ID scoring.

## Current sidecar routing

The [readout implementation](../sidecar-rs/src/main.rs) supports `lettered`,
`echo` and `auto`. Adapter capabilities are declared in code, not verified by
universal capability autodetection. `--engine-kind auto` performs best-effort
server-kind detection; readout `auto` probes lettered suitability and can select
echo. Neither enables descriptor `specific_token_scores` automatically.

| Route | Requirements and behavior |
|---|---|
| Lettered exact IDs | Explicit `specific_token_scores: true` and compatible adapter; distinct single-token labels; complete returned label scores or refusal |
| Lettered top-k | Exact opt-in inactive/unavailable; accepts only label-shaped tokens and requires every label returned, or refuses |
| Echo | Selected echo mode, or questions exceeding the configured letter alphabet without exact opt-in; requires chat template and prompt-logprob support |
| Parse | Unsupported; no point-answer downgrade or fabricated probability fallback |

Exact scoring uses **128-ID chunks of the same prompt**, not an arbitrary
per-adapter chunk size. Merging assumes absolute comparable full-vocabulary
scores at the same conditioned slot. The wire allows **255 criteria keys**;
the reference tokenizer yielded **206 usable single-token labels**, not any N.
Exact scoring is opt-in, absent from the shipped descriptor, and retains the
[mixed-MTP serving gate](mixed-readout-diagnosis.md).

Echo requires aligned, non-empty usable evidence, with finite scored tokens
(the first context token may be unscored) and non-empty scored continuations.
Prefix-overlapping keys are refused; large prompts retain capacity/memory risk.
Normalization requires finite positive temperature; normalized scores are not
established calibrated correctness probabilities.

The actual asynchronous `Engine` signatures, adapters and response parsers live
in [engine.rs](../sidecar-rs/src/engine.rs).

## Per-engine request/response shapes

These are illustrative researched API shapes. Current adapters use llama.cpp's
native `/completion` and SGLang's `/generate`, not every surface listed below.
Non-vLLM shapes remain unverified live against the sidecar.

### vLLM — exact IDs (opt-in)

```json
POST /v1/chat/completions
{"model":"qwen38-27b-nvfp4",
 "messages":[{"role":"system","content":"…"},{"role":"user","content":"…"}],
 "max_tokens":1,"temperature":0,"logprobs":true,"top_logprobs":0,
 "logprob_token_ids":[64,65,66],"return_tokens_as_token_ids":true}

200 ->
{"choices":[{"logprobs":{"content":[{
  "token":"token_id:66","logprob":-0.01,
  "top_logprobs":[
    {"token":"token_id:66","logprob":-0.01,"bytes":[…]},
    {"token":"token_id:64","logprob":-3.2,"bytes":[…]},
    {"token":"token_id:65","logprob":-4.0,"bytes":[…]}]}]}}]}
```

### vLLM — generic top-k / echo

```json
POST /v1/chat/completions
{"…","max_tokens":1,"temperature":0,"logprobs":true,"top_logprobs":20}
{"choices":[{"logprobs":{"content":[{"token":"b","top_logprobs":[{"token":"b","logprob":-0.1},…]}]}}]}

POST /v1/completions
{"model":"qwen38-27b-nvfp4","prompt":"…","echo":true,"max_tokens":0,"temperature":0,"logprobs":1}
{"choices":[{"logprobs":{"tokens":["…"],"token_logprobs":[…],"top_logprobs":[…]}}]}
```

The researched server's default top-k cap is 20. Requests above the configured
cap fail; even an accepted request must return every label for sidecar success.

### llama.cpp — top-k only

```json
POST /v1/chat/completions
{"model":"…","messages":[…],"max_tokens":1,"temperature":0,
 "logprobs":true,"top_logprobs":32}
{"choices":[{"logprobs":{"content":[{
  "id":66,"token":"b","bytes":[…],"logprob":-0.1,
  "top_logprobs":[{"id":66,"token":"b","bytes":[…],"logprob":-0.1},…]}]}}]}

POST /tokenize {"content":"b","add_special":false} -> {"tokens":[66]}
```

No `logprob_token_ids`; `n_probs`/`top_logprobs` is the ceiling. A large top-k is
the only way to approximate coverage, and it is still not guaranteed.

### SGLang — exact IDs via native `/generate` or `/v1/score`

```json
POST /generate
{"text":"…","return_logprob":true,"logprob_start_len":0,
 "sampling_params":{"max_new_tokens":1,"temperature":0,
                    "top_logprobs_num":0,"token_ids_logprob":[64,65,66]}}
-> meta_info.output_token_logprobs / output_token_ids_logprobs

POST /v1/score
{"model":"…","query":"…","items":["…"],
 "label_token_ids":[64,65,66],"apply_softmax":true}
-> {"scores":[[p64,p65,p66]], "usage":{…}}
```

SGLang OpenAI chat is top-k only:

```json
POST /v1/chat/completions
{"model":"…","messages":[…],"max_tokens":1,"temperature":0,
 "logprobs":true,"top_logprobs":32}
```

### OpenAI — top-k chat and legacy echo

```json
POST /v1/chat/completions
{"model":"gpt-4o","messages":[…],"logprobs":true,"top_logprobs":20,"max_tokens":1}
{"choices":[{"logprobs":{"content":[{"token":"b","logprob":-0.1,"bytes":[…],
  "top_logprobs":[{"token":"b","logprob":-0.1,"bytes":[…]},…]}],"refusal":null}}]}

POST /v1/completions
{"model":"gpt-3.5-turbo-instruct","prompt":"…","echo":true,
 "logprobs":5,"max_tokens":0,"temperature":0}
{"choices":[{"logprobs":{"tokens":[…],"token_logprobs":[…],
  "top_logprobs":[…],"text_offset":[…]}}]}
```

OpenAI cannot do exact IDs and has no `/tokenize`. Exact scoring is unavailable;
use top-k (chat) or echo (legacy completions, `logprobs<=5`).

## Mapping summary

| Researched engine API | Score surface | Exact-ID field | Tokenize | Echo/prompt-logprobs |
|---|---|---|---|---|
| vLLM | exact IDs / top-k | `logprob_token_ids` (+`return_tokens_as_token_ids`) | `/tokenize` | `/v1/completions` `echo:true` |
| llama.cpp | top-k | — | `/tokenize` | native endpoint has no echo |
| SGLang | exact IDs / top-k | `/generate` `token_ids_logprob`; `/v1/score` `label_token_ids` | `/tokenize` | `/generate` `logprob_start_len:0` |
| OpenAI | top-k | — | none | legacy `/v1/completions` `echo:true`, `logprobs<=5` |

Unresolved / to verify before claiming live adapter support: SGLang's OpenAI
chat coverage for top-k (`top_logprobs`) under reasoning models; whether OpenAI's
newer reasoning models accept `logprobs`/`top_logprobs` at all; llama.cpp `n_probs`
behaviour at very large k. The vLLM explicit-ID defect under MTP remains the
blocking gate for exact-ID shared serving on Apollo; see
[mixed-readout-diagnosis.md](mixed-readout-diagnosis.md).
Do not build a server-side SGLang `/v1/decisions` fast path yet — it is a
different conditioning path and needs its own correctness evidence.
