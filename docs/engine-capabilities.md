# Engine capabilities for the pqnld readout

**Status:** the engine abstraction is implemented as the `Engine` trait in
`sidecar-rs/src/engine.rs`; the vLLM adapter is exercised live, the others are
compile- and parse-unit-tested. No engine code changed.
**Owner / date:** PQNLD / 2026-10-01 UTC.
**Question:** which inference servers can give pqnld the answer-slot label
scores it needs, and what is the minimal client abstraction that drives all of
them?

> "openpai" is read as **OpenAI, the API provider**. If OpenPAI (Microsoft's
> cluster orchestrator) was intended, it is a deployment control plane with no
> inference or logprob surface and is out of scope for a readout.

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

## Degradation tiers

The readout picks the highest tier the endpoint advertises. Tiers 0–2 produce a
probability distribution; tier 3 does not.

| Tier | Needs | Guarantees | >26 options |
|---|---|---|---|
| **0 — exact IDs** | `tokenize` + `score_token_ids` | complete coverage by construction; labels are tokenizer-verified single distinct tokens; request chunked by `max_explicit_ids` (128) | yes (extended single-token labels, `ceil(n/128)` requests merged) |
| **1 — generic top-k** | `top_k` | conditional distribution only over labels that land in the top-k; a missing label is a hard refusal (HTTP 422), never fabricated | unreliable; labels can miss the top-k |
| **2 — echo / prompt logprobs** | `prompt_logprobs` + chat template | summed continuation logprobs; refuses prefix-overlapping keys | yes (this is the existing >26 fallback) |
| **3 — constrained parse** | chat text only | **no distribution**: ask for one label, parse the reply, report a point answer (degenerate mass) | no |
| **— none** | — | refuse (422); no probabilities, no guesses | no |

Consequences of the tiers that already hold in the code: tier 0 is the only one
that turns "ask for N labels" into a guaranteed complete score set; tier 1
cannot guarantee coverage and must refuse when a label is absent; tier 3 means
"typed answer without calibrated probabilities" and should be opt-in, not a
silent fallback.

## Minimal abstraction

Implemented as the `Engine` trait in `sidecar-rs/src/engine.rs`; the signatures
below mirror it:

```rust
/// Where a server exposes answer-slot logprobs.
enum Surface { ChatCompletions, Completions, NativeGenerate, Score }

/// One capability probe per engine, done once at startup.
struct Capabilities {
    tokenize: bool,           // text -> token ids endpoint exists
    exact_ids: bool,          // caller-supplied token-id set can be scored
    top_k: bool,              // engine top-k by text can be scored
    prompt_logprobs: bool,    // prompt/continuation tokens (echo) can be scored
    max_top_k: Option<usize>, // None = server-defined / unbounded
    max_explicit_ids: Option<usize>,
    surface: Surface,         // where the answer-slot request is sent
}

/// One answer-slot query. `chat=true` -> render system+body on the chat
/// surface; `chat=false` -> `body` is already a raw prompt (echo).
struct Query<'a> {
    system: &'a str,
    body: &'a str,
    chat: bool,
}

trait Engine: Send + Sync {
    fn name(&self) -> &'static str;
    fn capabilities(&self) -> Capabilities;
    fn tokenize(&self, text: &str) -> Option<Vec<u32>>;
    fn score_token_ids(&self, q: &Query, ids: &[u32]) -> Option<Vec<(u32, f64)>>;
    fn topk(&self, q: &Query, k: usize) -> Option<Vec<(String, f64)>>;
    fn prompt_logprobs(&self, raw_prompt: &str) -> Option<Vec<(String, f64)>>;
}
```

Readout selection, in order:

1. **Lettered, exact IDs available** (`exact_ids && tokenize`): resolve every
   label with `tokenize`; require one distinct id each; call `score_token_ids`
   in chunks of `max_explicit_ids.unwrap_or(128)`; merge by id.
2. **Lettered, top-k available**: call `topk(q, max(keys, 20))`; keep entries
   whose text is a single label; require complete coverage or refuse.
3. **>26 options**: exact IDs first (extended single-token labels); else echo
   via `prompt_logprobs`; else refuse.
4. **Echo descriptor**: `prompt_logprobs` or refuse.
5. **No logprobs at all**: constrained single-label parse (tier 3), explicit and
   distribution-free — or refuse, per descriptor policy.

This is the same routing the canonical code performs; the trait lifts the
vLLM-specific JSON out of `sidecar-rs/src/main.rs` into
`sidecar-rs/src/engine.rs`:

- `label_id` (`/tokenize`) → `Engine::tokenize` (engine.rs:269).
- `letter_scores` `logprob_token_ids` / `return_tokens_as_token_ids`
  → `Engine::score_token_ids` (engine.rs:281).
- `letter_scores` `top_logprobs` branch → `Engine::topk` (engine.rs:300).
- `echo` `/v1/completions` echo+logprobs → `Engine::prompt_logprobs`
  (engine.rs:308).

## Per-engine request/response shapes

### vLLM — exact IDs (tier 0, canonical)

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

### vLLM — generic top-k (tier 1) / echo (tier 2)

```json
POST /v1/chat/completions
{"…","max_tokens":1,"temperature":0,"logprobs":true,"top_logprobs":26}
{"choices":[{"logprobs":{"content":[{"token":"b","top_logprobs":[{"token":"b","logprob":-0.1},…]}]}}]}

POST /v1/completions
{"model":"qwen38-27b-nvfp4","prompt":"…","echo":true,"max_tokens":0,"temperature":0,"logprobs":1}
{"choices":[{"logprobs":{"tokens":["…"],"token_logprobs":[…],"top_logprobs":[…]}}]}
```

### llama.cpp — top-k only (tiers 1/2-via-top-k)

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

### OpenAI — top-k chat (tier 1) and legacy echo (tier 2)

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

OpenAI cannot do exact IDs and has no `/tokenize`. Tier 0 is unavailable; use
top-k (chat) or echo (legacy completions, `logprobs<=5`).

## Mapping summary

| Engine | Best tier | Exact-ID field | Tokenize | Echo/prompt-logprobs |
|---|---|---|---|---|
| vLLM | 0 | `logprob_token_ids` (+`return_tokens_as_token_ids`) | `/tokenize` | `/v1/completions` `echo:true` |
| llama.cpp | 1 | — | `/tokenize` | native endpoint has no echo |
| SGLang | 0 | `/generate` `token_ids_logprob`; `/v1/score` `label_token_ids` | `/tokenize` | `/generate` `logprob_start_len:0` |
| OpenAI | 1 | — | none | legacy `/v1/completions` `echo:true`, `logprobs<=5` |

Unresolved / to verify before implementing: SGLang's OpenAI chat coverage for
top-k (`top_logprobs`) under reasoning models; whether OpenAI's newer reasoning
models accept `logprobs`/`top_logprobs` at all; llama.cpp `n_probs` behaviour at
very large k; and the vLLM explicit-ID defect under MTP remains the blocking
gate for tier 0 on Apollo (see [mixed-readout-diagnosis.md](mixed-readout-diagnosis.md)).
Do not build a server-side SGLang `/v1/decisions` fast path yet — it is a
different conditioning path and needs its own correctness evidence.
