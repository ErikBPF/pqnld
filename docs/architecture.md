# Architecture

pqnld is a readout wrapper, not a model. It never generates text; it reads a
distribution the model already computed and reshapes it into a typed answer.

## The core idea

A decision is a **`max_tokens=1` chat request with `logprobs`**. If the options
are rendered with single-letter labels and the model is asked for exactly one
letter, then the top-logprob entries at the answer slot *are* the option scores.
Softmax them and you have the answer distribution.

```
render  ->  POST /v1/chat/completions (max_tokens=1, logprobs, top_logprobs>=N)
        ->  read top_logprobs at the answer slot
        ->  keep single-letter entries, map letter -> option key
        ->  require every label + softmax(temperature)
```

Consequences:

- **No autoregressive option generation** — options compete at one answer slot.
- **Multi-token option keys cost nothing** — the model is scored on the letter,
  not on the key's tokens.
- **No prompt-logprob memory** — nothing needs forced continuation scoring.
- **One prefill per question.**

### Why not score each option by echoing it?

An earlier design echoed the context once, then each `context + option_key`, and
summed the key-token log-probabilities. Accurate, but `n+1` full-context prefills
per question — and on a hybrid model whose unified block size exceeds the
decision prompt, nothing is prefix-cached, so every option re-prefills. The
lettered readout replaced it with one request.

With `specific_token_scores: true` and a compatible adapter, more than 26 options
can use an extended, tokenizer-verified single-token alphabet and explicit token
IDs, split above the tested 128-ID cap across `ceil(n/128)` requests. The shipped
descriptor does not enable this route. Without exact scoring, questions exceeding
the configured label alphabet fall back to echo; backend capacity still applies.

## The lettered readout

Steps:

1. Render `state`, the question instructions, and `letter) key-or-description`
   lines for each option, ending with "Reply with exactly one option letter."
2. Send one chat request with `max_tokens=1`, `logprobs=true`,
   `top_logprobs=max(len(keys), 20)`, `temperature=0`, and the descriptor's
   `enable_thinking` toggle.
3. Walk the answer slot's `top_logprobs`; keep the first entry per letter that is
   in `letters`, after trimming whitespace and lowercasing. Only a single ASCII
   letter qualifies; words and special-token strings are ignored.
4. Require every option label to be present. Missing labels raise `Unsupported`
   (HTTP 422); no scores are fabricated. Asking for at least N top tokens does
   not guarantee that all N labels appear.

Then `softmax(score / temperature)` gives the distribution, and `choice` is the
argmax. For `noul`, the keys are `false`/`true` and `noul` is
`P(true)` within the allowed-label readout. These normalized scores are not
automatically calibrated probabilities of correctness; calibration needs
held-out labeled data.

## The echo fallback

The echo fallback for engines without exact-token scoring (or a forced-echo
descriptor):

1. Echo the context (`echo=true, max_tokens=0, logprobs=1`) to get its tokens.
2. Echo each `context + key` in batches of `batch` (default 16) and sum the
   log-probabilities of the tokens after the shared prefix.
3. Softmax the summed scores.

The batch size bounds the prompt-logprob memory that can crash the engine.

## Startup probe

With `readout: auto`, pqnld asks the fixture question (options red/blue/green,
expected answer `b`):

- if the top-logprob token at the answer slot is a single letter in `letters`,
  keep `lettered`;
- otherwise switch to `echo` and require a `chat_template`;
- a descriptor that asks for `echo` with no `chat_template` is refused up front.

With `specific_token_scores: true`, unsupported or incomplete probe label coverage
is refused. With generic top-k scoring, that probe failure selects echo instead,
which still requires a chat template and usable score evidence.

The probe is best-effort: a transport error during the probe falls back to echo
rather than refusing to start. Point the server at a healthy engine (the
reference launcher health-checks the engine before starting pqnld).

## Concurrency

- The HTTP server handles requests concurrently.
- Question scoring and initial readout probes share a **global sidecar worker
  budget** (`--workers 1` by default). At one worker, each request keeps its stored
  question order; a larger budget permits concurrent question scoring across
  requests. Failed concurrent decisions cancel and drain remaining local tasks.
- This budget does not govern ordinary chat sent directly to the engine or unsend
  upstream requests already accepted. Engine batch numerics, MTP, restarts and
  cache state prevent a universal repeatability guarantee.
- The LRU cache is guarded by a lock.

Sequential also matches the engine's admission ramp: tiny one-token requests are
admitted a few at a time, so a multi-question decision costs about one scheduler
step per question (~0.12 s each).

## Cache

A bounded LRU (`cache_size`, default 256) keyed by
`model + readout mode + descriptor + temperature + state + questions`. Identical
repeated decisions are served from memory with no upstream call. Cache hits
return a shallow copy of the stored answer. The cache key preserves mapping
order because criteria order determines the rendered option-label assignment.

## Capacity ceiling

The binding limit is the model's context: all `context + option_key` prompts
(echo) or the rendered question (lettered) must fit. An over-capacity request is
refused with `422` and the **model's own capacity message is preserved** — pqnld
never truncates and never guesses. A 255-option `choice` is the stress case.

## Validation

Before replying, pqnld checks its own output: answer keys match question keys,
choice is inside the criteria, probabilities cover exactly the criteria and sum
to `1 ± 0.01`, `noul` is a probability. A failure is a `500`, not a wrong answer.

## Security posture

pqnld holds no credentials and binds `127.0.0.1` by default. If you expose it,
put auth or a firewall in front of it: an open decision endpoint is an open proxy
to your model.
# Experimental specific-token scoring

A descriptor may opt into `"specific_token_scores": true` for engines supporting
vLLM's `logprob_token_ids`, `return_tokens_as_token_ids`, and `/tokenize` APIs.
PQNLD resolves and caches each literal label's token ID, rejects multi-token or
duplicate mappings, and requests every label's score with `top_logprobs: 0`.
It neither constrains sampling nor applies logit bias. Returned IDs must cover
every label; uppercase and whitespace variants are not pooled into these events.

This remains opt-in: Apollo vLLM 0.30.0 passed isolated tests but returned HTTP 500
for decisions during simultaneous ordinary chat. See [the one-pager](readout-one-pager.md).
Do not enable it for shared-chat production until that engine failure is fixed
and mixed-load tests pass. Label probabilities still require held-out calibration.
