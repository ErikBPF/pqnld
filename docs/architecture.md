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
        ->  floor + softmax(temperature)
```

Consequences:

- **Options are independent** — no autoregressive coupling between choices.
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

The echo path survives only where letters cannot: **more than 26 options**.

## The lettered readout

`_letter_scores(state, question, keys)`:

1. Render `state`, the question instructions, and `letter) key-or-description`
   lines for each option, ending with "Reply with exactly one option letter."
2. Send one chat request with `max_tokens=1`, `logprobs=true`,
   `top_logprobs=max(len(keys), 20)`, `temperature=0`, and the descriptor's
   `enable_thinking` toggle.
3. Walk the answer slot's `top_logprobs`; keep the first entry per letter that is
   in `letters`; ignore non-letter tokens.
4. Letters the model did not rank get a floor (`min_ranked - 5.0`), so an
   option is never literally zero.

Then `softmax(score / temperature)` gives the distribution, and `choice` is the
argmax. For `noul`, the keys are `false`/`true` and `noul` is
`P(true)`.

## The echo fallback

`_echo_scores(context, keys)` for >26 options (or a forced-echo descriptor):

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

The probe is best-effort: a transport error during the probe falls back to echo
rather than refusing to start. Point the server at a healthy engine (the
reference launcher health-checks the engine before starting pqnld).

## Concurrency

- The HTTP server is threaded (`ThreadingHTTPServer`), so requests are handled
  concurrently.
- Questions inside one decision are read **in parallel** with a
  `ThreadPoolExecutor`, bounded by `MAX_QUESTION_WORKERS` (8), then gathered in
  submission order so answer order is stable.
- The LRU cache is guarded by a lock.

Note the engine's own behavior: tiny one-token requests are admitted to the
batch in a ramp, so a two-question decision can cost about two scheduler steps
(~2 × 0.12 s) even when the two reads are issued in parallel. The parallelism
still helps when the engine has spare admission slots, and it prevents the host
side from being the bottleneck.

## Cache

A bounded LRU (`cache_size`, default 256) keyed by
`model + readout mode + descriptor + temperature + state + questions`. Identical
repeated decisions are served from memory with no upstream call. Cache hits
return a shallow copy of the stored answer.

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
