# Wire format

pqnld speaks the [Decision Index][di] shape: a `state` and a set of typed
`questions`, answered with one typed answer per question.

[di]: https://huggingface.co/spaces/multimodalart/jev-decision-index

## `POST /v1/decide`

Request:

```json
{
  "model": "qwen38-27b-nvfp4",
  "state": "any JSON value, or a string",
  "questions": {
    "<key>": {
      "type": "choice",
      "instructions": "Which day is the meeting?",
      "criteria": {"monday": "Monday", "tuesday": "Tuesday", "friday": "Friday"}
    },
    "<key2>": {
      "type": "noul",
      "instructions": "Is the meeting urgent?"
    }
  }
}
```

- `model` is echoed back in the response; it does not select anything server-side.
- `state` may be a string or any JSON value (`{}` and `[]` render as `(empty)`).
- `questions` is required and non-empty.
- Every question is an object with required `instructions` and a supported
  `type`. Instructions and criterion descriptions may be any JSON value; the
  existing renderer preserves strings directly and serializes structured values.
  Choice `criteria` is an object with 2–255 keys. A `null` description prints its
  key. The wire ceiling does not guarantee enough tokenizer labels.

The reliability repair refuses missing instructions and malformed criteria with
422 before probing/scoring. The initial string-only proposal was superseded by
Q-PQNLD-5: structured values used by the official corpus remain supported, and MCP
schema is aligned with that existing HTTP renderer. Structured JSON `state` and
valid answer envelopes remain supported.

Question types:

| type | options | answer |
|---|---|---|
| `choice` | 2–255 keys in `criteria` (a value of `null` prints the key itself) | `choice` = argmax key, `probabilities` = full distribution over the criteria keys |
| `noul` | none (binary) | `noul` = probability of "true", in `[0, 1]` |

Response `200`:

```json
{
  "model": "qwen38-27b-nvfp4",
  "answers": {
    "<key>": {
      "type": "choice",
      "choice": "tuesday",
      "probabilities": {"monday": 0.02, "tuesday": 0.95, "friday": 0.03}
    },
    "<key2>": {"type": "noul", "noul": 0.11}
  },
  "usage": {"input_tokens": 99}
}
```

Invariants (checked by the server before it replies, and rejected if broken):

- the answer keys equal the question keys;
- `choice` is one of the criteria keys, `probabilities` covers exactly the
  criteria keys, each in `[0, 1]`, summing to `1 ± 0.01`;
- `noul` is in `[0, 1]`.

### Errors

| Status | When |
|---|---|
| `400` | Malformed JSON body, or a missing `questions` field |
| `422` | Empty questions, non-object question, missing instructions, missing/non-object criteria, choice cardinality outside 2–255, unsupported type, incomplete answer-slot labels, or model capacity refusal (`Unsupported`). The message is preserved in `error` |
| `404` | Unknown path |
| `500` | Engine error, malformed/incomplete/misaligned echo evidence, non-finite scores, or failed readout self-check |

A rejected question is **never** guessed or truncated: capacity overflow is a
`422`, not a partial answer.

## `POST /v1/systemone`

Alias of `/v1/decide`, kept so a stock Decision Index kit HTTP engine can point
at pqnld unchanged. Identical request and response.

## `POST /v1/chat/completions`

An OpenAI-compatible shim so a router (e.g. LiteLLM) can register a pqnld
endpoint as a chat model.

- If the last user message is a JSON object with a `questions` key, it is treated
  as a Decision Index request and answered as a decision; the response
  `message.content` is the JSON-encoded decision.
- Otherwise the shim replies with a short notice explaining the expected format.
- Streaming (`"stream": true`) is supported: one content chunk and a final
  `finish_reason: "stop"` chunk with usage.

## `GET /healthz`

```json
{"status": "ok"}
```

Use it for readiness/liveness. pqnld does not probe the upstream on this path.
## Echo scoring limitation

Echo scoring refuses prefix-overlapping option keys (for example `item-1` and
`item-13`) with HTTP 422. This applies to explicit echo mode and the large-choice
fallback. The same keys remain usable on the lettered route. Unterminated
continuations describe overlapping events and must not be normalized as distinct
answers. Other echo scores remain length-sensitive, not calibrated beliefs.
