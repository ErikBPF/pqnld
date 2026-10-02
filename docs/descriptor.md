# Model descriptors

A descriptor tells pqnld how to talk to a served model. It is a small JSON file,
`sidecar-rs/models/<name>.json`, selected with `--descriptor NAME` (or `--model
NAME` when no descriptor is given). Lookup order:

1. `--models-dir` (or `PQNLD_MODELS_DIR`), if set (e.g. `sidecar-rs/models`);
2. otherwise the `models/` directory beside the binary.

A missing file is not an error: pqnld falls back to the built-in defaults,
which reproduce the reference Qwen3.8 behaviour.

Unknown keys in the file are ignored, so a descriptor can carry extra notes
without breaking.

## Fields

| Field | Default | Meaning |
|---|---|---|
| `readout` | `"auto"` | `"auto"`, `"lettered"`, or `"echo"` (see below) |
| `letters` | `"abcdefghijklmnopqrstuvwxyz"` | The single-token labels used for the lettered readout. Limit is 26 by default; a model with single-token labels in another alphabet can supply them here |
| `temperature` | `1.0` | Softmax temperature applied to the letter scores |
| `enable_thinking` | `false` | Passed as `chat_template_kwargs.enable_thinking` on the lettered request |
| `system_prompt` | a decision-engine prompt | System message for the echo fallback context |
| `letter_system_prompt` | a decision-engine prompt | System message for the lettered request |
| `chat_template` | Qwen3.8 ChatML with an empty thinking block | Used by the echo fallback to build the raw context. `{system}` and `{body}` are substituted. Set to `null` only if you will not use echo |
| `engine_profile` | `"full"` | A hint for launchers: `"full"` (spec decoding + KV connector) or `"lean"`. pqnld itself does not read it; the reference `serve-litellm.sh` does |

## Readout modes

- **`lettered`** — render options with single-letter labels and read the
  answer-slot distribution over those letters. One request per question, options
  independent. Preferred; needs a model whose answer slot is a letter.
- **`echo`** — echo the context and each `context + key`, summing the key-token
  log-probabilities. Used as the fallback when the lettered path does not fit the
  model, and for engines without exact-token scoring (questions with more than 26
  options use the extended single-token alphabet when exact IDs are available).
- **`auto`** — probe at startup:
  1. Send a fixture choice (`red`/`blue`/`green`, expected `b`).
  2. Keep `lettered` **only if** the highest-logprob token at the answer slot is
     a single letter in `letters`.
  3. Otherwise switch to `echo` (and require a `chat_template`).

A descriptor that declares `readout: "echo"` while `chat_template` is `null` is
**refused** at construction (`Unsupported`): there would be no way to build the
context.

## Example

```json
{
  "readout": "auto",
  "temperature": 1.0,
  "enable_thinking": false,
  "chat_template": "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n thinking\n\n</think>\n\n",
  "engine_profile": "full"
}
```

## Adding one

1. Copy `sidecar-rs/models/qwen38-27b-nvfp4.json`.
2. Set the chat template for your model exactly (a wrong template silently
   degrades the echo path; the lettered path is template-free because it uses the
   model's own chat endpoint).
3. Serve it: `pqnld-rs --model <served-name> --descriptor <name>
   --models-dir sidecar-rs/models`.
4. Verify with a `curl` to `/v1/decide` (or the MCP `decide` tool), and note the
   engine revision and flags you tested.
