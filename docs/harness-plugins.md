# Harness integration: native MCP stdio

`pqnld-rs --mcp` exposes one `decide` tool using the same readout, validation,
cache and probe as HTTP decisions. It talks to the configured engine directly;
no separate HTTP sidecar is required. The harness's reasoning model may differ
from the served readout model. These snippets are configuration guidance, not
evidence of tested client integrations.

## Contract and errors

Transport is newline-delimited JSON-RPC 2.0 on stdin/stdout; diagnostics belong
on stderr. The implemented protocol version is `2025-06-18`, with `initialize`,
`ping`, `tools/list` and `tools/call`; notifications receive no response.
See [native MCP source and schema](../sidecar-rs/src/mcp.rs).

`decide` arguments require `state` (any JSON value) and non-empty `questions`:
- Each question is an object with supported `type` and required `instructions`.
- `choice` requires a `criteria` object with **2–255 keys**; `noul` is binary.
- Instructions and criterion descriptions accept **any JSON value**, including
  null. Required fields, object shape, supported types and cardinality still apply.

Success returns the [wire response](wire.md) in `structuredContent` and the same
JSON in text `content`, with `isError: false`. Answer keys match question keys;
choice probabilities cover all criteria. Evidence must be complete or the call
fails: no guessed, truncated, or parsed-choice answer fallback exists.

| Failure | Actual native response |
|---|---|
| Malformed JSON line | JSON-RPC `-32700` |
| Unknown method | JSON-RPC `-32601` |
| Unknown tool, missing state, or missing/non-object/empty questions | JSON-RPC `-32602` |
| Unsupported question, missing labels, or recognized capacity refusal | `isError: true`; text JSON error with `code: "unsupported"`, `status: 422`, message |
| Engine failure, unusable echo evidence, or failed output self-check | `isError: true`; text JSON error with `code: "engine_error"`, `status: 500`, message |

## OpenCode V2

Use project `opencode.json(c)` or global `~/.config/opencode/opencode.json(c)`.
Native V2 configuration nests local servers under `mcp.servers`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "servers": {
      "pqnld": {
        "type": "local",
        "command": [
          "/abs/path/to/pqnld-rs",
          "--mcp",
          "--vllm-url", "http://127.0.0.1:11542",
          "--model", "qwen38-27b-nvfp4",
          "--descriptor", "qwen38-27b-nvfp4",
          "--models-dir", "/abs/path/to/pqnld/sidecar-rs/models"
        ],
        "timeout": { "execution": 600000 }
      }
    }
  }
}
```

Local servers auto-connect; no `enabled` setting is needed. `timeout.execution`
is **milliseconds for tool execution**, not startup. V2 shape comes from
[MCP documentation](https://opencode.ai/v2/docs/mcp-servers) and
[config documentation](https://opencode.ai/v2/docs/config), not the schema URL.

## OpenAI Codex

Use `~/.codex/config.toml` or trusted project `.codex/config.toml`:

```toml
[mcp_servers.pqnld]
command = "/abs/path/to/pqnld-rs"
args = [
  "--mcp",
  "--vllm-url", "http://127.0.0.1:11542",
  "--model", "qwen38-27b-nvfp4",
  "--descriptor", "qwen38-27b-nvfp4",
  "--models-dir", "/abs/path/to/pqnld/sidecar-rs/models",
]
startup_timeout_sec = 10
tool_timeout_sec = 600
enabled = true
```

Codex timeouts are **seconds**. Size tool execution timeout for the whole
decision, not one engine request: questions are sequential by default and the
sidecar's engine timeout is 600 seconds per request. See
[Codex MCP](https://developers.openai.com/codex/mcp/) and
[config reference](https://developers.openai.com/codex/config-reference).

## Backend and operating limits

Descriptor `specific_token_scores: true` opts into exact-ID scoring on a
compatible backend; neither readout `auto` nor engine-kind detection enables it.
Otherwise lettered top-k needs every option label returned, or refuses. Echo
requires a chat template and usable prompt/continuation scores; `parse` is not
supported. Tokenizer/backend limits still apply: **255 wire keys, 206 reference
labels, 128-ID exact-scoring chunks**. See [descriptors](descriptor.md) and
[engine capabilities](engine-capabilities.md).

Only vLLM has live adapter evidence. Shared chat/decision serving with mandatory
MTP remains [gated](mixed-readout-diagnosis.md). Probabilities are normalized
readout scores, not established calibrated correctness probabilities. For flags,
global worker admission, cancellation and UDS safety, use the
[runtime guide](../sidecar-rs/README.md).
