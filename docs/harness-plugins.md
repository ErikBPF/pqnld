# Harness plugins: pqnld as a decision tool for any code harness

**Stage / revision:** design / 1\
**Status:** proposed; no harness code yet\
**Owner / date:** PQNLD / 2026-10-01 UTC

Goal: let a code harness (opencode, OpenAI Codex, or any MCP client) turn the
model it is already configured with into a typed decision engine, without
reimplementing the readout per harness. The readout core is unchanged; the
portable artifact is a **client-side MCP stdio server** exposing one `decide`
tool that speaks the same [Decision Index contract](wire.md) as `POST /v1/decide`.

## 1. Integration points (researched)

### opencode

- **MCP servers** are declared under the `mcp` key of the opencode config; local
  servers use `"type": "local"` with a `command` array plus optional
  `environment`, `cwd`, `enabled`, and `timeout`. MCP tools are then "available
  to the LLM alongside built-in tools"; the user references a server by name in a
  prompt (`use the <name> tool`). Servers can be enabled/disabled globally or
  per agent, with glob patterns. — <https://opencode.ai/docs/mcp-servers/>
- **Config locations and precedence** (lowest to highest): remote org config
  (`.well-known/opencode`), global `~/.config/opencode/opencode.json`, custom
  `OPENCODE_CONFIG`, project `opencode.json`, `OPENCODE_CONFIG_DIR` (overrides
  agent/command/mode/plugin files), managed system config, inline
  `OPENCODE_CONFIG_CONTENT`. Project config overrides global. — <https://opencode.ai/docs/config/>
- **Native plugins** are JS/TS modules dropped in `~/.config/opencode/plugins/`
  (global) or `.opencode/plugins/` (project), or npm packages listed in the
  `plugin` config array. A plugin's `tool` helper adds a **custom tool** that
  opencode exposes to the model next to built-in tools. — <https://opencode.ai/docs/plugins/>
- Tool exposure: MCP tools and plugin custom tools are **model-invoked**; the
  user drives them by prompting and sees/approves them. opencode has no
  in-config "user-only" tool surface.

### OpenAI Codex

- Codex stores MCP config in `config.toml`: user-level `~/.codex/config.toml`,
  or project-scoped `.codex/config.toml` (loaded only for trusted projects). The
  desktop app, CLI, and IDE extension share this config. — <https://developers.openai.com/codex/mcp/>
- Local (stdio) servers are `[mcp_servers.<name>]` tables with `command`
  (required), `args`, `env`, `env_vars`, `cwd`, plus `startup_timeout_sec`
  (default 10), `tool_timeout_sec` (default 60), `enabled`, `required`,
  `enabled_tools`/`disabled_tools`, and per-tool `approval_mode`. SSE is not
  supported for local servers; stdio is the local transport. `codex mcp add
  <name> -- <command>` writes the table; `/mcp` in a session lists connected
  servers. — <https://developers.openai.com/codex/mcp/> and
  <https://developers.openai.com/codex/config-reference>
- Codex has no native tool-plugin API analogous to opencode's `tool` helper:
  **MCP is the only extension path** for adding a `decide` tool.

| | opencode | Codex |
|---|---|---|
| MCP stdio | yes (`mcp`, `type: local`) | yes (`[mcp_servers.*]`) |
| Config file | `opencode.json` / `opencode.jsonc` | `config.toml` |
| Global path | `~/.config/opencode/opencode.json` | `~/.codex/config.toml` |
| Project path | `opencode.json` (untrusted OK) | `.codex/config.toml` (trusted projects only) |
| Native tool plugin | yes (`tool` helper) | no (MCP only) |
| Tool exposure | model-invoked | model-invoked |

## 2. Portable shape: an MCP stdio `decide` server

Transport is the MCP **stdio** transport: newline-delimited JSON-RPC 2.0
requests on stdin, one JSON-RPC response per line on stdout, diagnostics on
stderr. Nothing else may be written to stdout.

### 2.1 Handshake (2025-06-18 baseline)

The compatibility baseline is the `initialize` handshake used by the 2025-06-18
spec (structured tool output, which `decide` wants). — <https://modelcontextprotocol.io/specification/2025-06-18/server/tools>

```
client -> {"jsonrpc":"2.0","id":1,"method":"initialize","params":{
            "protocolVersion":"2025-06-18",
            "capabilities":{},
            "clientInfo":{"name":"opencode","version":"..."}}}
server -> {"jsonrpc":"2.0","id":1,"result":{
            "protocolVersion":"2025-06-18",
            "capabilities":{"tools":{"listChanged":false}},
            "serverInfo":{"name":"pqnld","version":"0.1.0"},
            "instructions":"..."}}

client -> {"jsonrpc":"2.0","method":"notifications/initialized"}

client -> {"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}
server -> {"jsonrpc":"2.0","id":2,"result":{"tools":[ <decide tool> ]}}

client -> {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
            "name":"decide","arguments":{"state":<value>,"questions":{...}}}}
server -> {"jsonrpc":"2.0","id":3,"result":{
            "content":[{"type":"text","text":"<json>"}],
            "structuredContent":{...},"isError":false}}
```

The server must echo a `protocolVersion` it supports and ignore unknown
notification methods. A `ping` request should answer `{}`.

**Forward compatibility.** The 2026-07-28 spec retires `initialize`/`initialized`
in favour of a stateless core: each request carries protocol version, client
identity, and capabilities in `_meta`, and capability discovery is an optional
`server/discover` RPC. — <https://blog.modelcontextprotocol.io/posts/2026-07-28/>
A server that supports only the handshake above will not serve a 2026-07-28
client. Until both harnesses advertise the new core, ship the 2025-06-18 server
and treat stateless support (accept `tools/list`/`tools/call` with no prior
`initialize`) as a follow-up negotiated by `protocolVersion` rather than a
rewrite.

### 2.2 Tool definition and input schema

```json
{
  "name": "decide",
  "title": "Typed decision readout",
  "description": "Answer Decision Index questions against the served model by reading one answer slot. Returns a probability distribution over the supplied criteria for each question. A question that does not fit the model or whose labels cannot be scored is refused, not guessed.",
  "inputSchema": {
    "type": "object",
    "required": ["state", "questions"],
    "properties": {
      "state": {
        "description": "Any JSON value or string; the context to decide from. Empty object/array render as (empty)."
      },
      "questions": {
        "type": "object",
        "minProperties": 1,
        "additionalProperties": {
          "oneOf": [
            {
              "type": "object",
              "required": ["type", "instructions", "criteria"],
              "properties": {
                "type": { "const": "choice" },
                "instructions": { "type": "string" },
                "criteria": {
                  "type": "object",
                  "minProperties": 2,
                  "maxProperties": 255,
                  "additionalProperties": { "type": ["string", "null"] }
                }
              }
            },
            {
              "type": "object",
              "required": ["type", "instructions"],
              "properties": {
                "type": { "const": "noul" },
                "instructions": { "type": "string" }
              }
            }
          ]
        }
      }
    }
  }
}
```

The tool is generic: the model supplies the decision content, pqnld supplies the
typed readout. `state` mirrors `POST /v1/decide` (string or any JSON value);
`questions` is keyed by the caller and may use short local keys.

### 2.3 Output

On success, `structuredContent` is the `/v1/decide` response body (answer keys
equal question keys; `choice`/`probabilities` for `choice`, `noul` for `noul`),
and `content[0]` carries the same JSON as text for clients that do not read
structured output. An optional `outputSchema` may mirror it; when one is
supplied a conforming `structuredContent` is mandatory, so omit `outputSchema`
until the shape is frozen.

### 2.4 Errors

Two mechanisms, per the MCP tools spec:

- **Protocol errors** (`error.code`), for requests the tool cannot even begin to
  serve: `-32700` parse, `-32600` invalid request, `-32601` method not found,
  `-32602` invalid params (missing/non-object `questions`, unknown tool name).
- **Tool execution errors** (`result.isError: true`), for a well-formed call
  that failed. These are returned as a result so the model can see, explain, and
  retry rather than treating it as a transport failure.

`decide` maps the sidecar's HTTP status into a stable code and never returns a
partial or invented answer (matching [wire.md](wire.md)):

| Sidecar | Code | Meaning |
|---|---|---|
| `400` | `bad_request` | malformed body / missing `questions` |
| `422` | `unsupported` | unsupported question type, incomplete label coverage, or a question that does not fit the model's context; the model's capacity message is preserved verbatim |
| `404` | `not_found` | unknown path (misconfiguration) |
| `500` | `engine_error` | engine error or readout self-check failure |
| transport | `engine_unreachable` | sidecar not running / wrong URL |

```json
{"jsonrpc":"2.0","id":3,"result":{
  "content":[{"type":"text","text":"{\"error\":{\"code\":\"unsupported\",\"message\":\"...\"}}"}],
  "isError":true}}
```

## 3. Config snippets

### opencode — `opencode.json` (or `~/.config/opencode/opencode.json`)

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "pqnld": {
      "type": "local",
      "command": ["python3", "/abs/path/sidecar-rs/mcp_bridge.py"],
      "enabled": true,
      "environment": {
        "PQNLD_SIDECAR_URL": "http://127.0.0.1:11560"
      }
    }
  }
}
```

Local plugins (JS/TS) are loaded from `.opencode/plugins/` or
`~/.config/opencode/plugins/`; npm plugins from the `plugin` array. A native
plugin would export a `tool` named `decide` that calls the same endpoint
(<https://opencode.ai/docs/plugins/>).

### Codex — `~/.codex/config.toml` (or trusted `.codex/config.toml`)

```toml
[mcp_servers.pqnld]
command = "python3"
args = ["/abs/path/sidecar-rs/mcp_bridge.py"]
startup_timeout_sec = 10
tool_timeout_sec = 600
enabled = true

[mcp_servers.pqnld.env]
PQNLD_SIDECAR_URL = "http://127.0.0.1:11560"
```

Or generate the table with
`codex mcp add pqnld -- python3 /abs/path/sidecar-rs/mcp_bridge.py`.
`tool_timeout_sec` must exceed the slowest decision (the sidecar default engine
timeout is 600 s; decisions are sequential by default). Use `enabled_tools =
["decide"]` if the server later grows more tools.

## 4. Reaching the model endpoint: capability tiers

A `decide` tool needs a model that exposes either token scores or, failing that,
a single parseable choice. These are the same tiers the sidecar already reasons
about; the adapter only has to select and expose them.

| Tier | Endpoint | Mechanism | Coverage | Output | Config |
|---|---|---|---|---|---|
| **T1 exact-ID** | vLLM / OpenAI-compatible with `/tokenize` + `logprob_token_ids` + `return_tokens_as_token_ids` | resolve each single-token label, request its id at the answer slot | full distribution, any N; >26 relabel + `ceil(N/128)` split/merge | `choice` + `probabilities` | descriptor `specific_token_scores: true` |
| **T2 top-k** | OpenAI-compatible `logprobs` only, no explicit ids | `top_logprobs` at the answer slot | full distribution only while every label is in the returned top-k; the API caps `top_logprobs` at 20, so practical ceiling <=20 options | `choice` + `probabilities` | default readout, no `specific_token_scores` |
| **T3 parsed** | provider with no logprob support | one `max_tokens=1` chat call, parse the returned letter/key | one choice per question | `choice` only, **no** `probabilities` | forced readout `parse` |

Notes grounded in this repo and the API docs:

- T1 is the only tier that answers the >26-option case losslessly; it is what the
  canonical sidecar uses, via tokenizer-verified single-token labels and the
  128-id request cap ([architecture.md](architecture.md), [descriptor.md](descriptor.md)).
- T2 is bounded by the OpenAI `top_logprobs` maximum of 20; a 26-option request
  is rejected by the engine as `greater than max allowed: 20`
  ([readout-baseline.md](readout-baseline.md)). OpenAI documents that it "may be
  fewer than the requested `top_logprobs`", so missing labels are refused rather
  than assigned low scores. — <https://platform.openai.com/docs/api-reference/chat>
- T3 is a **capability downgrade, not a probability**: it returns a point choice
  and must set a `distribution: false` marker so callers do not read confidence
  into it. pqnld's normalization is not calibration in any tier.

**Detection (`auto`).** Follow the sidecar's probe order and cache the result:
try a single-token `/tokenize` and a fixture lettered request; if explicit ids
work, T1; else if the answer slot carries `top_logprobs`, T2; else T3. A
descriptor/config value (`readout: lettered|echo|parse`) forces a tier and skips
the probe. A T2 request whose labels are not all returned, and any T1 request
whose labels do not tokenize to distinct single tokens, is a `422 unsupported` —
never a fabricated score.

**Adapter environment / config.** The MCP server is configured by environment so
the same snippet works in both harnesses:

| Variable | Meaning | Default |
|---|---|---|
| `PQNLD_SIDECAR_URL` | bridge mode: running sidecar base URL | `http://127.0.0.1:11560` |
| `PQNLD_ENGINE` | embedded mode: model endpoint base URL | `http://127.0.0.1:11542` |
| `PQNLD_MODEL` | served model id | `qwen38-27b` |
| `PQNLD_DESCRIPTOR` | `models/<name>.json` descriptor | model name |
| `PQNLD_MODELS_DIR` | descriptor directory | `models` |
| `PQNLD_READOUT` | `auto`/`lettered`/`echo`/`parse` | `auto` |
| `PQNLD_TEMPERATURE` | softmax temperature | descriptor / `1.0` |
| `PQNLD_API_KEY` | bearer token for a hosted endpoint | unset |

Per endpoint:

- **Local vLLM** — `PQNLD_ENGINE=http://127.0.0.1:11542`, `PQNLD_MODEL=<served
  id>`, descriptor with `specific_token_scores: true`; no token. Selects T1.
- **Hosted OpenAI** — `PQNLD_ENGINE=https://api.openai.com/v1`,
  `PQNLD_MODEL=gpt-...`, `PQNLD_API_KEY=...`. No `/tokenize`, so detection lands
  on T2; only <=20-option questions answer.
- **No-logprob provider** — T3; `decide` returns choices but no distribution.

## 5. Minimal implementation path

1. **MCP stdio bridge (recommended first).** A ~100-line stdio<->HTTP proxy that
   forwards `tools/call decide` to an already-running `pqnld-rs`/Python sidecar
   `POST /v1/decide` and maps the errors above. Zero readout changes, so it
   inherits T1/T2, the LRU cache, the `auto` probe, and the output self-check,
   and both harnesses get a working `decide` tool today. A prototype lives at
   `sidecar-rs/mcp_bridge.py` (it does not touch `sidecar-rs/src/main.rs`).
2. **Embedded MCP server (portability).** Reuse the readout core directly
   (`src/pqnld/decision.py` today, or a small library split of the Rust core) so
   no separate sidecar process is required, and add the T3 parsed fallback plus
   an explicit T2 top-k ceiling check. This is the step that makes "the
   harness's own model, no extra service" true; it is also where the tiers in
   section 4 are actually implemented.
3. **Harness-native wrapper (only if UX demands it).** An opencode plugin can add
   a `decide` tool that pre-fills `state` from the session context and avoids
   MCP catalog overhead. Codex has no plugin API, so it stays on MCP.

What each harness gains / blockers:

| | Gains | Blockers |
|---|---|---|
| opencode | MCP tool auto-available to the model; optional native plugin for session-context prefills | MCP catalog adds context tokens; the configured reasoning model may itself lack logprobs, so only T3 is possible unless pointed at a logprob endpoint |
| Codex | MCP stdio with env passthrough, tool allow/deny and per-tool approval; shared CLI/app/IDE config | MCP only, no native tool; project config requires a trusted project; same model-logprob limitation |

## 6. Limits

- Probabilities are readout scores, not calibrated beliefs; calibration needs
  held-out labeled data ([architecture.md](architecture.md)).
- The harness's reasoning model and the readout model can differ; when they do,
  `decide` must be pointed at the model that exposes token scores (T1/T2).
- MCP servers add tools to the model's context; the single `decide` tool is
  deliberately minimal to bound that cost.
- 2026-07-28 stateless MCP is not yet handled; see section 2.1.
