# pqnld-rs

A single static binary that serves the decision wire contract
(`POST /v1/decide`, alias `/v1/systemone`, plus a `/v1/chat/completions` shim
and `GET /healthz`) or runs as an MCP stdio server (`--mcp`). It reads a served
vLLM/OpenAI model's answer-slot logprobs over tokenizer-verified option labels
and returns a typed distribution — no runtime dependency.

## Build

```sh
cargo build --release
# -> target/release/pqnld-rs
```

For a fully static binary (no glibc dependency, easiest to package):

```sh
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl
```

`build.sh` is the Apollo convenience wrapper: it installs a minimal toolchain
under `RUSTUP_HOME`/`CARGO_HOME` if `cargo` is missing and builds in place.

## Run

```sh
pqnld-rs --vllm-url http://127.0.0.1:11542 \
         --model qwen38-27b-nvfp4 \
         --descriptor qwen38-27b-nvfp4 --models-dir ./models \
         --host 127.0.0.1 --port 11560 \
         --uds /tmp/pqnld.sock
```

| Flag | Env (legacy) | Default | Meaning |
|---|---|---|---|
| `--vllm-url` | `PQNLD_VLLM_URL` (`DECISION_VLLM_URL`) | `http://127.0.0.1:11542` | Engine base URL |
| `--model` | `PQNLD_MODEL` (`DECISION_MODEL`) | `qwen38-27b` | Served model id |
| `--descriptor` | `PQNLD_DESCRIPTOR` (`DECISION_DESCRIPTOR`) | model name | `models/<name>.json` |
| `--models-dir` | `PQNLD_MODELS_DIR` (`DECISION_MODELS_DIR`) | `models` | Descriptor directory |
| `--host` / `--port` | `PQNLD_HOST` / `PQNLD_PORT` | `127.0.0.1` / `11560` | TCP listener |
| `--uds` | — | off | Also listen on a Unix domain socket |
| `--mcp` | — | off | Serve the `decide` tool on MCP stdio instead of HTTP |
| `--workers` | — | `1` | Global budget for active sidecar question scoring and probes |
| `--temperature` | `PQNLD_TEMPERATURE` (`DECISION_TEMPERATURE`) | descriptor / `1.0` | Softmax temperature |
| `--timeout` | — | `600` | Engine request timeout (s) |

**UDS safety limit:** existing startup code removes the supplied UDS path without
checking that it is a socket. Do not point `--uds` at a pre-existing regular file
or other non-socket path.

## Scheduling

Question scoring and initial readout probes share a **global sidecar budget**
(`--workers 1` by default). Each request retains stored question order at one
worker; raising the budget permits concurrency. Failed concurrent decisions
cancel and drain their remaining local tasks, reclaiming worker permits.

The budget covers sidecar scoring, not ordinary chat sent directly to the model.
Cancellation stops local tasks; upstream requests already accepted stay accepted.
Engine batch numerics, MTP, restarts and cache state can change scores between runs.

### MCP

```sh
pqnld-rs --mcp --models-dir ./models --vllm-url http://127.0.0.1:11542
```

Speaks newline-delimited JSON-RPC 2.0 on stdio and exposes one `decide` tool
taking `{state, questions}`. See [`docs/harness-plugins.md`](../docs/harness-plugins.md)
for opencode/Codex config.

## Descriptor

`models/<name>.json`: `readout` (`lettered`|`echo`|
`auto`), `letters`, `temperature`, `enable_thinking`, `system_prompt`,
`letter_system_prompt`, `chat_template`, `engine_profile`,
`specific_token_scores`. Set `specific_token_scores: true` with a compatible
adapter to enable explicit-ID scoring and an extended single-token alphabet.
Above the tested 128-ID cap, the same prompt is split across `ceil(n/128)` requests
and scores are merged. Without this opt-in, questions exceeding the configured
alphabet use echo instead; tokenizer and backend capacity still apply.
