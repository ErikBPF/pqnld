//! Native MCP stdio server: exposes pqnld's decision readout as a `decide`
//! tool over newline-delimited JSON-RPC 2.0 on stdin/stdout.
//!
//! This replaces the former `mcp_bridge.py`: it serves decisions in-process via
//! the same readout, cache, probe, and self-check, with no separate sidecar.

use std::sync::Arc;

use serde_json::{json, Value};

use super::engine::EngineError;
use super::{readout_call, Ctx};

const PROTOCOL: &str = "2025-06-18";

fn tool() -> Value {
    json!({
        "name": "decide",
        "title": "Typed decision readout",
        "description": "Answer Decision Index questions against the served model by reading one answer slot. Returns a probability distribution over the supplied criteria for each question. A question that does not fit the model or whose labels cannot be scored is refused, not guessed.",
        "inputSchema": {
            "type": "object",
            "required": ["state", "questions"],
            "properties": {
                "state": {"description": "Any JSON value or string; the context to decide from."},
                "questions": {
                    "type": "object",
                    "minProperties": 1,
                    "additionalProperties": {
                        "oneOf": [
                            {
                                "type": "object",
                                "required": ["type", "instructions", "criteria"],
                                "properties": {
                                    "type": {"const": "choice"},
                                    "instructions": {"type": "string"},
                                    "criteria": {
                                        "type": "object",
                                        "minProperties": 2,
                                        "maxProperties": 255,
                                        "additionalProperties": {"type": ["string", "null"]}
                                    }
                                }
                            },
                            {
                                "type": "object",
                                "required": ["type", "instructions"],
                                "properties": {
                                    "type": {"const": "noul"},
                                    "instructions": {"type": "string"}
                                }
                            }
                        ]
                    }
                }
            }
        }
    })
}

fn write_value(value: &Value) {
    println!("{}", serde_json::to_string(value).unwrap_or_default());
}

fn invalid(id: &Value, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32602, "message": message}})
}

async fn call_tool(ctx: &Arc<Ctx>, id: &Value, message: &Value) -> Value {
    let params = message.get("params").cloned().unwrap_or(Value::Null);
    if params.get("name").and_then(|n| n.as_str()) != Some("decide") {
        return invalid(id, "unknown tool");
    }
    let arguments = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
    let state = arguments.get("state").cloned();
    let questions = arguments.get("questions").and_then(|q| q.as_object()).cloned();
    let (state, questions) = match (state, questions) {
        (Some(state), Some(questions)) if !questions.is_empty() => (state, questions),
        _ => return invalid(id, "decide requires state and non-empty questions"),
    };
    match readout_call(ctx.clone(), &state, &questions).await {
        Ok(mut answer) => {
            answer["model"] = Value::String(ctx.model.clone());
            let text = serde_json::to_string(&answer).unwrap_or_default();
            json!({"jsonrpc": "2.0", "id": id, "result": {
                "content": [{"type": "text", "text": text}],
                "structuredContent": answer,
                "isError": false
            }})
        }
        Err(error) => {
            let (code, status, message) = match error {
                EngineError::Unsupported(message) => ("unsupported", 422, message),
                EngineError::Other(message) => ("engine_error", 500, message),
            };
            let payload = json!({"error": {"code": code, "status": status, "message": message}});
            let text = serde_json::to_string(&payload).unwrap_or_default();
            json!({"jsonrpc": "2.0", "id": id, "result": {
                "content": [{"type": "text", "text": text}],
                "isError": true
            }})
        }
    }
}

async fn handle(ctx: &Arc<Ctx>, message: &Value) -> Option<Value> {
    let method = message.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let id = message.get("id").cloned();
    let id = match id {
        Some(id) if !id.is_null() => id,
        _ => return None, // notification
    };
    let result = match method {
        "initialize" => {
            let requested = message
                .get("params")
                .and_then(|p| p.get("protocolVersion"))
                .and_then(|v| v.as_str())
                .unwrap_or(PROTOCOL);
            let version = if requested == PROTOCOL { requested } else { PROTOCOL };
            json!({"jsonrpc": "2.0", "id": id, "result": {
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": false}},
                "serverInfo": {"name": "pqnld", "version": env!("CARGO_PKG_VERSION")},
                "instructions": "Use the decide tool to answer typed choice/noul questions from a state. Prefer a served model that exposes answer-slot logprobs."
            }})
        }
        "ping" => json!({"jsonrpc": "2.0", "id": id, "result": {}}),
        "tools/list" => json!({"jsonrpc": "2.0", "id": id, "result": {"tools": [tool()]}}),
        "tools/call" => return Some(call_tool(ctx, &id, message).await),
        other => json!({"jsonrpc": "2.0", "id": id, "error": {
            "code": -32601, "message": format!("method not found: {other}")
        }}),
    };
    Some(result)
}

pub async fn serve(ctx: Arc<Ctx>) {
    use std::io::{BufRead, Write};
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(_) => break,
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let message: Value = match serde_json::from_str(trimmed) {
            Ok(message) => message,
            Err(e) => {
                let _ = writeln!(
                    stdout,
                    "{}",
                    json!({"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": e.to_string()}})
                );
                let _ = stdout.flush();
                continue;
            }
        };
        if let Some(response) = handle(&ctx, &message).await {
            write_value(&response);
        }
    }
}
