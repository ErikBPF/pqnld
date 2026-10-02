#!/usr/bin/env python3
"""MCP stdio bridge: expose pqnld's Decision Index contract as a ``decide`` tool.

Newline-delimited JSON-RPC 2.0 on stdin/stdout, diagnostics on stderr. Each
``tools/call`` for ``decide`` is forwarded to a running pqnld sidecar's
``POST /v1/decide`` at ``PQNLD_SIDECAR_URL`` (default http://127.0.0.1:11560).

This is the minimal adapter from docs/harness-plugins.md: it reuses the
sidecar's readout, cache, auto probe, and self-check unchanged. It does not
implement the model-side capability tiers itself; the sidecar it points at does.

Run:   python3 mcp_bridge.py             # serve MCP on stdio
Check: python3 mcp_bridge.py --selftest  # no network, no harness needed
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

PROTOCOL = "2025-06-18"
SERVER_INFO = {"name": "pqnld", "version": "0.1.0"}
SIDECAR_URL = os.environ.get("PQNLD_SIDECAR_URL", "http://127.0.0.1:11560").rstrip("/")
MODEL = os.environ.get("PQNLD_MODEL", "qwen38-27b-nvfp4")
TIMEOUT = float(os.environ.get("PQNLD_TIMEOUT", "600"))

TOOL = {
    "name": "decide",
    "title": "Typed decision readout",
    "description": (
        "Answer Decision Index questions against the served model by reading one "
        "answer slot. Returns a probability distribution over the supplied "
        "criteria for each question. A question that does not fit the model or "
        "whose labels cannot be scored is refused, not guessed."
    ),
    "inputSchema": {
        "type": "object",
        "required": ["state", "questions"],
        "properties": {
            "state": {
                "description": "Any JSON value or string; the context to decide from."
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
                                "type": {"const": "choice"},
                                "instructions": {"type": "string"},
                                "criteria": {
                                    "type": "object",
                                    "minProperties": 2,
                                    "maxProperties": 255,
                                    "additionalProperties": {"type": ["string", "null"]},
                                },
                            },
                        },
                        {
                            "type": "object",
                            "required": ["type", "instructions"],
                            "properties": {
                                "type": {"const": "noul"},
                                "instructions": {"type": "string"},
                            },
                        },
                    ]
                },
            },
        },
    },
}

ERROR_CODES = {400: "bad_request", 422: "unsupported", 404: "not_found", 500: "engine_error"}


class ToolError(Exception):
    def __init__(self, code, message, status=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def call_sidecar(state, questions):
    body = json.dumps({"model": MODEL, "state": state, "questions": questions}).encode("utf-8")
    request = urllib.request.Request(
        SIDECAR_URL + "/v1/systemone",
        data=body,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        message = error.read().decode("utf-8", "replace")
        raise ToolError(ERROR_CODES.get(error.code, "engine_error"), message, error.code)
    except OSError as error:
        raise ToolError("engine_unreachable", str(error))


def error_result(error):
    payload = json.dumps(
        {"error": {"code": error.code, "status": error.status, "message": error.message}},
        ensure_ascii=False,
    )
    return {"content": [{"type": "text", "text": payload}], "isError": True}


def handle(message):
    method = message.get("method")
    request_id = message.get("id")
    if request_id is None:
        return None  # notification

    if method == "initialize":
        params = message.get("params") or {}
        requested = params.get("protocolVersion") or PROTOCOL
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": requested if requested == PROTOCOL else PROTOCOL,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
                "instructions": (
                    "Use the decide tool to answer typed choice/noul questions from a "
                    "state. Prefer a served model that exposes answer-slot logprobs."
                ),
            },
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": [TOOL]}}
    if method == "tools/call":
        params = message.get("params") or {}
        if params.get("name") != "decide":
            return _invalid(request_id, "unknown tool")
        arguments = params.get("arguments") or {}
        questions = arguments.get("questions")
        if "state" not in arguments or not isinstance(questions, dict) or not questions:
            return _invalid(request_id, "decide requires state and non-empty questions")
        try:
            answer = call_sidecar(arguments["state"], questions)
        except ToolError as error:
            return {"jsonrpc": "2.0", "id": request_id, "result": error_result(error)}
        text = json.dumps(answer, ensure_ascii=False)
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": text}],
                "structuredContent": answer,
                "isError": False,
            },
        }
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"method not found: {method}"},
    }


def _invalid(request_id, message):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": message}}


def write(message):
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def serve():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as error:
            write({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(error)}})
            continue
        response = handle(message)
        if response is not None:
            write(response)


def selftest():
    assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    init = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})["result"]
    assert init["protocolVersion"] == PROTOCOL and "tools" in init["capabilities"]
    tools = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
    assert tools[0]["name"] == "decide"
    assert tools[0]["inputSchema"]["required"] == ["state", "questions"]
    bad = handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "decide", "arguments": {}}})
    assert bad["error"]["code"] == -32602

    global call_sidecar
    real = call_sidecar
    call_sidecar = lambda state, questions: {"model": "m", "answers": {}, "usage": {}}
    ok = handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "decide", "arguments": {"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}}}})
    assert ok["result"]["isError"] is False and ok["result"]["structuredContent"]["model"] == "m"

    def boom(state, questions):
        raise ToolError("unsupported", "capacity", 422)

    call_sidecar = boom
    refused = handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "decide", "arguments": {"state": "s", "questions": {"q": {"type": "noul", "instructions": "x"}}}}})
    body = json.loads(refused["result"]["content"][0]["text"])
    assert refused["result"]["isError"] is True and body["error"]["code"] == "unsupported"
    call_sidecar = real
    print("ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        serve()
