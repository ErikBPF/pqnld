#!/usr/bin/env python3
"""REST/OpenAPI tester for a pqnld decision endpoint.

Exposes two endpoints: POST /decide forwards a Decision Index request to an
upstream OpenAI chat endpoint that answers with a typed decision, and returns
the parsed decision; POST /complete runs a plain text chat completion through
the same upstream. Swagger UI is served at /docs and the machine-readable
schema at /openapi.json.

    pqnld-tester                  # or: uvicorn pqnld.tester:app --port 8088

Config (env):
    PQNLD_BASE_URL  default http://127.0.0.1:11560/v1 (a pqnld sidecar)
    PQNLD_MODEL     default pqnld (decision model for POST /decide)
    PQNLD_COMPLETION_MODEL default qwen-chat (text model for POST /complete)
    PQNLD_API_KEY   default $LITELLM_API_KEY (empty -> no Authorization header)
    PQNLD_TIMEOUT   default 600 seconds
    PQNLD_MAX_TOKENS default 256 (thinking is on by default)
"""

import json
import os
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_URL = os.environ.get("PQNLD_BASE_URL", "http://127.0.0.1:11560/v1").rstrip("/")
MODEL = os.environ.get("PQNLD_MODEL", "pqnld")
COMPLETION_MODEL = os.environ.get("PQNLD_COMPLETION_MODEL", "qwen-chat")
COMPLETION_THINKING = os.environ.get("PQNLD_COMPLETION_THINKING", "false").strip().lower() in {"1", "true", "yes"}
API_KEY = os.environ.get("PQNLD_API_KEY") or os.environ.get("LITELLM_API_KEY") or ""
TIMEOUT = float(os.environ.get("PQNLD_TIMEOUT", "600"))
MAX_TOKENS = int(os.environ.get("PQNLD_MAX_TOKENS", "256"))
CLIENT = httpx.Client(timeout=TIMEOUT)

EXAMPLE = {
    "state": "The meeting is on Tuesday at 3pm in room B.",
    "questions": {
        "day": {
            "type": "choice",
            "instructions": "Which day is the meeting?",
            "criteria": {"monday": "Monday", "tuesday": "Tuesday", "friday": "Friday"},
        },
        "is_urgent": {"type": "noul", "instructions": "Is the meeting urgent?"},
    },
}


class Question(BaseModel):
    type: str = Field(..., description="'choice' (2-255 options) or 'noul'")
    instructions: Any = Field(None, description="What is being asked")
    criteria: Optional[dict[str, Optional[str]]] = Field(
        None, description="option key -> description; required for 'choice'"
    )


class Answer(BaseModel):
    type: str
    choice: Optional[str] = None
    probabilities: Optional[dict[str, float]] = None
    noul: Optional[float] = None


class DecisionRequest(BaseModel):
    state: Any = Field("", description="Context the engine reads")
    questions: dict[str, Question]

    model_config = {"json_schema_extra": {"examples": [EXAMPLE]}}


class DecisionResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Optional[dict[str, Any]] = None


class CompletionRequest(BaseModel):
    prompt: str = Field(..., description="User message for the upstream chat model")
    system: Optional[str] = Field(None, description="Optional system message")
    model: Optional[str] = Field(None, description="Override the upstream model (default qwen-chat)")
    max_tokens: Optional[int] = Field(None, description="Override the output token budget")
    temperature: float = Field(0.0, description="Sampling temperature; 0 is deterministic")

    model_config = {
        "json_schema_extra": {"examples": [{"prompt": "Say hello in one word.", "system": "Be terse."}]}
    }


class CompletionResponse(BaseModel):
    model: str
    content: str
    usage: Optional[dict[str, Any]] = None


app = FastAPI(
    title="pqnld decision tester",
    version="1.0.0",
    description=(
        "Forwards Decision Index requests to a pqnld decision endpoint and "
        "returns typed decisions (choice + probability distribution, or noul)."
    ),
)


def _request(
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int,
    temperature: float,
    chat_template_kwargs: Optional[dict[str, Any]] = None,
) -> dict:
    headers = {}
    if API_KEY:
        headers["Authorization"] = "Bearer " + API_KEY
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if chat_template_kwargs is not None:
        body["chat_template_kwargs"] = chat_template_kwargs
    try:
        response = CLIENT.post(BASE_URL + "/chat/completions", headers=headers, json=body)
    except httpx.HTTPError as error:
        raise HTTPException(502, f"upstream unreachable: {error}")
    if response.status_code >= 400:
        raise HTTPException(502, f"upstream {response.status_code}: {response.text[:500]}")
    return response.json()


def _content(payload: dict) -> str:
    content = (payload.get("choices") or [{}])[0].get("message", {}).get("content")
    if not content:
        raise HTTPException(502, "upstream returned empty content; raise max_tokens")
    return content


def _chat(content: str) -> str:
    return _content(_request([{"role": "user", "content": content}], MODEL, MAX_TOKENS, 0))


@app.get("/health")
def health():
    return {"status": "ok", "base_url": BASE_URL, "model": MODEL, "auth": bool(API_KEY)}


@app.post("/decide", response_model=DecisionResponse, summary="Answer typed decision questions")
def decide(request: DecisionRequest):
    payload = {
        "model": MODEL,
        "state": request.state,
        "questions": {
            key: question.model_dump(exclude_none=True) for key, question in request.questions.items()
        },
    }
    content = _chat(json.dumps(payload, ensure_ascii=False))
    try:
        return DecisionResponse.model_validate_json(content)
    except Exception as error:
        raise HTTPException(502, f"qwen-jev returned non-decision content: {content[:300]!r} ({error})")


@app.post("/complete", response_model=CompletionResponse, summary="Plain text chat completion")
def complete(request: CompletionRequest):
    messages = []
    if request.system:
        messages.append({"role": "system", "content": request.system})
    messages.append({"role": "user", "content": request.prompt})
    payload = _request(
        messages,
        request.model or COMPLETION_MODEL,
        request.max_tokens or MAX_TOKENS,
        request.temperature,
        chat_template_kwargs={"enable_thinking": COMPLETION_THINKING},
    )
    return CompletionResponse(
        model=payload.get("model", request.model or COMPLETION_MODEL),
        content=_content(payload),
        usage=payload.get("usage"),
    )


def main():
    import uvicorn

    uvicorn.run(app, host=os.environ.get("PQNLD_HOST", "127.0.0.1"), port=int(os.environ.get("PQNLD_PORT", "8088")))


if __name__ == "__main__":
    main()
