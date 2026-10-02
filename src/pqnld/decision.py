#!/usr/bin/env python3
"""pqnld — turn any served vLLM/OpenAI model into a typed decision engine.

Renders a Decision Index question with letter-labelled options, reads the
model's answer-slot distribution over those letters in ONE chat request, and
serves the typed distribution at POST /v1/decide. The reference kit's wire
path /v1/systemone is kept as an alias so a stock kit client still runs.

One request per question (not one per option): the previous echo design issued
n+1 full-context prefills per question because this hybrid model's unified block
size (864) exceeds a decision prompt, so nothing is prefix-cached. Letter labels
must be single tokens; complete answer-slot scores give a distribution
conditioned on the selected labels, without prompt-logprob memory risk.

A model's readout is described by a models/<name>.json descriptor: the readout
route (lettered|echo|auto), the chat template, the thinking toggle, the letters
and the temperature. An "auto" descriptor probes the model at startup and keeps
the lettered readout only when a single option letter is the top token for a
fixture; otherwise it forces the echo readout. A descriptor that declares the
echo readout without a chat template is refused.

Models that support explicit-token scores can also answer questions with more
than 26 options: the option set is relabelled with tokenizer-verified
single-token symbols and scored with the same answer-slot readout, splitting the
shared prompt across ceil(n/128) requests when the engine's per-request
explicit-token cap is exceeded. Models without that support fall back to a
chunked echo-score path above 26 options.

    pqnld --vllm-url http://127.0.0.1:11542 \
        --model qwen38-27b-nvfp4 --descriptor qwen38-27b-nvfp4

Config falls back to PQNLD_VLLM_URL / PQNLD_MODEL / PQNLD_DESCRIPTOR /
PQNLD_MODELS_DIR / PQNLD_HOST / PQNLD_PORT / PQNLD_TEMPERATURE (the earlier
DECISION_* names are still read).
"""

import argparse
import json
import math
import os
import string
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

SYSTEM_PROMPT = (
    "You are a decision engine. Read the state, then answer the question by "
    "replying with exactly one option key from the list. Do not explain."
)

LETTER_SYSTEM_PROMPT = (
    "You are a decision engine. Read the state, then answer the question by "
    "replying with exactly one option letter from the list. Do not explain."
)

# Qwen3.8 ChatML with the thinking block pre-filled empty (enable_thinking=false),
# used by the echo fallback.
CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{body}<|im_end|>\n"
    "<|im_start|>assistant\n thinking\n\n</think>\n\n"
)

LETTERS = "abcdefghijklmnopqrstuvwxyz"
# Single-token labels for questions with more than 26 options: lowercase first
# (consistent with LETTERS), then uppercase, digits, punctuation and Greek and
# Cyrillic letters. Each label is kept only if the tokenizer renders it as one
# distinct token, so the exact set is resolved against the live tokenizer.
EXTENDED_LABELS = (
    string.ascii_lowercase
    + string.ascii_uppercase
    + string.digits
    + string.punctuation
    + "".join(chr(c) for c in range(0x391, 0x3AA))
    + "".join(chr(c) for c in range(0x3B1, 0x3CA))
    + "".join(chr(c) for c in range(0x410, 0x450))
)
# vLLM caps explicit logprob_token_ids per request (MAX_LOGPROB_TOKEN_IDS=128),
# so wider option sets share one prompt across ceil(n/128) merged requests.
MAX_LOGPROB_IDS = 128
# Cap on questions read concurrently. Bounds engine load for 255-question
# documents and keeps the echo fallback's memory (per-question batch) in check.
MAX_QUESTION_WORKERS = 1

PROBE_STATE = "The sky is blue."
PROBE_QUESTION = {
    "type": "choice",
    "instructions": "Which colour is named?",
    "criteria": {"red": "red", "blue": "blue", "green": "green"},
}

CAPACITY_MARKERS = (
    "options per choice",
    "a choice needs at least two options",
    "a score takes 2 to 10 levels",
    "the canvas holds",
    "maximum context length",
    "maximum model length",
    "longer than the maximum model length",
    "context window",
    "too many tokens",
)

CHAT_NOTICE = (
    "This model is a typed decision engine. Send a JSON object with 'state' and "
    "'questions' (Decision Index format) as the user message."
)


class Unsupported(ValueError):
    """Question the engine cannot answer, or a request the model rejected."""


@dataclass
class ModelSpec:
    """Per-model readout configuration (a models/<name>.json descriptor)."""

    readout: str = "auto"
    letters: str = LETTERS
    temperature: float = 1.0
    enable_thinking: bool = False
    system_prompt: str = SYSTEM_PROMPT
    letter_system_prompt: str = LETTER_SYSTEM_PROMPT
    chat_template: Optional[str] = CHAT_TEMPLATE
    engine_profile: str = "full"
    specific_token_scores: bool = False


def load_spec(name, directory=None):
    """Load models/<name>.json, or the built-in defaults when there is none."""
    directory = directory or os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    path = os.path.join(directory, f"{name}.json")
    if not os.path.isfile(path):
        return ModelSpec()
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    known = {field.name for field in fields(ModelSpec)}
    return ModelSpec(**{key: value for key, value in data.items() if key in known})


def decision_from_messages(messages):
    """Return (state, questions) parsed from the last user message, or None."""
    content = None
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            break
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except ValueError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("questions"), dict):
        return payload.get("state", ""), payload["questions"]
    return None


def text(value):
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _options(criteria):
    body = ["Options:"]
    for key, description in criteria.items():
        body.append(f"{key}: {key if description is None else text(description)}")
    return body


def render_prompt(state, question):
    body = [
        "State:",
        text(state) if state not in ("", None, {}, []) else "(empty)",
        "",
        "Question:",
        text(question.get("instructions", "")),
        "",
    ]
    criteria = question["criteria"] if question["type"] == "choice" else {"false": "False", "true": "True"}
    body += _options(criteria)
    body += ["", "Reply with exactly one option key."]
    return "\n".join(body)


def render_lettered(state, question, keys, letters=LETTERS):
    body = [
        "State:",
        text(state) if state not in ("", None, {}, []) else "(empty)",
        "",
        "Question:",
        text(question.get("instructions", "")),
        "",
        "Options:",
    ]
    criteria = question.get("criteria") or {}
    for letter, key in zip(letters, keys):
        description = criteria.get(key)
        body.append(f"{letter}) {key if description is None else text(description)}")
    body += ["", "Reply with exactly one option letter."]
    return "\n".join(body)


def letter_of(token):
    token = token.strip().lower()
    return token if len(token) == 1 and token in LETTERS else None


def validate(questions, response):
    """Kit-compatible check of a response against the questions it answers."""
    answers = response.get("answers", {})
    if set(answers) != set(questions):
        raise Unsupported("answer keys do not match question keys")
    for key, question in questions.items():
        answer = answers[key]
        if answer.get("type") != question["type"]:
            raise Unsupported(f"{key}: type mismatch")
        if question["type"] == "choice":
            if answer.get("choice") not in question["criteria"]:
                raise Unsupported(f"{key}: choice outside criteria")
            probabilities = answer.get("probabilities", {})
            if set(probabilities) != set(question["criteria"]):
                raise Unsupported(f"{key}: probability keys do not match criteria")
            if not all(isinstance(p, (int, float)) and math.isfinite(p) and 0.0 <= p <= 1.0 for p in probabilities.values()):
                raise Unsupported(f"{key}: probability out of range")
            if abs(sum(probabilities.values()) - 1.0) > 0.01:
                raise Unsupported(f"{key}: probabilities do not sum to 1")
        elif question["type"] == "noul":
            value = answer.get("noul")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise Unsupported(f"{key}: noul out of range")
        else:
            raise Unsupported(f"{key}: unsupported question type {question['type']!r}")


class Readout:
    def __init__(self, vllm_url, model, spec=None, temperature=None, timeout=600, batch=16, cache_size=256):
        self.model = model
        self.spec = spec or ModelSpec()
        if self.spec.readout not in ("auto", "lettered", "echo"):
            raise Unsupported(f"unknown readout {self.spec.readout!r}")
        if self.spec.readout == "echo":
            self._require_template()
        self.vllm_url = vllm_url.rstrip("/")
        self.temperature = self.spec.temperature if temperature is None else temperature
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.timeout = timeout
        self.batch = batch
        self.cache_size = cache_size
        self.letters = self.spec.letters
        self.mode = self.spec.readout if self.spec.readout in ("lettered", "echo") else None
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self._label_ids = {}

    def _require_template(self):
        if not self.spec.chat_template:
            raise Unsupported(f"the echo readout for {self.model!r} requires a chat_template")

    def ensure_mode(self):
        return self.mode or self.probe()

    def probe(self):
        """Resolve an "auto" descriptor against the live model, once."""
        if self.mode is not None:
            return self.mode
        try:
            _, _, top_token = self._letter_scores(PROBE_STATE, PROBE_QUESTION, list(PROBE_QUESTION["criteria"]))
            letter = letter_of(top_token or "")
            lettered = letter is not None and letter in self.letters
        except Unsupported:
            if self.spec.specific_token_scores:
                raise
            lettered = False
        except Exception:
            lettered = False
        if lettered:
            self.mode = "lettered"
        else:
            self._require_template()
            self.mode = "echo"
        return self.mode

    def _post(self, path, body):
        request = urllib.request.Request(
            self.vllm_url + path,
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            message = error.read().decode("utf-8", "replace")
            if error.code in (400, 413, 422) and any(marker in message for marker in CAPACITY_MARKERS):
                raise Unsupported(message)
            raise

    def _echo(self, prompt):
        return self._post("/v1/completions", {
            "model": self.model,
            "prompt": prompt,
            "echo": True,
            "max_tokens": 0,
            "temperature": 0,
            "logprobs": 1,
        })

    def _label_id(self, label, required=True):
        if label not in self._label_ids:
            tokens = self._post("/tokenize", {
                "model": self.model, "prompt": label, "add_special_tokens": False,
            })["tokens"]
            if len(tokens) != 1:
                if required:
                    raise Unsupported(f"option label {label!r} is not a single token")
                return None
            self._label_ids[label] = tokens[0]
        return self._label_ids[label]

    def _labels(self, count):
        """Return `count` distinct single-token labels, in option order."""
        if count <= len(self.letters):
            return list(self.letters[:count])
        labels, seen = [], set()
        for char in EXTENDED_LABELS:
            if len(labels) == count:
                break
            token_id = self._label_id(char, required=False)
            if token_id is None or token_id in seen:
                continue
            seen.add(token_id)
            labels.append(char)
        if len(labels) < count:
            raise Unsupported(
                f"tokenizer has only {len(labels)} single-token option labels, need {count}"
            )
        return labels

    def _letter_scores(self, state, question, keys):
        """Answer-slot logprobs over the option labels; one or more chat requests."""
        labels = self._labels(len(keys))
        body = render_lettered(state, question, keys, labels)
        token_letters = {}
        if self.spec.specific_token_scores:
            with self._lock:
                ids = [self._label_id(label) for label in labels]
                if len(set(ids)) != len(ids):
                    raise Unsupported("option labels do not have distinct token IDs")
                token_letters = {f"token_id:{token_id}": label for label, token_id in zip(labels, ids)}
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.spec.letter_system_prompt},
                {"role": "user", "content": body},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": max(len(keys), 20),
            "chat_template_kwargs": {"enable_thinking": self.spec.enable_thinking},
        }
        if token_letters:
            request.update(top_logprobs=0, return_tokens_as_token_ids=True)
            id_by_label = {label: self._label_ids[label] for label in labels}
            chunks = [labels[start : start + MAX_LOGPROB_IDS] for start in range(0, len(labels), MAX_LOGPROB_IDS)]
        else:
            chunks = [labels]
        by_label, usage, top_token = {}, {}, None
        for chunk in chunks:
            if token_letters:
                request["logprob_token_ids"] = [id_by_label[label] for label in chunk]
            payload = self._post("/v1/chat/completions", request)
            usage = payload.get("usage") or usage
            content = (payload["choices"][0].get("logprobs") or {}).get("content") or []
            if not content:
                raise RuntimeError("chat response carried no logprobs at the answer slot")
            if top_token is None:
                if token_letters:
                    top_token = token_letters.get(content[0].get("token"), "")
                else:
                    items = content[0].get("top_logprobs", [])
                    top_token = max(items, key=lambda item: item["logprob"])["token"] if items else None
            for item in content[0].get("top_logprobs", []):
                label = (token_letters.get(item.get("token")) if token_letters
                         else letter_of(item.get("token", "")))
                if label and label in labels and label not in by_label:
                    by_label[label] = item["logprob"]
        missing = [label for label in labels if label not in by_label]
        if missing:
            raise Unsupported(f"answer-slot logprobs missing option labels: {', '.join(missing)}")
        return [by_label[label] for label in labels], usage, top_token

    def _echo_scores(self, context, keys):
        """Fallback for >26 options: summed key-token logprobs, chunked."""
        ordered = sorted(keys)
        if any(right.startswith(left) for left, right in zip(ordered, ordered[1:])):
            raise Unsupported("echo scoring cannot compare prefix-overlapping option keys")
        context_tokens = self._echo(context)["choices"][0]["logprobs"]["tokens"]
        scores = []
        usage = {}
        for start in range(0, len(keys), self.batch):
            chunk = keys[start : start + self.batch]
            payload = self._echo([context + key for key in chunk])
            usage = payload.get("usage") or usage
            for choice in payload["choices"]:
                tokens = choice["logprobs"]["tokens"]
                token_logprobs = choice["logprobs"]["token_logprobs"]
                shared = 0
                while shared < len(context_tokens) and shared < len(tokens) and tokens[shared] == context_tokens[shared]:
                    shared += 1
                scores.append(sum(token_logprobs[shared:]))
        return scores, usage

    def _echo_context(self, state, question):
        if not self.spec.chat_template:
            raise Unsupported("this question needs the echo readout but the model has no chat_template")
        return self.spec.chat_template.format(system=self.spec.system_prompt, body=render_prompt(state, question))

    def _cache_key(self, state, questions):
        return json.dumps({
            "model": self.model,
            "readout": self.mode,
            "spec": asdict(self.spec),
            "temperature": self.temperature,
            "state": state,
            "questions": questions,
        }, ensure_ascii=False, default=str)

    def __call__(self, state, questions):
        mode = self.ensure_mode()
        cache_key = self._cache_key(state, questions)
        with self._lock:
            hit = self._cache.get(cache_key)
            if hit is not None:
                self._cache.move_to_end(cache_key)
                return dict(hit)
        answers = {}
        input_tokens = 0
        items = list(questions.items())
        # Questions are independent, so their upstream reads run at once: one
        # round trip per question in parallel rather than a sequential chain.
        workers = min(len(items), MAX_QUESTION_WORKERS) or 1
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(self._score_question, state, key, question, mode)
                for key, question in items
            ]
            for future in futures:
                key, answer, tokens = future.result()
                answers[key] = answer
                input_tokens += tokens
        response = {"model": self.model, "answers": answers, "usage": {"input_tokens": input_tokens}}
        try:
            validate(questions, response)
        except Unsupported as error:
            raise RuntimeError(f"readout self-check failed: {error}") from error
        with self._lock:
            self._cache[cache_key] = dict(response)
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return response

    def _score_question(self, state, key, question, mode):
        """Score one question; returns (key, answer, prompt_tokens)."""
        question_type = question.get("type")
        if question_type == "choice":
            keys = list(question["criteria"])
        elif question_type == "noul":
            keys = ["false", "true"]
        else:
            raise Unsupported(f"{key}: unsupported question type {question_type!r}")
        if mode == "lettered" and (len(keys) <= len(self.letters) or self.spec.specific_token_scores):
            scores, usage, _ = self._letter_scores(state, question, keys)
        else:
            scores, usage = self._echo_scores(self._echo_context(state, question), keys)
        peak = max(scores)
        weights = [math.exp((score - peak) / self.temperature) for score in scores]
        total = sum(weights)
        probabilities = {option: weight / total for option, weight in zip(keys, weights)}
        if question_type == "choice":
            choice = max(keys, key=lambda option: probabilities[option])
            answer = {"type": "choice", "choice": choice, "probabilities": probabilities}
        else:
            answer = {"type": "noul", "noul": probabilities["true"]}
        return key, answer, usage.get("prompt_tokens") or 0


class Handler(BaseHTTPRequestHandler):
    readout = None

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self._chat_completions()
            return
        # /v1/decide is the endpoint name; /v1/systemone is the frozen kit's
        # wire alias and is kept so the authoritative suite still runs.
        if self.path not in ("/v1/decide", "/v1/systemone"):
            self.send_error(404)
            return
        length = int(self.headers.get("content-length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            state = payload.get("state", "")
            questions = payload["questions"]
        except Exception as error:
            self._send(400, {"error": f"bad request: {error}"})
            return
        try:
            response = self.readout(state, questions)
        except Unsupported as error:
            self._send(422, {"error": str(error)})
            return
        except Exception as error:
            self._send(500, {"error": str(error)})
            return
        response["model"] = payload.get("model") or self.readout.model
        self._send(200, response)

    def _chat_completions(self):
        length = int(self.headers.get("content-length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            messages = payload["messages"]
        except Exception as error:
            self._send(400, {"error": f"bad request: {error}"})
            return
        decision = decision_from_messages(messages)
        try:
            if decision is None:
                content = CHAT_NOTICE
                tokens = 0
            else:
                response = self.readout(*decision)
                content = json.dumps(response, ensure_ascii=False)
                tokens = response["usage"]["input_tokens"]
        except Unsupported as error:
            self._send(422, {"error": str(error)})
            return
        except Exception as error:
            self._send(500, {"error": str(error)})
            return
        completion = {
            "id": "chatcmpl-decision",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model") or self.readout.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": tokens, "completion_tokens": 0, "total_tokens": tokens},
        }
        if payload.get("stream"):
            self._send_stream(completion)
            return
        self._send(200, completion)

    def _send_stream(self, completion):
        choice = completion["choices"][0]
        chunks = [
            {
                "id": completion["id"],
                "object": "chat.completion.chunk",
                "created": completion["created"],
                "model": completion["model"],
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": choice["message"]["content"]}, "finish_reason": None}],
            },
            {
                "id": completion["id"],
                "object": "chat.completion.chunk",
                "created": completion["created"],
                "model": completion["model"],
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": completion["usage"],
            },
        ]
        body = "".join(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        data = body.encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/healthz":
            self._send(200, {"status": "ok"})
            return
        self.send_error(404)

    def log_message(self, *args):
        pass


def make_server(host, port, readout):
    handler = type("BoundHandler", (Handler,), {"readout": readout})
    return ThreadingHTTPServer((host, port), handler)


def _env(name, legacy, fallback=None):
    return os.environ.get(name, os.environ.get(legacy, fallback))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-url", default=_env("PQNLD_VLLM_URL", "DECISION_VLLM_URL", "http://127.0.0.1:11542"))
    parser.add_argument("--model", default=_env("PQNLD_MODEL", "DECISION_MODEL", "qwen38-27b"))
    parser.add_argument("--descriptor", default=_env("PQNLD_DESCRIPTOR", "DECISION_DESCRIPTOR"))
    parser.add_argument("--models-dir", default=_env("PQNLD_MODELS_DIR", "DECISION_MODELS_DIR"))
    parser.add_argument("--host", default=_env("PQNLD_HOST", "DECISION_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(_env("PQNLD_PORT", "DECISION_PORT", "11560")))
    parser.add_argument("--temperature", type=float, default=_env("PQNLD_TEMPERATURE", "DECISION_TEMPERATURE"))
    args = parser.parse_args()
    spec = load_spec(args.descriptor or args.model, args.models_dir)
    readout = Readout(vllm_url=args.vllm_url, model=args.model, spec=spec, temperature=args.temperature)
    mode = readout.probe()
    server = make_server(args.host, args.port, readout)
    print(f"decision readout on http://{args.host}:{args.port}/v1/decide -> {args.vllm_url} (readout={mode})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
