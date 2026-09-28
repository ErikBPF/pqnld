"""Bounded chat/score overlap diagnostic; requires an otherwise idle engine."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import threading
import time
import urllib.error
import urllib.request


def score_status(response, ids):
    content = response["choices"][0].get("logprobs") or {}
    rows = content.get("content") or []
    scores = {item["token"]: item.get("logprob")
              for item in rows[0].get("top_logprobs", [])} if rows else {}
    return "complete" if all(
        isinstance(scores.get(f"token_id:{token}"), (int, float))
        and math.isfinite(scores[f"token_id:{token}"]) for token in ids
    ) else "incomplete"


def request(url, payload):
    return urllib.request.urlopen(urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}), timeout=45)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="qwen38-27b-nvfp4")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    common = {"model": args.model, "temperature": 0,
              "chat_template_kwargs": {"enable_thinking": False}}
    ids = []
    for label in "abc":
        with request(url + "/tokenize", {"model": args.model, "prompt": label,
                                        "add_special_tokens": False}) as response:
            tokens = json.load(response)["tokens"]
        if len(tokens) != 1 or tokens[0] in ids:
            raise ValueError("labels require distinct single tokens")
        ids.extend(tokens)
    decision = {**common, "messages": [{"role": "user", "content":
                "The sky is blue. Which color is the sky? a: red, b: blue, c: green. Reply with one letter."}],
                "max_tokens": 1, "logprobs": True, "top_logprobs": 0,
                "logprob_token_ids": ids, "return_tokens_as_token_ids": True}
    endpoint = url + "/v1/chat/completions"
    rows = []
    for mode in ("isolated", "chat-no-logprobs", "chat-topk"):
        started, finished = threading.Event(), threading.Event()

        def chat():
            payload = {**common, "stream": True, "max_tokens": 128,
                       "messages": [{"role": "user", "content":
                                     "List integers from 1 to 200, separated by spaces."}]}
            if mode == "chat-topk":
                payload.update(logprobs=True, top_logprobs=3)
            try:
                with request(endpoint, payload) as response:
                    for line in response:
                        if line.startswith(b"data: {"):
                            chunk = json.loads(line[6:])
                            if any(c.get("delta", {}).get("content") for c in chunk.get("choices", [])):
                                started.set()
                return "ok"
            except Exception as error:
                return str(error)
            finally:
                finished.set()

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(chat) if mode != "isolated" else None
            row = {"mode": mode, "chat_started": started.wait(15) if future else False}
            row["chat_active_at_send"] = bool(future and row["chat_started"] and not finished.is_set())
            begin = time.perf_counter()
            try:
                with request(endpoint, decision) as response:
                    body = json.load(response)
                row.update(status=score_status(body, ids), response=body)
            except urllib.error.HTTPError as error:
                with error:
                    row.update(status="http-error", http_status=error.code,
                               error=error.read().decode(errors="replace"))
            row["latency_s"] = time.perf_counter() - begin
            row["chat_active_at_return"] = bool(future and not finished.is_set())
            row["chat_status"] = future.result() if future else "not-run"
            rows.append(row)
    print(json.dumps({"model": args.model, "label_ids": ids, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
