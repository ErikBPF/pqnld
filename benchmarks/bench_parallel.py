#!/usr/bin/env python3
"""Measure whether one vLLM engine serves chat and decision requests in parallel.

Fires `--each N` concurrent decision (sidecar `/v1/decide`) and chat
(vLLM `/v1/chat/completions`) requests after a one-of-each warm baseline. A
`parallelism` near 1 means the work serialized; near the request count it
batched. RED for S2: decisions queue behind chat (concurrent latency grows with
N, parallelism stays near 1). GREEN: concurrent latency stays near the baseline
and parallelism grows.

    python3 bench_parallel.py --decide-url http://127.0.0.1:11560 \
        --chat-url http://127.0.0.1:11542 --chat-model qwen38-27b-nvfp4 --each 4

State and prompts carry a nonce so nothing is served from the sidecar cache or
pinned by prefix caching. Run it on the engine host (loopback) or through an
allowed peer; it binds no ports. `--selftest` checks the percentile math only.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TIMEOUT = 600


def _post(url, body, timeout=TIMEOUT):
    started = time.perf_counter()
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return payload, time.perf_counter() - started


def decision_body(nonce):
    return {
        "model": "qwen38-27b-nvfp4",
        "state": f"Run {nonce}: the meeting is on Tuesday at 3pm in room B.",
        "questions": {
            "day": {
                "type": "choice",
                "instructions": "Which day is the meeting?",
                "criteria": {"monday": "Monday", "tuesday": "Tuesday", "friday": "Friday"},
            },
            "urgent": {"type": "noul", "instructions": "Is the meeting urgent?"},
        },
    }


def chat_body(nonce, model):
    return {
        "model": model,
        "messages": [{"role": "user", "content": f"Run {nonce}: reply with one word: the colour of the sky."}],
        "max_tokens": 16,
        "temperature": 0,
    }


def percentile(values, fraction):
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize(latencies):
    return {
        "n": len(latencies),
        "p50": percentile(latencies, 0.50),
        "p95": percentile(latencies, 0.95),
        "max": max(latencies) if latencies else float("nan"),
    }


def run_task(kind, args, nonce):
    if kind == "decision":
        try:
            _post(args.decide_url.rstrip("/") + "/v1/decide", decision_body(nonce))
            return kind, True, None
        except urllib.error.HTTPError as error:
            return kind, error.code < 500, None
        except Exception as error:  # noqa: BLE001 - report, do not abort the benchmark
            return kind, False, str(error)
    try:
        _post(args.chat_url.rstrip("/") + "/v1/chat/completions", chat_body(nonce, args.chat_model))
        return kind, True, None
    except urllib.error.HTTPError as error:
        return kind, error.code < 500, None
    except Exception as error:  # noqa: BLE001
        return kind, False, str(error)


def timed(kind, args, nonce):
    started = time.perf_counter()
    _, ok, error = run_task(kind, args, nonce)
    return kind, ok, error, time.perf_counter() - started


def selftest():
    assert percentile([0.1, 0.2, 0.3, 0.4], 0.5) == 0.3
    assert percentile([0.1, 0.2, 0.3, 0.4], 0.95) == 0.4
    assert percentile([5.0], 0.95) == 5.0
    assert summarize([])["n"] == 0
    offline = argparse.Namespace(decide_url="http://127.0.0.1:1", chat_url="http://127.0.0.1:1", chat_model="x")
    kind, ok, error, seconds = timed("chat", offline, "selftest")
    assert kind == "chat" and not ok and error and seconds >= 0
    print("selftest ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decide-url", default="http://127.0.0.1:11560")
    parser.add_argument("--chat-url", default="http://127.0.0.1:11542")
    parser.add_argument("--chat-model", default="qwen38-27b-nvfp4")
    parser.add_argument("--each", type=int, default=4, help="concurrent requests per kind")
    parser.add_argument("--json", dest="out", help="write the report to this path")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return

    print(f"baseline: one decision, one chat (nonce {time.time_ns()})", flush=True)
    baseline = {}
    for kind in ("decision", "chat"):
        _, ok, error, seconds = timed(kind, args, f"baseline-{kind}-{time.time_ns()}")
        if not ok:
            print(f"  {kind} baseline FAILED: {error}", file=sys.stderr)
            raise SystemExit(1)
        baseline[kind] = seconds
        print(f"  {kind}: {seconds:.3f} s", flush=True)

    total = 2 * args.each
    print(f"concurrent: {args.each} decisions + {args.each} chats ({total} in flight)", flush=True)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=total) as pool:
        futures = []
        for index in range(args.each):
            futures.append(pool.submit(timed, "decision", args, f"d-{index}-{time.time_ns()}"))
            futures.append(pool.submit(timed, "chat", args, f"c-{index}-{time.time_ns()}"))
        results = [future.result() for future in futures]
    wall = time.perf_counter() - started

    latencies = {"decision": [], "chat": []}
    failures = []
    for kind, ok, error, seconds in results:
        if ok:
            latencies[kind].append(seconds)
        else:
            failures.append((kind, error))

    report = {
        "each": args.each,
        "baseline_seconds": baseline,
        "concurrent": {kind: summarize(values) for kind, values in latencies.items()},
        "wall_seconds": wall,
        "throughput_per_s": len([1 for _, ok, _, _ in results if ok]) / wall,
        "parallelism": sum(seconds for _, ok, _, seconds in results if ok) / wall,
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()