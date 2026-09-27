#!/usr/bin/env python3
"""Mixed-load benchmark for the Apollo engine: long-context chat + parallel decisions.

Phases (--phases, default all):
  single    one long-context conversation alone -> single-conversation TPS
  pair      two long-context conversations alone -> aggregate TPS baseline
  mixed     two conversations + --decisions parallel decisions -> chat TPS under
            decision load, decision latency/parallelism, decisions overlap chats
  recovery  two conversations alone again -> chat TPS recovers to the pair baseline

Each conversation sends an ~`--context-tokens` token prompt (distinct nonce, so
no prefix-cache help) and generates `--chat-max-tokens` output tokens.
Conversations stream, so the report separates TTFT (prefill) from decode TPS;
`tps`/`tps_aggregate` are decode-only and would otherwise be dominated by the
~95k-token prefill.

Stdlib only. Run against live Apollo from inside the engine container:

    python3 bench_mixed.py --context-tokens 100000 --json /cache/mixed.json
    python3 bench_mixed.py --selftest
"""

import argparse
import json
import math
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

LOCK = threading.Lock()
ACTIVE_CONVERSATIONS = 0


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return ordered[int(k)]
    return ordered[lo] * (hi - k) + ordered[hi] * (k - lo)


def post(url, body, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    start = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return True, json.load(response), time.time() - start
    except urllib.error.HTTPError as error:
        return False, error.read().decode("utf-8", "replace"), time.time() - start
    except Exception as error:
        return False, str(error), time.time() - start


WORDS = ("baseline note record observation subject context state request answer option "
         "question decision channel pattern sample detail summary session topic reference "
         "value shift window range field report index signal factor boundary segment").split()


def filler(nonce, approx_tokens):
    """A ~approx_tokens-token body of plain words (~1 token each), distinct per nonce."""
    words = ["Reference", nonce + "."]
    i = 0
    while len(words) < approx_tokens:
        words.append(WORDS[i % len(WORDS)])
        i += 1
    return " ".join(words)


def stream_chat(url, body, timeout):
    """Stream a chat completion; return (ok, info) with TTFT and decode-only TPS.

    Decode TPS excludes the prefill: with a ~95k-token prompt the prompt
    processing time dominates the wall clock, so dividing tokens by the total
    would understate generation speed.
    """
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    start = time.time()
    ttft = None
    deltas = 0
    usage = {}
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                delta = (choices[0].get("delta") or {}) if choices else {}
                if any(delta.get(field) for field in ("content", "reasoning_content", "reasoning")):
                    if ttft is None:
                        ttft = time.time() - start
                    deltas += 1
        ok = True
    except urllib.error.HTTPError as failure:
        error = failure.read().decode("utf-8", "replace")[:300]
        ok = False
    except Exception as failure:
        error = str(failure)[:300]
        ok = False
    end = time.time()
    tokens = usage.get("completion_tokens") or deltas
    prompt_tokens = usage.get("prompt_tokens") or 0
    decode = (end - start - ttft) if ttft is not None else None
    return ok, {
        "ok": ok,
        "seconds": end - start,
        "ttft": ttft,
        "decode_seconds": decode,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": tokens,
        "tps": (tokens / decode) if ok and ttft is not None and decode and decode > 0 else None,
        "prefill_tps": (prompt_tokens / ttft) if ok and ttft and prompt_tokens else None,
        "error": error,
    }


def conversation(args, index, results):
    global ACTIVE_CONVERSATIONS
    nonce = "conv%d-%d" % (index, time.time_ns())
    prompt = (
        filler(nonce, args.context_tokens)
        + "\n\nUsing only the notes above, write a long essay about the recurring "
        "subject. Give it a title and at least five sections."
    )
    body = {
        "model": args.chat_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.chat_max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    with LOCK:
        ACTIVE_CONVERSATIONS += 1
    ok, info = stream_chat(args.chat_url, body, args.timeout)
    with LOCK:
        ACTIVE_CONVERSATIONS -= 1
    results.append(info)


def decision(args, index, results):
    nonce = "dec%d-%d" % (index, time.time_ns())
    body = {
        "state": "Nonce %s: the sky on a clear day is blue." % nonce,
        "questions": {
            "colour": {
                "type": "choice",
                "instructions": "Which colour is the sky?",
                "criteria": {"red": "red", "blue": "blue", "green": "green"},
            }
        },
    }
    with LOCK:
        active = ACTIVE_CONVERSATIONS
    ok, payload, seconds = post(args.decide_url, body, args.timeout)
    choice = None
    if ok:
        choice = (payload.get("answers", {}).get("colour", {}) or {}).get("choice")
    results.append({
        "ok": ok,
        "seconds": seconds,
        "choice": choice,
        "conversations_active": active,
        "error": None if ok else str(payload)[:200],
    })


def warm(args):
    post(args.chat_url, {
        "model": args.chat_model,
        "messages": [{"role": "user", "content": "Reply with the word ready."}],
        "max_tokens": 4, "temperature": 0,
    }, args.timeout)
    decision(args, -1, [])


def summarize_conversations(rows):
    done = [r for r in rows if r["ok"]]
    tps = [r["tps"] for r in done if r["tps"]]
    decode_max = max((r["decode_seconds"] for r in done if r["decode_seconds"]), default=None)
    wall = max((r["seconds"] for r in done), default=None)
    return {
        "count": len(rows),
        "ok": len(done),
        "prompt_tokens_each": [r["prompt_tokens"] for r in done],
        "ttft_each": [r["ttft"] for r in done],
        "prefill_tps_each": [r["prefill_tps"] for r in done],
        "completion_tokens_total": sum(r["completion_tokens"] for r in done),
        "wall_seconds": wall,
        "decode_seconds_max": decode_max,
        "tps_each": tps,
        "tps_aggregate": (sum(r["completion_tokens"] for r in done) / decode_max) if done and decode_max else None,
        "errors": [r["error"] for r in rows if not r["ok"]],
    }


def summarize_decisions(rows, workers):
    done = [r for r in rows if r["ok"]]
    latencies = [r["seconds"] for r in done]
    wall = max(latencies, default=0)
    return {
        "count": len(rows),
        "ok": len(done),
        "failures": [r["error"] for r in rows if not r["ok"]],
        "p50": percentile(latencies, 0.5),
        "p95": percentile(latencies, 0.95),
        "max": max(latencies, default=None),
        "throughput_per_s": (len(done) / wall) if wall else None,
        "parallelism": (sum(latencies) / wall) if wall else None,
        "workers": workers,
        "with_conversations_active": sum(1 for r in done if r["conversations_active"] > 0),
        "choices": sorted({r["choice"] for r in done if r["choice"]}),
    }


def run_phase(args, label, n_conv, with_decisions):
    conversations, decisions = [], []
    print("[%s] start: %d conversation(s)%s" % (
        label, n_conv, " + %d decisions" % args.decisions if with_decisions else ""), flush=True)
    with ThreadPoolExecutor(max_workers=max(n_conv, args.workers)) as pool:
        conv_futures = [pool.submit(conversation, args, i, conversations) for i in range(n_conv)]
        if with_decisions:
            time.sleep(args.lead)
            dec_futures = [pool.submit(decision, args, i, decisions) for i in range(args.decisions)]
            for future in dec_futures:
                future.result()
        for future in conv_futures:
            future.result()
    conv = summarize_conversations(conversations)
    dec = summarize_decisions(decisions, args.workers) if with_decisions else None
    print("[%s] chat tps=%s decisions=%s" % (
        label,
        round(conv["tps_aggregate"], 2) if conv["tps_aggregate"] else None,
        ("p50=%.2f p95=%.2f %.1f/s par=%.2f" % (dec["p50"], dec["p95"],
         dec["throughput_per_s"], dec["parallelism"])) if dec else "-"), flush=True)
    return conv, dec


def selftest():
    assert abs(percentile([0.1, 0.2, 0.3, 0.4], 0.5) - 0.25) < 1e-6
    assert abs(percentile([0.1, 0.2, 0.3, 0.4], 0.95) - 0.385) < 1e-6
    rows = [{"ok": True, "seconds": 2.0, "ttft": 1.0, "decode_seconds": 1.0, "prompt_tokens": 90000,
             "prefill_tps": 90000.0, "completion_tokens": 10, "tps": 10.0, "error": None}]
    assert summarize_conversations(rows)["tps_aggregate"] == 10.0
    decisions = [{"ok": True, "seconds": 0.5, "choice": "blue", "conversations_active": 1, "error": None},
                 {"ok": False, "seconds": 0.1, "choice": None, "conversations_active": 0, "error": "x"}]
    summary = summarize_decisions(decisions, 16)
    assert summary["ok"] == 1 and summary["failures"] == ["x"] and summary["parallelism"] == 1.0
    assert len(filler("n", 500).split()) >= 500
    print("selftest ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chat-url", default="http://100.77.14.27:11542/v1/chat/completions")
    parser.add_argument("--decide-url", default="http://100.77.14.27:11560/v1/decide")
    parser.add_argument("--chat-model", default="qwen38-27b-nvfp4")
    parser.add_argument("--context-tokens", type=int, default=95000,
                        help="approximate prompt tokens per conversation (~1 word each)")
    parser.add_argument("--chat-max-tokens", type=int, default=1024)
    parser.add_argument("--decisions", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--lead", type=float, default=3.0, help="seconds before firing decisions")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--phases", default="single,pair,mixed,recovery")
    parser.add_argument("--json", default=None)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return

    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    print("warming", flush=True)
    warm(args)

    report = {}
    if "single" in phases:
        report["single_conversation"], _ = run_phase(args, "single", 1, False)
    if "pair" in phases:
        report["pair_conversations"], _ = run_phase(args, "pair", 2, False)
    if "mixed" in phases:
        report["mixed_conversations"], report["mixed_decisions"] = run_phase(args, "mixed", 2, True)
    if "recovery" in phases:
        report["recovery_conversations"], _ = run_phase(args, "recovery", 2, False)

    pair = report.get("pair_conversations", {}).get("tps_aggregate")
    mixed = report.get("mixed_conversations", {}).get("tps_aggregate")
    recovery = report.get("recovery_conversations", {}).get("tps_aggregate")
    report["analysis"] = {
        "chat_tps_retention_under_decisions": (mixed / pair) if pair and mixed else None,
        "chat_tps_recovered": (recovery / pair) if pair and recovery else None,
        "decisions_parallel": (report.get("mixed_decisions", {}).get("parallelism") or 0) > 1.5,
        "single_conversation_tps": report.get("single_conversation", {}).get("tps_aggregate"),
        "prefill_tps_each": report.get("pair_conversations", {}).get("prefill_tps_each"),
        "context_reached_each": report.get("pair_conversations", {}).get("prompt_tokens_each"),
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2)
        print("wrote %s" % args.json, flush=True)


if __name__ == "__main__":
    main()
