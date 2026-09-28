"""Paired choice-only experiment. JSONL stdout; synthetic smoke is not Decision Index."""

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

from pqnld.decision import ModelSpec, Readout, render_prompt


def evaluate(answer, expected, keys):
    if answer.get("choice") not in keys:
        raise ValueError("choice outside criteria")
    result = {"correct": answer["choice"] == expected}
    if "probabilities" in answer:
        p = answer["probabilities"]
        if set(p) != set(keys) or not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in p.values()) or abs(sum(p.values()) - 1) > 1e-6:
            raise ValueError("invalid probability distribution")
        result["nll"] = -math.log(p[expected]) if p[expected] else "infinity"
        result["brier"] = sum((p[k] - (k == expected)) ** 2 for k in keys)
    return result


def summarize(rows):
    answered = [r for r in rows if r["status"] == "ok"]
    scored = [r for r in answered if "nll" in r]
    return {"cases": len(rows), "coverage": len(answered) / len(rows) if rows else None,
            "accuracy_all": sum(r["correct"] for r in answered) / len(rows) if rows else None,
            "probability_cases": len(scored),
            "nll_answered": ("infinity" if any(r["nll"] == "infinity" for r in scored) else statistics.mean(r["nll"] for r in scored)) if scored else None,
            "brier_answered": statistics.mean(r["brier"] for r in scored) if scored else None,
            "latency_p50_all_s": statistics.median(r["latency_s"] for r in rows) if rows else None,
            "latency_p50_answered_s": statistics.median(r["latency_s"] for r in answered) if answered else None}


def smoke():
    for n in (2, 3, 10, 26):
        keys = [f"item-{i}" for i in range(n)]
        expected = keys[n // 2]
        for reverse in (False, True):
            yield {"id": f"select-{n}-{reverse}", "state": f"The selected item is {expected}.",
                   "question": {"type": "choice", "instructions": "Which item is selected?",
                                "criteria": {k: k for k in (keys[::-1] if reverse else keys)}},
                   "expected": expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="qwen38-27b-nvfp4")
    parser.add_argument("--rows", type=Path, help="JSONL: id, state, question (choice), expected; not raw kit rows")
    parser.add_argument("--arms", nargs="+", choices=["exact", "json", "reason", "echo"], default=["exact", "json", "reason", "echo"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reason-tokens", type=int, default=128)
    args = parser.parse_args()
    raw = args.rows.read_bytes() if args.rows else None
    cases = [json.loads(line) for line in raw.splitlines() if line.strip()] if raw else list(smoke())
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("nonempty cases with unique IDs required")
    for c in cases:
        if c["question"]["type"] != "choice" or c["expected"] not in c["question"]["criteria"]:
            raise ValueError("choice questions with known expected keys required")
    emit = lambda row: print(json.dumps(row, allow_nan=False), flush=True)
    digest = hashlib.sha256(raw or json.dumps(cases, sort_keys=True).encode()).hexdigest()
    emit({"kind": "manifest", "model": args.model, "dataset_sha256": digest,
          "dataset": str(args.rows) if args.rows else "synthetic-selection-smoke-v1",
          "arms": args.arms, "seed": args.seed, "reason_tokens": args.reason_tokens,
          "result_cache": False, "scope": "paired choice experiment; not official Decision Index scoring"})
    exact = Readout(args.url, args.model, ModelSpec(readout="lettered", specific_token_scores=True), timeout=60, cache_size=0)
    echo = Readout(args.url, args.model, ModelSpec(readout="echo"), timeout=60, cache_size=0, batch=1)
    rng = random.Random(args.seed)
    rows = []
    for case in cases:
        arms = list(args.arms)
        rng.shuffle(arms)
        for arm in arms:
            start = time.perf_counter()
            row = {"kind": "result", "id": case["id"], "arm": arm}
            try:
                state, q = case["state"], case["question"]
                if arm in ("json", "reason"):
                    body = {"model": args.model, "messages": [{"role": "user", "content": render_prompt(state, q)}],
                            "temperature": 0, "max_tokens": args.reason_tokens if arm == "reason" else 128,
                            "chat_template_kwargs": {"enable_thinking": False}}
                    if arm == "json":
                        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "decision", "strict": True,
                            "schema": {"type": "object", "properties": {"choice": {"type": "string", "enum": list(q["criteria"])}},
                                       "required": ["choice"], "additionalProperties": False}}}
                    else:
                        body["messages"][0]["content"] += "\nAnalyze the evidence briefly before choosing."
                    response = exact._post("/v1/chat/completions", body)
                    choice = response["choices"][0]
                    row["finish_reason"] = choice["finish_reason"]
                    content = choice["message"]["content"]
                    if arm == "json":
                        if choice["finish_reason"] != "stop":
                            raise ValueError("JSON generation did not finish")
                        answer = json.loads(content)
                        answer.pop("probabilities", None)
                    else:
                        state = {"original_state": state, "model_analysis": content}
                if arm != "json":
                    reader = echo if arm == "echo" else exact
                    answer = reader(state, {"q": q})["answers"]["q"]
                row.update(evaluate(answer, case["expected"], list(q["criteria"])), status="ok", answer=answer)
            except Exception as error:
                row.update(status="error", error=f"{type(error).__name__}: {error}")
            row["latency_s"] = time.perf_counter() - start
            rows.append(row)
            emit(row)
    emit({"kind": "summary", "arms": {arm: summarize([r for r in rows if r["arm"] == arm]) for arm in args.arms}})


if __name__ == "__main__":
    main()
