"""Small synthetic live readout smoke test, not a calibration benchmark.

PYTHONPATH=src python3 benchmarks/bench_readout.py --url http://localhost:11542
"""

import argparse
import json
import statistics
import time

from pqnld.decision import ModelSpec, Readout, Unsupported


def summarize(rows):
    answered = [row for row in rows if row["status"] == "ok"]
    correct = sum(row["correct"] for row in answered)
    return {
        "cases": len(rows), "answered": len(answered),
        "coverage": len(answered) / len(rows) if rows else None,
        "accuracy_all_cases": correct / len(rows) if rows else None,
        "accuracy_answered": correct / len(answered) if answered else None,
        "latency_p50_s": statistics.median(row["latency_s"] for row in rows) if rows else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="qwen38-27b-nvfp4")
    parser.add_argument("--specific-token-scores", action="store_true")
    args = parser.parse_args()
    readout = Readout(args.url, args.model, ModelSpec(readout="lettered", specific_token_scores=args.specific_token_scores), timeout=30, cache_size=0)
    rows = []
    for count in (2, 3, 10, 20, 26):
        keys = [f"item-{i}" for i in range(count)]
        expected = keys[count // 2]
        for reverse in (False, True):
            ordered = list(reversed(keys)) if reverse else keys
            question = {"q": {"type": "choice", "instructions": "Which item is selected?",
                              "criteria": {key: key for key in ordered}}}
            start = time.perf_counter()
            row = {"options": count, "reversed": reverse, "expected": expected}
            try:
                answer = readout(f"The selected item is {expected}.", question)["answers"]["q"]
                row.update(status="ok", correct=answer["choice"] == expected, answer=answer)
            except Unsupported as error:
                row.update(status="unsupported", error=str(error))
            except Exception as error:
                row.update(status="error", error=str(error))
            row["latency_s"] = time.perf_counter() - start
            rows.append(row)
    print(json.dumps({"model": args.model, "readout": "lettered", "result_cache": False,
                      "specific_token_scores": args.specific_token_scores,
                      "scope": "synthetic smoke only; not held-out calibration evidence",
                      "summary": summarize(rows), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
