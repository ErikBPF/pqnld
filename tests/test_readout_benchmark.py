import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "bench_readout", Path(__file__).resolve().parents[1] / "benchmarks/bench_readout.py"
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class ReadoutBenchmarkTest(unittest.TestCase):
    def test_refusals_count_against_coverage_and_overall_accuracy(self):
        result = benchmark.summarize([
            {"status": "ok", "correct": True, "latency_s": 1.0},
            {"status": "ok", "correct": False, "latency_s": 3.0},
            {"status": "unsupported", "latency_s": 2.0},
        ])
        self.assertEqual(result["answered"], 2)
        self.assertAlmostEqual(result["coverage"], 2 / 3)
        self.assertAlmostEqual(result["accuracy_all_cases"], 1 / 3)
        self.assertEqual(result["accuracy_answered"], 0.5)
        self.assertEqual(result["latency_p50_s"], 2.0)

    def test_no_answers_does_not_report_perfect_accuracy(self):
        result = benchmark.summarize([{"status": "error", "latency_s": 1.0}])
        self.assertEqual(result["coverage"], 0)
        self.assertIsNone(result["accuracy_answered"])


if __name__ == "__main__":
    unittest.main()
