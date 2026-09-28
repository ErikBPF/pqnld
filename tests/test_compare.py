import importlib.util
import pathlib
import unittest

spec = importlib.util.spec_from_file_location("compare", pathlib.Path(__file__).resolve().parents[1] / "benchmarks/compare.py")
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class ComparisonTest(unittest.TestCase):
    def test_failures_count_and_json_is_not_probability_evidence(self):
        rows = [{"status": "ok", "correct": True, "latency_s": 1},
                {"status": "error", "latency_s": 2}]
        result = compare.summarize(rows)
        self.assertEqual(result["accuracy_all"], 0.5)
        self.assertEqual(result["coverage"], 0.5)
        self.assertIsNone(result["nll_answered"])

    def test_probability_metrics_and_invalid_distributions(self):
        result = compare.evaluate({"choice": "b", "probabilities": {"a": .25, "b": .75}}, "b", ["a", "b"])
        self.assertAlmostEqual(result["brier"], .125)
        for probabilities in ({"a": .5}, {"a": .8, "b": .8}, {"a": float("nan"), "b": .5}):
            with self.assertRaises(ValueError):
                compare.evaluate({"choice": "b", "probabilities": probabilities}, "b", ["a", "b"])

    def test_zero_probability_has_infinite_nll_not_silent_clipping(self):
        result = compare.evaluate({"choice": "a", "probabilities": {"a": 1., "b": 0.}}, "b", ["a", "b"])
        self.assertEqual(result["nll"], "infinity")
