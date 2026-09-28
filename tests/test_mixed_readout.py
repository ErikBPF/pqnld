import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "mixed_readout", Path(__file__).resolve().parents[1] / "benchmarks/mixed_readout.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MixedReadoutTest(unittest.TestCase):
    def test_missing_or_nonfinite_scores_fail_the_gate(self):
        response = {"choices": [{"logprobs": {"content": [{"top_logprobs": [
            {"token": "token_id:64", "logprob": -0.1},
            {"token": "token_id:65", "logprob": -2.0},
        ]}]}}]}
        self.assertEqual(module.score_status(response, [64, 65]), "complete")
        self.assertEqual(module.score_status(response, [64, 65, 66]), "incomplete")
        response["choices"][0]["logprobs"]["content"][0]["top_logprobs"][1]["logprob"] = None
        self.assertEqual(module.score_status(response, [64, 65]), "incomplete")
        response["choices"][0]["logprobs"]["content"] = []
        self.assertEqual(module.score_status(response, [64, 65]), "incomplete")
