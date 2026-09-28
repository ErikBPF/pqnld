"""Readout correctness regressions; synthetic transport, real decision logic.

Run: PYTHONPATH=src python3 -m unittest discover -s tests -p test_readout_integrity.py -v
These checks establish correctness, not model accuracy or calibration.
"""

import math
import unittest
from unittest.mock import Mock

from pqnld.decision import ModelSpec, Readout, Unsupported


class ReadoutIntegrityTest(unittest.TestCase):
    def readout(self, entries, mode="lettered"):
        readout = Readout("http://unused", "fixture", ModelSpec(readout=mode))
        readout._post = Mock(return_value={
            "choices": [{"logprobs": {"content": [{"top_logprobs": [
                {"token": token, "logprob": score} for token, score in entries
            ]}]}}],
        })
        return readout

    def questions(self, keys=("red", "blue")):
        return {"color": {"type": "choice", "instructions": "Which color?",
                          "criteria": {key: key for key in keys}}}

    def test_complete_labels_preserve_probability_ratio(self):
        readout = self.readout([("a", -2.0), ("b", -1.0)])
        answer = readout("A color.", self.questions())["answers"]["color"]
        self.assertEqual(answer["choice"], "blue")
        self.assertAlmostEqual(answer["probabilities"]["blue"], 1 / (1 + math.exp(-1)))
        readout("A color.", self.questions())
        self.assertEqual(readout._post.call_count, 1)

    def test_words_do_not_override_exact_label_scores(self):
        readout = self.readout([("because", -0.1), ("a", -1.0), ("b", -3.0)])
        answer = readout("A color.", self.questions())["answers"]["color"]
        self.assertEqual(answer["choice"], "red")
        self.assertAlmostEqual(answer["probabilities"]["red"], 1 / (1 + math.exp(-2)))

    def test_missing_label_is_refused_instead_of_floored(self):
        readout = self.readout([("a", -0.1), ("!", -1.0)])
        with self.assertRaises(Unsupported):
            readout("A color.", self.questions())

    def test_no_label_evidence_is_refused_instead_of_uniform(self):
        readout = self.readout([("!", -0.1)])
        with self.assertRaises(Unsupported):
            readout("A color.", self.questions())

    def test_probe_does_not_accept_a_word_as_a_letter(self):
        readout = self.readout([("blue", -0.1), ("a", -2.0), ("b", -3.0), ("c", -4.0)], "auto")
        self.assertEqual(readout.probe(), "echo")

    def test_probe_falls_back_when_top_k_has_no_complete_label_set(self):
        readout = self.readout([("blue", -0.1)], "auto")
        self.assertEqual(readout.probe(), "echo")

    def test_cached_reordered_options_match_a_fresh_readout(self):
        entries = [("a", -0.1), ("b", -3.0)]
        cached = self.readout(entries)
        cached("A color.", self.questions())
        reordered = self.questions(("blue", "red"))
        actual = cached("A color.", reordered)
        expected = self.readout(entries)("A color.", reordered)
        self.assertEqual(actual["answers"], expected["answers"])

    def test_specific_token_scores_bypass_top_k_without_constraining_sampling(self):
        readout = self.readout([])
        readout.spec.specific_token_scores = True
        def post(path, body):
            if path == "/tokenize":
                return {"tokens": [ord(body["prompt"])]}
            self.assertEqual(body["logprob_token_ids"], [97, 98])
            self.assertEqual(body["top_logprobs"], 0)
            self.assertNotIn("allowed_token_ids", body)
            return {"choices": [{"logprobs": {"content": [{"token": "token_id:98", "top_logprobs": [
                {"token": "token_id:97", "logprob": -2.0},
                {"token": "token_id:98", "logprob": -1.0},
            ]}]}}]}
        readout._post = Mock(side_effect=post)
        answer = readout("state", self.questions())["answers"]["color"]
        self.assertAlmostEqual(answer["probabilities"]["blue"], 1 / (1 + math.exp(-1)))

    def test_specific_scores_reject_multitoken_labels(self):
        readout = self.readout([])
        readout.spec.specific_token_scores = True
        readout._post = Mock(return_value={"tokens": [1, 2]})
        with self.assertRaises(Unsupported):
            readout("state", self.questions())

    def test_echo_refuses_overlapping_keys_before_inference(self):
        readout = self.readout([], "echo")
        with self.assertRaisesRegex(Unsupported, "prefix"):
            readout("item-13 is selected", self.questions(("item-1", "item-13")))
        readout._post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
