#!/usr/bin/env python3
"""Checks for the pqnld tester (no network; _chat is stubbed).

    python3 -m unittest discover -s tests -v
"""

import json
import unittest

from fastapi.testclient import TestClient

from pqnld import tester

DECISION = {
    "model": "qwen38-27b-nvfp4",
    "answers": {
        "day": {"type": "choice", "choice": "tuesday", "probabilities": {"monday": 0.0, "tuesday": 1.0, "friday": 0.0}},
        "is_urgent": {"type": "noul", "noul": 0.12},
    },
    "usage": {"input_tokens": 120},
}

REQUEST = {
    "state": "The meeting is on Tuesday at 3pm in room B.",
    "questions": {
        "day": {"type": "choice", "instructions": "Which day?", "criteria": {"monday": "Monday", "tuesday": "Tuesday", "friday": "Friday"}},
        "is_urgent": {"type": "noul", "instructions": "Is it urgent?"},
    },
}


class QwenJevApiTest(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.chat_calls = []

        def fake_chat(content):
            self.sent.append(content)
            return json.dumps(DECISION)

        def fake_request(messages, model, max_tokens, temperature, chat_template_kwargs=None):
            self.chat_calls.append({"messages": messages, "model": model, "max_tokens": max_tokens, "temperature": temperature, "chat_template_kwargs": chat_template_kwargs})
            return {
                "model": model,
                "choices": [{"message": {"role": "assistant", "content": "Hello from the model."}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }

        self._real = tester._chat
        self._real_request = tester._request
        tester._chat = fake_chat
        tester._request = fake_request
        self.client = TestClient(tester.app)

    def tearDown(self):
        tester._chat = self._real
        tester._request = self._real_request

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_openapi_describes_decide(self):
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/decide", schema["paths"])
        self.assertIn("DecisionRequest", schema["components"]["schemas"])

    def test_decide_forwards_and_parses(self):
        response = self.client.post("/decide", json=REQUEST)
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["model"], "qwen38-27b-nvfp4")
        self.assertEqual(body["answers"]["day"]["choice"], "tuesday")
        self.assertAlmostEqual(sum(body["answers"]["day"]["probabilities"].values()), 1.0)
        self.assertAlmostEqual(body["answers"]["is_urgent"]["noul"], 0.12)
        forwarded = json.loads(self.sent[0])
        self.assertEqual(forwarded["model"], tester.MODEL)
        self.assertEqual(forwarded["state"], REQUEST["state"])
        self.assertEqual(set(forwarded["questions"]), {"day", "is_urgent"})
        self.assertEqual(forwarded["questions"]["day"]["criteria"]["tuesday"], "Tuesday")

    def test_non_decision_content_is_502(self):
        tester._chat = lambda content: "I am not JSON"
        response = self.client.post("/decide", json=REQUEST)
        self.assertEqual(response.status_code, 502)

    def test_upstream_failure_is_502(self):
        def boom(content):
            raise tester.HTTPException(502, "upstream 500")

        tester._chat = boom
        response = self.client.post("/decide", json=REQUEST)
        self.assertEqual(response.status_code, 502)

    def test_malformed_request_is_422(self):
        response = self.client.post("/decide", json={"questions": "nope"})
        self.assertEqual(response.status_code, 422)


    def test_complete_returns_text_and_forwards_messages(self):
        response = self.client.post(
            "/complete",
            json={"prompt": "Say hi.", "system": "Be terse.", "model": "qwen-chat", "max_tokens": 32},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["content"], "Hello from the model.")
        self.assertEqual(body["model"], "qwen-chat")
        self.assertEqual(body["usage"]["completion_tokens"], 3)
        call = self.chat_calls[0]
        self.assertEqual(call["model"], "qwen-chat")
        self.assertEqual(call["max_tokens"], 32)
        self.assertEqual(
            call["messages"],
            [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Say hi."}],
        )

    def test_openapi_describes_complete(self):
        schema = self.client.get("/openapi.json").json()
        self.assertIn("/complete", schema["paths"])
        self.assertIn("CompletionRequest", schema["components"]["schemas"])


    def test_complete_defaults_to_the_text_model(self):
        response = self.client.post("/complete", json={"prompt": "Say hi."})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.chat_calls[0]["model"], "qwen-chat")


if __name__ == "__main__":
    unittest.main(verbosity=2)
