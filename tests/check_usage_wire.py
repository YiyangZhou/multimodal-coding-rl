"""Responses SSE compatibility without inventing unreported token counts."""

import copy
import json
import unittest

from check_gym_protocol import ProtocolTests
from nemo_gym.responses_streaming import synthesize_responses_sse


def events(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


class UsageWireTests(unittest.TestCase):
    def test_native_http_omits_unreported_details(self):
        fixture = ProtocolTests()
        fixture.setUp()
        response = fixture.client.post("/v1/responses", json={"input": "Toy task.", "stream": True})
        self.assertEqual(response.status_code, 200)
        for event in events(response.text):
            if "response" in event:
                self.assertEqual(event["response"]["usage"], {
                    "input_tokens": 2, "output_tokens": 1, "total_tokens": 3,
                })

    def test_detail_groups_preserve_known_counts_and_internal_unknowns(self):
        for cached, reasoning in ((None, None), (0, 0), (4, None), (None, 3), (4, 3)):
            with self.subTest(cached=cached, reasoning=reasoning):
                original = {
                    "id": "resp_toy", "output": [], "status": "completed",
                    "usage": {
                        "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                        "input_tokens_details": {"cached_tokens": cached},
                        "output_tokens_details": {"reasoning_tokens": reasoning},
                    },
                }
                source = copy.deepcopy(original)
                result = events("".join(synthesize_responses_sse(source)))[-1]["response"]
                self.assertEqual(source, original)
                for group, counter, value in (
                    ("input_tokens_details", "cached_tokens", cached),
                    ("output_tokens_details", "reasoning_tokens", reasoning),
                ):
                    if value is None:
                        self.assertNotIn(group, result["usage"])
                    else:
                        self.assertEqual(result["usage"][group], {counter: value})

    def test_empty_details_are_omitted_and_output_identity_is_unchanged(self):
        response = {
            "id": "resp_toy", "status": "completed", "metadata": {"lineage": "toy"},
            "output": [{"id": "call_toy", "call_id": "call_toy", "type": "function_call",
                        "name": "exec_command", "arguments": '{"cmd":"echo toy"}'}],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                      "input_tokens_details": {}, "output_tokens_details": None},
        }
        original = copy.deepcopy(response)
        result = events("".join(synthesize_responses_sse(response)))
        self.assertEqual(response, original)
        self.assertEqual(result[-1]["response"], {
            **original, "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        })
        self.assertEqual(result[1]["item"], original["output"][0])


if __name__ == "__main__":
    unittest.main()
