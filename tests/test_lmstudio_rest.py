import unittest

from localia.providers.lmstudio import (
    LMStudioConfigurationError,
    LMStudioRestBackend,
    QWEN_Q4_MODEL_KEY,
)


class FakeAuditor:
    def __init__(self):
        self.calls = 0

    def audit(self):
        self.calls += 1
        return {"quantization_verified": True}


class CapturePost:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, url, payload, timeout_seconds):
        self.calls.append((url, payload, timeout_seconds))
        return self.response


def completion_response(
    *,
    content='{"route":"codex","task_type":"code","reason":"repository task"}',
    reasoning_content=None,
    finish_reason="stop",
):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": QWEN_Q4_MODEL_KEY,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning_content,
                },
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
    }


class LMStudioStructuredRestBackendTests(unittest.TestCase):
    def test_uses_openai_chat_completions_with_json_schema(self):
        auditor = FakeAuditor()
        post = CapturePost(completion_response())
        backend = LMStudioRestBackend(
            auditor=auditor,
            http_post=post,
        )

        schema = {
            "type": "object",
            "properties": {
                "route": {"type": "string", "enum": ["local_fast", "codex"]},
                "task_type": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["route", "task_type", "reason"],
            "additionalProperties": False,
        }

        result = backend.respond_structured(
            system_prompt="system",
            user_prompt="user",
            response_format=schema,
            config={"temperature": 0.0, "maxTokens": 128},
        )

        self.assertEqual(result["route"], "codex")
        self.assertEqual(auditor.calls, 1)

        url, payload, _ = post.calls[0]
        self.assertTrue(url.endswith("/v1/chat/completions"))
        self.assertEqual(payload["model"], QWEN_Q4_MODEL_KEY)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["max_tokens"], 128)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertTrue(
            payload["response_format"]["json_schema"]["strict"]
        )
        self.assertEqual(
            payload["response_format"]["json_schema"]["schema"],
            schema,
        )

    def test_audit_runs_only_once_per_backend_instance(self):
        auditor = FakeAuditor()
        post = CapturePost(completion_response())
        backend = LMStudioRestBackend(
            auditor=auditor,
            http_post=post,
        )

        for _ in range(2):
            backend.respond_structured(
                system_prompt="system",
                user_prompt="user",
                response_format={"type": "object"},
                config={},
            )

        self.assertEqual(auditor.calls, 1)

    def test_accepts_schema_json_from_content(self):
        backend = LMStudioRestBackend(
            auditor=FakeAuditor(),
            http_post=CapturePost(completion_response()),
        )

        result = backend.respond_structured(
            system_prompt="system",
            user_prompt="user",
            response_format={"type": "object"},
            config={},
        )

        self.assertEqual(result["route"], "codex")

    def test_accepts_schema_json_from_reasoning_content_when_content_empty(self):
        backend = LMStudioRestBackend(
            auditor=FakeAuditor(),
            http_post=CapturePost(
                completion_response(
                    content="",
                    reasoning_content=(
                        '{"route":"codex","task_type":"code",'
                        '"reason":"repository task"}'
                    ),
                )
            ),
        )

        result = backend.respond_structured(
            system_prompt="system",
            user_prompt="user",
            response_format={"type": "object"},
            config={},
        )

        self.assertEqual(result["route"], "codex")

    def test_rejects_yaml_like_free_form_output(self):
        backend = LMStudioRestBackend(
            auditor=FakeAuditor(),
            http_post=CapturePost(
                completion_response(
                    content=(
                        "task_type: codex\n"
                        "reason: repository task"
                    )
                )
            ),
        )

        with self.assertRaises(LMStudioConfigurationError):
            backend.respond_structured(
                system_prompt="system",
                user_prompt="user",
                response_format={"type": "object"},
                config={},
            )

    def test_rejects_partial_json(self):
        backend = LMStudioRestBackend(
            auditor=FakeAuditor(),
            http_post=CapturePost(
                completion_response(
                    content='{"route":"codex","task_type":"code"'
                )
            ),
        )

        with self.assertRaises(LMStudioConfigurationError):
            backend.respond_structured(
                system_prompt="system",
                user_prompt="user",
                response_format={"type": "object"},
                config={},
            )

    def test_rejects_length_finish_reason(self):
        backend = LMStudioRestBackend(
            auditor=FakeAuditor(),
            http_post=CapturePost(
                completion_response(finish_reason="length")
            ),
        )

        with self.assertRaises(LMStudioConfigurationError):
            backend.respond_structured(
                system_prompt="system",
                user_prompt="user",
                response_format={"type": "object"},
                config={},
            )

    def test_rejects_conflicting_json_channels(self):
        backend = LMStudioRestBackend(
            auditor=FakeAuditor(),
            http_post=CapturePost(
                completion_response(
                    content=(
                        '{"route":"local_fast","task_type":"general",'
                        '"reason":"simple"}'
                    ),
                    reasoning_content=(
                        '{"route":"codex","task_type":"code",'
                        '"reason":"repository"}'
                    ),
                )
            ),
        )

        with self.assertRaises(LMStudioConfigurationError):
            backend.respond_structured(
                system_prompt="system",
                user_prompt="user",
                response_format={"type": "object"},
                config={},
            )


if __name__ == "__main__":
    unittest.main()
