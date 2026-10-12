import unittest

from localia.jev import JevRequest, Route
from localia.jev.policy import PolicyEngine
from localia.jev.registry import RouteRegistry
from localia.jev.semantic import (
    LMStudioSemanticRouter,
    SemanticRouterError,
)


class FakeBackend:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def respond_structured(
        self,
        *,
        system_prompt,
        user_prompt,
        response_format,
        config,
    ):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_format": response_format,
                "config": config,
            }
        )
        return dict(self.payload)


class LMStudioSemanticRouterTests(unittest.TestCase):
    def _policy_and_registry(self):
        registry = RouteRegistry()
        registry.set_enabled(Route.CODEX, True)
        registry.set_enabled(Route.CLAUDE, True)
        request = JevRequest(task="Corrija os testes deste repositório")
        policy = PolicyEngine().evaluate(request, registry)
        return request, policy, registry

    def test_valid_structured_decision(self):
        request, policy, registry = self._policy_and_registry()
        backend = FakeBackend(
            {
                "route": "codex",
                "task_type": "code_edit",
                "reason": "Repository coding task.",
            }
        )

        decision = LMStudioSemanticRouter(backend).decide(
            request, policy, registry
        )

        self.assertEqual(decision.route, Route.CODEX)
        self.assertEqual(decision.task_type, "code_edit")
        self.assertIsNone(decision.confidence)

    def test_schema_enumerates_only_policy_allowed_routes(self):
        request, policy, registry = self._policy_and_registry()
        backend = FakeBackend(
            {
                "route": "local_fast",
                "task_type": "general",
                "reason": "Adequate local route.",
            }
        )

        LMStudioSemanticRouter(backend).decide(request, policy, registry)

        enum_values = backend.calls[0]["response_format"]["properties"]["route"]["enum"]
        self.assertEqual(
            set(enum_values),
            {route.value for route in policy.allowed_routes},
        )
        self.assertNotIn("api_fallback", enum_values)
        self.assertNotIn("local_heavy", enum_values)

    def test_prediction_profile_is_temperature_zero_cpu8(self):
        request, policy, registry = self._policy_and_registry()
        backend = FakeBackend(
            {
                "route": "local_fast",
                "task_type": "general",
                "reason": "Adequate local route.",
            }
        )

        LMStudioSemanticRouter(backend).decide(request, policy, registry)

        config = backend.calls[0]["config"]
        self.assertEqual(config["temperature"], 0.0)
        self.assertEqual(config["cpuThreads"], 8)
        self.assertLessEqual(config["maxTokens"], 256)

    def test_user_task_is_marked_untrusted(self):
        request, policy, registry = self._policy_and_registry()
        backend = FakeBackend(
            {
                "route": "local_fast",
                "task_type": "general",
                "reason": "Adequate local route.",
            }
        )

        LMStudioSemanticRouter(backend).decide(request, policy, registry)

        call = backend.calls[0]
        self.assertIn("UNTRUSTED DATA", call["system_prompt"])
        self.assertIn("untrusted user data", call["user_prompt"])
        self.assertIn(request.task, call["user_prompt"])

    def test_forbidden_route_from_backend_is_rejected(self):
        registry = RouteRegistry()
        request = JevRequest(task="private task")
        policy = PolicyEngine().evaluate(request, registry)
        backend = FakeBackend(
            {
                "route": "claude",
                "task_type": "general",
                "reason": "Attempted policy bypass.",
            }
        )

        with self.assertRaises(SemanticRouterError):
            LMStudioSemanticRouter(backend).decide(
                request, policy, registry
            )

    def test_missing_fields_are_rejected(self):
        request, policy, registry = self._policy_and_registry()
        backend = FakeBackend({"route": "codex"})

        with self.assertRaises(SemanticRouterError):
            LMStudioSemanticRouter(backend).decide(
                request, policy, registry
            )


    def test_extra_fields_are_rejected(self):
        request, policy, registry = self._policy_and_registry()
        backend = FakeBackend(
            {
                "route": "codex",
                "task_type": "code_edit",
                "reason": "Repository coding task.",
                "approval_required": False,
            }
        )

        with self.assertRaises(SemanticRouterError):
            LMStudioSemanticRouter(backend).decide(
                request, policy, registry
            )



if __name__ == "__main__":
    unittest.main()
