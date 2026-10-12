import unittest

from localia.jev import (
    Capability,
    DecisionStatus,
    JevRequest,
    Route,
)
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

    def test_valid_structured_routed_decision(self):
        request, policy, registry = self._policy_and_registry()
        backend = FakeBackend(
            {
                "status": "routed",
                "route": "codex",
                "task_type": "code_repository_editing",
                "required_capabilities": [
                    "repository_read",
                    "repository_edit",
                ],
                "reason": "Repository coding task.",
            }
        )

        decision = LMStudioSemanticRouter(backend).decide(
            request, policy, registry
        )

        self.assertEqual(decision.status, DecisionStatus.ROUTED)
        self.assertEqual(decision.route, Route.CODEX)
        self.assertIn(
            Capability.REPOSITORY_EDIT,
            decision.required_capabilities,
        )
        self.assertIsNone(decision.confidence)

    def test_valid_no_suitable_route_decision(self):
        registry = RouteRegistry()
        request = JevRequest(task="Edite os arquivos deste repositório")
        policy = PolicyEngine().evaluate(request, registry)
        backend = FakeBackend(
            {
                "status": "no_suitable_route",
                "route": "none",
                "task_type": "repository_edit",
                "required_capabilities": ["repository_edit"],
                "reason": "No allowed route can edit repository files.",
            }
        )

        decision = LMStudioSemanticRouter(backend).decide(
            request, policy, registry
        )

        self.assertEqual(
            decision.status,
            DecisionStatus.NO_SUITABLE_ROUTE,
        )
        self.assertIsNone(decision.route)

    def test_schema_enumerates_only_policy_routes_plus_none(self):
        request, policy, registry = self._policy_and_registry()
        backend = FakeBackend(
            {
                "status": "routed",
                "route": "local_fast",
                "task_type": "general",
                "required_capabilities": ["general"],
                "reason": "Adequate local route.",
            }
        )

        LMStudioSemanticRouter(backend).decide(request, policy, registry)

        enum_values = backend.calls[0]["response_format"]["properties"]["route"]["enum"]
        self.assertEqual(
            set(enum_values),
            {route.value for route in policy.allowed_routes} | {"none"},
        )
        self.assertNotIn("api_fallback", enum_values)
        self.assertNotIn("local_heavy", enum_values)

    def test_schema_capabilities_are_closed_enum(self):
        request, policy, registry = self._policy_and_registry()
        backend = FakeBackend(
            {
                "status": "routed",
                "route": "local_fast",
                "task_type": "general",
                "required_capabilities": ["general"],
                "reason": "Adequate local route.",
            }
        )

        LMStudioSemanticRouter(backend).decide(request, policy, registry)

        capability_enum = (
            backend.calls[0]["response_format"]["properties"]
            ["required_capabilities"]["items"]["enum"]
        )
        self.assertEqual(
            set(capability_enum),
            {item.value for item in Capability},
        )

    def test_route_catalog_exposes_typed_capabilities(self):
        request, policy, registry = self._policy_and_registry()
        backend = FakeBackend(
            {
                "status": "routed",
                "route": "codex",
                "task_type": "repository_edit",
                "required_capabilities": ["repository_edit"],
                "reason": "Repository task.",
            }
        )

        LMStudioSemanticRouter(backend).decide(request, policy, registry)

        user_prompt = backend.calls[0]["user_prompt"]
        self.assertIn("repository_edit", user_prompt)
        self.assertIn("code_execution", user_prompt)

    def test_prediction_profile_is_temperature_zero_cpu8(self):
        request, policy, registry = self._policy_and_registry()
        backend = FakeBackend(
            {
                "status": "routed",
                "route": "local_fast",
                "task_type": "general",
                "required_capabilities": ["general"],
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
                "status": "routed",
                "route": "local_fast",
                "task_type": "general",
                "required_capabilities": ["general"],
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
                "status": "routed",
                "route": "claude",
                "task_type": "general",
                "required_capabilities": ["general"],
                "reason": "Attempted policy bypass.",
            }
        )

        with self.assertRaises(SemanticRouterError):
            LMStudioSemanticRouter(backend).decide(
                request, policy, registry
            )

    def test_no_suitable_requires_none_route(self):
        request, policy, registry = self._policy_and_registry()
        backend = FakeBackend(
            {
                "status": "no_suitable_route",
                "route": "codex",
                "task_type": "repository_edit",
                "required_capabilities": ["repository_edit"],
                "reason": "Inconsistent output.",
            }
        )

        with self.assertRaises(SemanticRouterError):
            LMStudioSemanticRouter(backend).decide(
                request, policy, registry
            )

    def test_unknown_capability_is_rejected(self):
        request, policy, registry = self._policy_and_registry()
        backend = FakeBackend(
            {
                "status": "routed",
                "route": "codex",
                "task_type": "magic",
                "required_capabilities": ["telepathy"],
                "reason": "Invalid capability.",
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
                "status": "routed",
                "route": "codex",
                "task_type": "code_edit",
                "required_capabilities": ["repository_edit"],
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
