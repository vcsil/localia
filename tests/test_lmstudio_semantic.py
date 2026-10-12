import unittest

from localia.jev import JevRequest, Route
from localia.jev.contracts import Capability, DecisionStatus
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


def payload(
    *,
    route,
    task_type="general",
    required_capabilities=None,
    reason="routing test",
):
    return {
        "route": route,
        "task_type": task_type,
        "required_capabilities": (
            required_capabilities
            if required_capabilities is not None
            else ["general"]
        ),
        "reason": reason,
    }


class LMStudioSemanticRouterTests(unittest.TestCase):
    def _policy_and_registry(self, *, codex=True, claude=False):
        registry = RouteRegistry()
        registry.set_enabled(Route.CODEX, codex)
        registry.set_enabled(Route.CLAUDE, claude)
        request = JevRequest(task="Corrija os testes deste repositório")
        policy = PolicyEngine().evaluate(request, registry)
        return request, policy, registry

    def test_schema_does_not_ask_model_for_status(self):
        request, policy, registry = self._policy_and_registry()
        backend = FakeBackend(
            payload(
                route="codex",
                task_type="repository_edit",
                required_capabilities=["repository_edit"],
            )
        )

        LMStudioSemanticRouter(backend).decide(request, policy, registry)

        schema = backend.calls[0]["response_format"]
        self.assertNotIn("status", schema["properties"])
        self.assertNotIn("status", schema["required"])

    def test_single_suitable_route_is_resolved_deterministically(self):
        registry = RouteRegistry()
        request = JevRequest(task="Explique cache L3")
        policy = PolicyEngine().evaluate(request, registry)
        backend = FakeBackend(
            payload(
                route="none",
                task_type="general_explanation",
                required_capabilities=["general"],
                reason="model incorrectly preferred none",
            )
        )

        decision = LMStudioSemanticRouter(backend).decide(
            request, policy, registry
        )

        self.assertEqual(decision.status, DecisionStatus.ROUTED)
        self.assertEqual(decision.route, Route.LOCAL_FAST)
        self.assertEqual(
            decision.required_capabilities,
            frozenset({Capability.GENERAL}),
        )

    def test_single_capability_compatible_route_overrides_bad_preference(self):
        request, policy, registry = self._policy_and_registry()
        backend = FakeBackend(
            payload(
                route="local_fast",
                task_type="repository_edit",
                required_capabilities=["repository_edit"],
                reason="bad route preference",
            )
        )

        decision = LMStudioSemanticRouter(backend).decide(
            request, policy, registry
        )

        self.assertEqual(decision.status, DecisionStatus.ROUTED)
        self.assertEqual(decision.route, Route.CODEX)

    def test_no_suitable_route_is_derived_from_capability_gate(self):
        registry = RouteRegistry()
        request = JevRequest(task="Edite este repositório")
        policy = PolicyEngine().evaluate(request, registry)
        backend = FakeBackend(
            payload(
                route="local_fast",
                task_type="repository_edit",
                required_capabilities=["repository_edit"],
                reason="model tried local_fast",
            )
        )

        decision = LMStudioSemanticRouter(backend).decide(
            request, policy, registry
        )

        self.assertEqual(
            decision.status,
            DecisionStatus.NO_SUITABLE_ROUTE,
        )
        self.assertIsNone(decision.route)

    def test_multiple_suitable_routes_use_model_preference(self):
        registry = RouteRegistry()
        registry.set_enabled(Route.CLAUDE, True)
        request = JevRequest(task="Explique cache L3")
        policy = PolicyEngine().evaluate(request, registry)
        backend = FakeBackend(
            payload(
                route="local_fast",
                task_type="general_explanation",
                required_capabilities=["general"],
            )
        )

        decision = LMStudioSemanticRouter(backend).decide(
            request, policy, registry
        )

        self.assertEqual(decision.route, Route.LOCAL_FAST)

    def test_none_is_rejected_when_multiple_routes_are_suitable(self):
        registry = RouteRegistry()
        registry.set_enabled(Route.CLAUDE, True)
        request = JevRequest(task="Explique cache L3")
        policy = PolicyEngine().evaluate(request, registry)
        backend = FakeBackend(
            payload(
                route="none",
                task_type="general_explanation",
                required_capabilities=["general"],
            )
        )

        with self.assertRaises(SemanticRouterError):
            LMStudioSemanticRouter(backend).decide(
                request, policy, registry
            )

    def test_schema_enumerates_only_policy_allowed_routes_plus_none(self):
        request, policy, registry = self._policy_and_registry()
        backend = FakeBackend(
            payload(
                route="codex",
                task_type="repository_edit",
                required_capabilities=["repository_edit"],
            )
        )

        LMStudioSemanticRouter(backend).decide(request, policy, registry)

        enum_values = backend.calls[0]["response_format"]["properties"]["route"]["enum"]
        self.assertEqual(
            set(enum_values),
            {route.value for route in policy.allowed_routes} | {"none"},
        )
        self.assertNotIn("api_fallback", enum_values)
        self.assertNotIn("local_heavy", enum_values)

    def test_prediction_profile_is_temperature_zero_cpu8(self):
        request, policy, registry = self._policy_and_registry()
        backend = FakeBackend(
            payload(
                route="codex",
                task_type="repository_edit",
                required_capabilities=["repository_edit"],
            )
        )

        LMStudioSemanticRouter(backend).decide(request, policy, registry)

        config = backend.calls[0]["config"]
        self.assertEqual(config["temperature"], 0.0)
        self.assertEqual(config["cpuThreads"], 8)
        self.assertLessEqual(config["maxTokens"], 256)

    def test_user_task_is_marked_untrusted(self):
        request, policy, registry = self._policy_and_registry()
        backend = FakeBackend(
            payload(
                route="codex",
                task_type="repository_edit",
                required_capabilities=["repository_edit"],
            )
        )

        LMStudioSemanticRouter(backend).decide(request, policy, registry)

        call = backend.calls[0]
        self.assertIn("UNTRUSTED DATA", call["system_prompt"])
        self.assertIn("untrusted user data", call["user_prompt"])
        self.assertIn(request.task, call["user_prompt"])

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
                **payload(
                    route="codex",
                    task_type="repository_edit",
                    required_capabilities=["repository_edit"],
                ),
                "status": "routed",
            }
        )

        with self.assertRaises(SemanticRouterError):
            LMStudioSemanticRouter(backend).decide(
                request, policy, registry
            )


if __name__ == "__main__":
    unittest.main()
