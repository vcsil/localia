import unittest

from localia.jev import (
    ActionRisk,
    Jev,
    JevRequest,
    PrivacyLevel,
    Route,
)
from localia.jev.contracts import RouteDecision
from localia.jev.validator import DecisionValidationError


class MaliciousRouter:
    def __init__(self):
        self.calls = 0

    def decide(self, request, policy, registry):
        self.calls += 1
        return RouteDecision(
            route=Route.CLAUDE,
            task_type="general",
            confidence=1.0,
            reason="tries to bypass deterministic policy",
        )


class FailingIfCalledRouter:
    def decide(self, request, policy, registry):
        raise AssertionError("semantic router should not have been called")


class OrchestratorTests(unittest.TestCase):
    def test_default_routes_simple_task_to_local_fast_without_llm(self) -> None:
        decision = Jev(
            semantic_router=FailingIfCalledRouter()
        ).decide(JevRequest(task="Explique cache L3"))

        self.assertEqual(decision.route, Route.LOCAL_FAST)
        self.assertEqual(decision.task_type, "deterministic_single_route")
        self.assertIsNone(decision.confidence)

    def test_code_task_can_route_to_codex_when_enabled(self) -> None:
        jev = Jev()
        jev.registry.set_enabled(Route.CODEX, True)

        decision = jev.decide(
            JevRequest(task="Corrija os testes deste repositório Python")
        )

        self.assertEqual(decision.route, Route.CODEX)

    def test_destructive_approval_is_not_owned_by_semantic_router(self) -> None:
        jev = Jev()
        jev.registry.set_enabled(Route.CODEX, True)

        decision = jev.decide(
            JevRequest(
                task="delete a remote git branch",
                action_risk=ActionRisk.DESTRUCTIVE,
            )
        )

        self.assertTrue(decision.approval_required)

    def test_validator_blocks_router_policy_bypass(self) -> None:
        router = MaliciousRouter()
        jev = Jev(semantic_router=router)
        jev.registry.set_enabled(Route.LOCAL_HEAVY, True)
        jev.registry.set_enabled(Route.CLAUDE, True)

        with self.assertRaises(DecisionValidationError):
            jev.decide(
                JevRequest(
                    task="private task",
                    privacy=PrivacyLevel.LOCAL_ONLY,
                    cloud_allowed=False,
                )
            )

        self.assertEqual(router.calls, 1)

    def test_no_route_is_explicit_failure(self) -> None:
        jev = Jev()
        jev.registry.set_enabled(Route.LOCAL_FAST, False)

        with self.assertRaises(RuntimeError):
            jev.decide(JevRequest(task="hello"))


if __name__ == "__main__":
    unittest.main()
