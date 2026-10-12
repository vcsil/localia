import unittest

from localia.jev import (
    ActionRisk,
    Capability,
    DecisionStatus,
    Jev,
    JevRequest,
    PrivacyLevel,
    Route,
)
from localia.jev.contracts import RouteDecision
from localia.jev.validator import DecisionValidationError


class MaliciousPolicyBypassRouter:
    def __init__(self):
        self.calls = 0

    def decide(self, request, policy, registry):
        self.calls += 1
        return RouteDecision(
            status=DecisionStatus.ROUTED,
            route=Route.CLAUDE,
            task_type="general",
            required_capabilities=frozenset({Capability.GENERAL}),
            confidence=1.0,
            reason="tries to bypass deterministic policy",
        )


class FalseNoSuitableRouter:
    def decide(self, request, policy, registry):
        return RouteDecision(
            status=DecisionStatus.NO_SUITABLE_ROUTE,
            route=None,
            task_type="general",
            required_capabilities=frozenset({Capability.GENERAL}),
            confidence=None,
            reason="incorrectly claims no route can handle general text",
        )


class OrchestratorTests(unittest.TestCase):
    def test_default_general_task_routes_to_local_fast(self) -> None:
        decision = Jev().decide(JevRequest(task="Explique cache L3"))

        self.assertEqual(decision.status, DecisionStatus.ROUTED)
        self.assertEqual(decision.route, Route.LOCAL_FAST)
        self.assertIn(Capability.GENERAL, decision.required_capabilities)

    def test_code_repository_task_routes_to_codex_when_enabled(self) -> None:
        jev = Jev()
        jev.registry.set_enabled(Route.CODEX, True)

        decision = jev.decide(
            JevRequest(
                task="Edite os arquivos deste repositório e rode os testes Python"
            )
        )

        self.assertEqual(decision.status, DecisionStatus.ROUTED)
        self.assertEqual(decision.route, Route.CODEX)
        self.assertIn(
            Capability.REPOSITORY_EDIT,
            decision.required_capabilities,
        )
        self.assertIn(
            Capability.CODE_EXECUTION,
            decision.required_capabilities,
        )

    def test_local_only_repository_edit_returns_no_suitable_route(self) -> None:
        jev = Jev()
        jev.registry.set_enabled(Route.CODEX, True)

        decision = jev.decide(
            JevRequest(
                task="Edite os arquivos deste repositório e rode os testes Python",
                privacy=PrivacyLevel.LOCAL_ONLY,
                cloud_allowed=False,
            )
        )

        self.assertEqual(
            decision.status,
            DecisionStatus.NO_SUITABLE_ROUTE,
        )
        self.assertIsNone(decision.route)
        self.assertIn(
            Capability.REPOSITORY_EDIT,
            decision.required_capabilities,
        )
        self.assertIn(
            "cloud routes removed by privacy/cloud policy",
            decision.policy_reasons,
        )

    def test_destructive_approval_is_not_owned_by_semantic_router(self) -> None:
        jev = Jev()
        jev.registry.set_enabled(Route.CODEX, True)

        decision = jev.decide(
            JevRequest(
                task="Edite este repositório e delete a branch remota",
                action_risk=ActionRisk.DESTRUCTIVE,
            )
        )

        self.assertTrue(decision.approval_required)

    def test_validator_blocks_router_policy_bypass(self) -> None:
        router = MaliciousPolicyBypassRouter()
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

    def test_validator_rejects_false_no_suitable_claim(self) -> None:
        jev = Jev(semantic_router=FalseNoSuitableRouter())

        with self.assertRaises(DecisionValidationError):
            jev.decide(JevRequest(task="Explain something simple"))

    def test_zero_policy_routes_returns_no_suitable_without_semantic_call(self) -> None:
        jev = Jev()
        jev.registry.set_enabled(Route.LOCAL_FAST, False)

        decision = jev.decide(JevRequest(task="hello"))

        self.assertEqual(
            decision.status,
            DecisionStatus.NO_SUITABLE_ROUTE,
        )
        self.assertIsNone(decision.route)
        self.assertEqual(decision.task_type, "policy_blocked")
        self.assertEqual(decision.required_capabilities, ())


if __name__ == "__main__":
    unittest.main()
