import unittest

from localia.jev import ActionRisk, JevRequest, PrivacyLevel, Route
from localia.jev.policy import PolicyEngine
from localia.jev.registry import RouteRegistry


class PolicyEngineTests(unittest.TestCase):
    def test_local_only_removes_cloud_routes(self) -> None:
        registry = RouteRegistry()
        registry.set_enabled(Route.CODEX, True)
        registry.set_enabled(Route.CLAUDE, True)

        result = PolicyEngine().evaluate(
            JevRequest(
                task="private task",
                privacy=PrivacyLevel.LOCAL_ONLY,
                cloud_allowed=False,
            ),
            registry,
        )

        self.assertIn(Route.LOCAL_FAST, result.allowed_routes)
        self.assertNotIn(Route.CODEX, result.allowed_routes)
        self.assertNotIn(Route.CLAUDE, result.allowed_routes)

    def test_destructive_action_requires_approval(self) -> None:
        result = PolicyEngine().evaluate(
            JevRequest(
                task="delete a remote branch",
                action_risk=ActionRisk.DESTRUCTIVE,
            ),
            RouteRegistry(),
        )
        self.assertTrue(result.approval_required)

    def test_context_can_remove_local_fast(self) -> None:
        result = PolicyEngine().evaluate(
            JevRequest(task="large context task", context_tokens_estimate=5000),
            RouteRegistry(),
        )
        self.assertNotIn(Route.LOCAL_FAST, result.allowed_routes)


if __name__ == "__main__":
    unittest.main()
