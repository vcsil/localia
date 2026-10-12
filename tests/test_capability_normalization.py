import unittest

from localia.jev.contracts import (
    Capability,
    DecisionStatus,
    Route,
    RouteDecision,
    RouteSpec,
    normalize_capabilities,
)
from localia.jev.registry import RouteRegistry


class CapabilityNormalizationTests(unittest.TestCase):
    def test_repository_edit_implies_repository_read(self):
        normalized = normalize_capabilities(
            {Capability.REPOSITORY_EDIT}
        )

        self.assertEqual(
            normalized,
            frozenset(
                {
                    Capability.REPOSITORY_EDIT,
                    Capability.REPOSITORY_READ,
                }
            ),
        )

    def test_route_decision_is_normalized_on_construction(self):
        decision = RouteDecision(
            status=DecisionStatus.ROUTED,
            route=Route.CODEX,
            task_type="repository_edit",
            required_capabilities=frozenset(
                {Capability.REPOSITORY_EDIT}
            ),
            confidence=None,
            reason="deterministic test reason",
        )

        self.assertIn(
            Capability.REPOSITORY_READ,
            decision.required_capabilities,
        )

    def test_route_spec_capabilities_are_normalized(self):
        spec = RouteSpec(
            route=Route.CODEX,
            provider="test",
            local=False,
            capabilities=frozenset({Capability.REPOSITORY_EDIT}),
        )

        self.assertIn(Capability.REPOSITORY_READ, spec.capabilities)

    def test_registry_suitability_normalizes_external_requirements(self):
        registry = RouteRegistry()
        suitable = registry.suitable_routes(
            frozenset({Route.CODEX}),
            frozenset({Capability.REPOSITORY_EDIT}),
        )

        self.assertEqual(suitable, (Route.CODEX,))


if __name__ == "__main__":
    unittest.main()
