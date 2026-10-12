import unittest

from localia.jev import Capability, DecisionStatus, JevRequest, Route
from localia.jev.contracts import RouteDecision
from localia.jev.policy import PolicyEngine
from localia.jev.registry import RouteRegistry
from localia.jev.validator import DecisionValidationError, DecisionValidator


class DecisionValidatorCapabilityTests(unittest.TestCase):
    def setUp(self):
        self.registry = RouteRegistry()
        self.validator = DecisionValidator()

    def test_rejects_route_missing_declared_capability(self):
        request = JevRequest(task="Edit repository")
        policy = PolicyEngine().evaluate(request, self.registry)
        candidate = RouteDecision(
            status=DecisionStatus.ROUTED,
            route=Route.LOCAL_FAST,
            task_type="repository_edit",
            required_capabilities=frozenset(
                {Capability.REPOSITORY_EDIT}
            ),
            confidence=None,
            reason="bad candidate",
        )

        with self.assertRaises(DecisionValidationError):
            self.validator.validate(
                request,
                policy,
                candidate,
                self.registry,
            )

    def test_accepts_verified_no_suitable_route(self):
        request = JevRequest(task="Edit repository")
        policy = PolicyEngine().evaluate(request, self.registry)
        candidate = RouteDecision(
            status=DecisionStatus.NO_SUITABLE_ROUTE,
            route=None,
            task_type="repository_edit",
            required_capabilities=frozenset(
                {Capability.REPOSITORY_EDIT}
            ),
            confidence=None,
            reason="no local repository editor",
        )

        self.validator.validate(
            request,
            policy,
            candidate,
            self.registry,
        )

    def test_rejects_no_suitable_when_capable_route_exists(self):
        request = JevRequest(task="General question")
        policy = PolicyEngine().evaluate(request, self.registry)
        candidate = RouteDecision(
            status=DecisionStatus.NO_SUITABLE_ROUTE,
            route=None,
            task_type="general",
            required_capabilities=frozenset({Capability.GENERAL}),
            confidence=None,
            reason="false negative",
        )

        with self.assertRaises(DecisionValidationError):
            self.validator.validate(
                request,
                policy,
                candidate,
                self.registry,
            )


if __name__ == "__main__":
    unittest.main()
