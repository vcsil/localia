from __future__ import annotations

from .contracts import DecisionType, JevDecision, JevRequest
from .policy import PolicyEngine
from .registry import RouteRegistry
from .semantic import SemanticRouter, StubSemanticRouter
from .validator import DecisionValidator


class Jev:
    """Generic decision harness.

    V0 capability: route_task. Future capabilities can reuse the same shell.
    """

    def __init__(
        self,
        *,
        registry: RouteRegistry | None = None,
        policy_engine: PolicyEngine | None = None,
        semantic_router: SemanticRouter | None = None,
        validator: DecisionValidator | None = None,
    ) -> None:
        self.registry = registry or RouteRegistry()
        self.policy_engine = policy_engine or PolicyEngine()
        self.semantic_router = semantic_router or StubSemanticRouter()
        self.validator = validator or DecisionValidator()

    def decide(self, request: JevRequest) -> JevDecision:
        if request.decision_type is not DecisionType.ROUTE_TASK:
            raise NotImplementedError(
                f"decision type not implemented in V0: {request.decision_type.value}"
            )

        policy = self.policy_engine.evaluate(request, self.registry)
        if not policy.allowed_routes:
            raise RuntimeError(
                "Jev cannot route this request: deterministic policy left no allowed route"
            )

        candidate = self.semantic_router.decide(request, policy, self.registry)
        self.validator.validate(request, policy, candidate, self.registry)

        return JevDecision(
            decision_type=request.decision_type,
            route=candidate.route,
            task_type=candidate.task_type,
            confidence=candidate.confidence,
            reason=candidate.reason,
            approval_required=policy.approval_required,
            allowed_routes=tuple(sorted(policy.allowed_routes, key=lambda item: item.value)),
            policy_reasons=policy.reasons,
        )
