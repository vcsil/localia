from __future__ import annotations

from .contracts import (
    DecisionStatus,
    DecisionType,
    JevDecision,
    JevRequest,
)
from .policy import PolicyEngine
from .registry import RouteRegistry
from .semantic import SemanticRouter, StubSemanticRouter
from .validator import DecisionValidator


class Jev:
    """Generic decision harness.

    Deterministic policy runs first. Semantic routing then identifies the
    capabilities the task actually requires and proposes either a capable
    allowed route or NO_SUITABLE_ROUTE. The validator is authoritative.

    Important: a single allowed route is no longer auto-selected. Permission
    and capability are different questions; the semantic layer must still
    determine whether that one route can actually perform the task.
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
                f"decision type not implemented in V0: "
                f"{request.decision_type.value}"
            )

        policy = self.policy_engine.evaluate(request, self.registry)

        if not policy.allowed_routes:
            return JevDecision(
                decision_type=request.decision_type,
                status=DecisionStatus.NO_SUITABLE_ROUTE,
                route=None,
                task_type="policy_blocked",
                required_capabilities=(),
                confidence=None,
                reason=(
                    "No route remains after deterministic policy; semantic "
                    "routing was skipped."
                ),
                approval_required=policy.approval_required,
                allowed_routes=(),
                policy_reasons=policy.reasons,
            )

        candidate = self.semantic_router.decide(
            request,
            policy,
            self.registry,
        )

        self.validator.validate(
            request,
            policy,
            candidate,
            self.registry,
        )

        return JevDecision(
            decision_type=request.decision_type,
            status=candidate.status,
            route=candidate.route,
            task_type=candidate.task_type,
            required_capabilities=tuple(
                sorted(
                    candidate.required_capabilities,
                    key=lambda item: item.value,
                )
            ),
            confidence=candidate.confidence,
            reason=candidate.reason,
            approval_required=policy.approval_required,
            allowed_routes=tuple(
                sorted(policy.allowed_routes, key=lambda item: item.value)
            ),
            policy_reasons=policy.reasons,
        )
