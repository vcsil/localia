from __future__ import annotations

from .contracts import JevRequest, PolicyResult, RouteDecision
from .registry import RouteRegistry


class DecisionValidationError(RuntimeError):
    pass


class DecisionValidator:
    """Final deterministic gate between semantic decision and execution."""

    def validate(
        self,
        request: JevRequest,
        policy: PolicyResult,
        candidate: RouteDecision,
        registry: RouteRegistry,
    ) -> None:
        spec = registry.get(candidate.route)

        if not spec.enabled:
            raise DecisionValidationError(f"route is disabled: {candidate.route.value}")

        if candidate.route not in policy.allowed_routes:
            raise DecisionValidationError(f"route violates policy: {candidate.route.value}")

        if (
            spec.max_context_tokens is not None
            and request.context_tokens_estimate > spec.max_context_tokens
        ):
            raise DecisionValidationError(
                f"route context limit exceeded: {candidate.route.value}"
            )
