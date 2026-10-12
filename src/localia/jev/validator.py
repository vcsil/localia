from __future__ import annotations

from .contracts import (
    DecisionStatus,
    JevRequest,
    PolicyResult,
    RouteDecision,
)
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
        suitable = registry.suitable_routes(
            policy.allowed_routes,
            candidate.required_capabilities,
        )

        if candidate.status is DecisionStatus.NO_SUITABLE_ROUTE:
            if candidate.route is not None:
                raise DecisionValidationError(
                    "no_suitable_route candidate unexpectedly contains a route"
                )
            if suitable:
                names = ", ".join(route.value for route in suitable)
                raise DecisionValidationError(
                    "semantic router claimed no suitable route, but these "
                    f"allowed routes satisfy the declared capabilities: {names}"
                )
            return

        if candidate.route is None:
            raise DecisionValidationError("routed candidate is missing a route")

        spec = registry.get(candidate.route)

        if not spec.enabled:
            raise DecisionValidationError(
                f"route is disabled: {candidate.route.value}"
            )

        if candidate.route not in policy.allowed_routes:
            raise DecisionValidationError(
                f"route violates policy: {candidate.route.value}"
            )

        if (
            spec.max_context_tokens is not None
            and request.context_tokens_estimate > spec.max_context_tokens
        ):
            raise DecisionValidationError(
                f"route context limit exceeded: {candidate.route.value}"
            )

        missing = candidate.required_capabilities - spec.capabilities
        if missing:
            names = ", ".join(sorted(item.value for item in missing))
            raise DecisionValidationError(
                f"route {candidate.route.value} lacks required capabilities: {names}"
            )
