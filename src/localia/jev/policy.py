from __future__ import annotations

from .contracts import ActionRisk, JevRequest, PolicyResult, PrivacyLevel
from .registry import RouteRegistry


class PolicyEngine:
    """Deterministic constraints that semantic models cannot override."""

    def evaluate(self, request: JevRequest, registry: RouteRegistry) -> PolicyResult:
        allowed = {spec.route for spec in registry.enabled()}
        reasons: list[str] = []

        if request.privacy is PrivacyLevel.LOCAL_ONLY or not request.cloud_allowed:
            cloud_routes = {spec.route for spec in registry.all() if not spec.local}
            removed = allowed & cloud_routes
            allowed -= cloud_routes
            if removed:
                reasons.append("cloud routes removed by privacy/cloud policy")

        for route in tuple(allowed):
            spec = registry.get(route)
            if (
                spec.max_context_tokens is not None
                and request.context_tokens_estimate > spec.max_context_tokens
            ):
                allowed.remove(route)
                reasons.append(
                    f"{route.value} removed: estimated context "
                    f"{request.context_tokens_estimate} > {spec.max_context_tokens}"
                )

        approval_required = request.action_risk is ActionRisk.DESTRUCTIVE
        if approval_required:
            reasons.append("destructive action requires human approval")

        if not allowed:
            reasons.append("no route remains after deterministic policy")

        return PolicyResult(
            allowed_routes=frozenset(allowed),
            approval_required=approval_required,
            reasons=tuple(reasons),
        )
