from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .contracts import Capability, Route, RouteSpec


DEFAULT_ROUTES: tuple[RouteSpec, ...] = (
    RouteSpec(
        route=Route.LOCAL_FAST,
        provider="lmstudio:qwen3.5-9b-q4_k_m",
        local=True,
        enabled=True,
        max_context_tokens=4096,
        capabilities=frozenset(
            {
                Capability.GENERAL,
                Capability.CLASSIFICATION,
                Capability.EXTRACTION,
                Capability.SUMMARIZATION,
                Capability.STRUCTURED_OUTPUT,
                Capability.CODE_REASONING,
            }
        ),
    ),
    RouteSpec(
        route=Route.LOCAL_HEAVY,
        provider="local_heavy:not_configured",
        local=True,
        enabled=False,
        max_context_tokens=None,
        capabilities=frozenset(
            {
                Capability.GENERAL,
                Capability.REASONING,
                Capability.STRUCTURED_OUTPUT,
                Capability.CODE_REASONING,
            }
        ),
    ),
    RouteSpec(
        route=Route.CODEX,
        provider="codex_cli",
        local=False,
        enabled=False,
        max_context_tokens=None,
        capabilities=frozenset(
            {
                Capability.CODE_REASONING,
                Capability.REPOSITORY_READ,
                Capability.REPOSITORY_EDIT,
                Capability.CODE_EXECUTION,
            }
        ),
    ),
    RouteSpec(
        route=Route.CLAUDE,
        provider="claude_code",
        local=False,
        enabled=False,
        max_context_tokens=None,
        capabilities=frozenset(
            {
                Capability.GENERAL,
                Capability.REASONING,
                Capability.LONG_CONTEXT,
                Capability.CODE_REASONING,
                Capability.REPOSITORY_READ,
                Capability.REPOSITORY_EDIT,
            }
        ),
    ),
    RouteSpec(
        route=Route.API_FALLBACK,
        provider="openrouter",
        local=False,
        enabled=False,
        max_context_tokens=None,
        capabilities=frozenset(
            {
                Capability.GENERAL,
                Capability.REASONING,
                Capability.STRUCTURED_OUTPUT,
                Capability.CODE_REASONING,
            }
        ),
    ),
)


class RouteRegistry:
    """Source of truth for route existence, availability and capabilities."""

    def __init__(self, routes: Iterable[RouteSpec] = DEFAULT_ROUTES) -> None:
        items = tuple(routes)
        self._routes = {item.route: item for item in items}
        if len(self._routes) != len(items):
            raise ValueError("duplicate routes in registry")

    def get(self, route: Route) -> RouteSpec:
        try:
            return self._routes[route]
        except KeyError as exc:
            raise KeyError(f"unknown route: {route.value}") from exc

    def all(self) -> tuple[RouteSpec, ...]:
        return tuple(self._routes.values())

    def enabled(self) -> tuple[RouteSpec, ...]:
        return tuple(item for item in self._routes.values() if item.enabled)

    def set_enabled(self, route: Route, enabled: bool) -> None:
        self._routes[route] = replace(self.get(route), enabled=bool(enabled))

    def suitable_routes(
        self,
        routes: frozenset[Route],
        required_capabilities: frozenset[Capability],
    ) -> tuple[Route, ...]:
        return tuple(
            sorted(
                (
                    route
                    for route in routes
                    if required_capabilities <= self.get(route).capabilities
                ),
                key=lambda item: item.value,
            )
        )
