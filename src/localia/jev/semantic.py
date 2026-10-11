from __future__ import annotations

from typing import Protocol

from .contracts import JevRequest, PolicyResult, Route, RouteDecision
from .registry import RouteRegistry


class SemanticRouter(Protocol):
    """Pluggable semantic engine. Next milestone: LM Studio/Qwen."""

    def decide(
        self,
        request: JevRequest,
        policy: PolicyResult,
        registry: RouteRegistry,
    ) -> RouteDecision:
        ...


class StubSemanticRouter:
    """Temporary deterministic router used only to prove the architecture."""

    CODE_WORDS = (
        "código",
        "codigo",
        "python",
        "javascript",
        "typescript",
        "repository",
        "repositório",
        "repositorio",
        "debug",
        "refator",
        "testes",
        "tests",
        "bug",
        "git",
        "branch",
    )

    LONG_CONTEXT_WORDS = (
        "muitos artigos",
        "contexto longo",
        "documentos longos",
        "sintetize os documentos",
    )

    def decide(
        self,
        request: JevRequest,
        policy: PolicyResult,
        registry: RouteRegistry,
    ) -> RouteDecision:
        if not policy.allowed_routes:
            raise RuntimeError("no allowed routes")

        text = request.task.casefold()

        if Route.CODEX in policy.allowed_routes and any(word in text for word in self.CODE_WORDS):
            return RouteDecision(
                route=Route.CODEX,
                task_type="code",
                confidence=0.85,
                reason="stub heuristic detected a code/repository task",
            )

        if Route.CLAUDE in policy.allowed_routes and (
            request.context_tokens_estimate > 20_000
            or any(word in text for word in self.LONG_CONTEXT_WORDS)
        ):
            return RouteDecision(
                route=Route.CLAUDE,
                task_type="long_context",
                confidence=0.80,
                reason="stub heuristic detected a long-context task",
            )

        if Route.LOCAL_FAST in policy.allowed_routes:
            return RouteDecision(
                route=Route.LOCAL_FAST,
                task_type="general",
                confidence=0.70,
                reason="stub defaulted to the local fast route",
            )

        route = sorted(policy.allowed_routes, key=lambda item: item.value)[0]
        return RouteDecision(
            route=route,
            task_type="fallback",
            confidence=0.50,
            reason="stub selected the first remaining allowed route",
        )
