from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class DecisionType(str, Enum):
    """Decision capabilities exposed by Jev."""

    ROUTE_TASK = "route_task"


class DecisionStatus(str, Enum):
    ROUTED = "routed"
    NO_SUITABLE_ROUTE = "no_suitable_route"


class Route(str, Enum):
    LOCAL_FAST = "local_fast"
    LOCAL_HEAVY = "local_heavy"
    CODEX = "codex"
    CLAUDE = "claude"
    API_FALLBACK = "api_fallback"


class Capability(str, Enum):
    """Operational abilities a route can actually provide."""

    GENERAL = "general"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    STRUCTURED_OUTPUT = "structured_output"
    CODE_REASONING = "code_reasoning"
    REPOSITORY_READ = "repository_read"
    REPOSITORY_EDIT = "repository_edit"
    CODE_EXECUTION = "code_execution"
    LONG_CONTEXT = "long_context"
    REASONING = "reasoning"


# Deterministic semantic implications between operational capabilities.
# Keep this deliberately small: implications must be true by definition, not
# merely common in practice.
CAPABILITY_IMPLICATIONS: dict[Capability, frozenset[Capability]] = {
    Capability.REPOSITORY_EDIT: frozenset({Capability.REPOSITORY_READ}),
}


def normalize_capabilities(
    capabilities: Iterable[Capability],
) -> frozenset[Capability]:
    """Return the transitive closure of deterministic capability implications."""

    normalized = set(capabilities)
    changed = True
    while changed:
        changed = False
        for capability in tuple(normalized):
            implied = CAPABILITY_IMPLICATIONS.get(capability, frozenset())
            before = len(normalized)
            normalized.update(implied)
            if len(normalized) != before:
                changed = True
    return frozenset(normalized)


def format_capabilities(capabilities: Iterable[Capability]) -> str:
    values = sorted(item.value for item in capabilities)
    return ", ".join(values) if values else "none"


class PrivacyLevel(str, Enum):
    NORMAL = "normal"
    LOCAL_ONLY = "local_only"


class ActionRisk(str, Enum):
    NONE = "none"
    REVERSIBLE = "reversible"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True, slots=True)
class JevRequest:
    """Normalized decision request sent to Jev."""

    task: str
    decision_type: DecisionType = DecisionType.ROUTE_TASK
    privacy: PrivacyLevel = PrivacyLevel.NORMAL
    cloud_allowed: bool = True
    context_tokens_estimate: int = 0
    action_risk: ActionRisk = ActionRisk.NONE
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("task cannot be empty")
        if self.context_tokens_estimate < 0:
            raise ValueError("context_tokens_estimate cannot be negative")


@dataclass(frozen=True, slots=True)
class RouteSpec:
    route: Route
    provider: str
    local: bool
    enabled: bool = True
    max_context_tokens: int | None = None
    capabilities: frozenset[Capability] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capabilities",
            normalize_capabilities(self.capabilities),
        )


@dataclass(frozen=True, slots=True)
class PolicyResult:
    allowed_routes: frozenset[Route]
    approval_required: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Candidate semantic decision, later checked by deterministic validation."""

    status: DecisionStatus
    route: Route | None
    task_type: str
    required_capabilities: frozenset[Capability]
    confidence: float | None
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_capabilities",
            normalize_capabilities(self.required_capabilities),
        )
        if self.status is DecisionStatus.ROUTED and self.route is None:
            raise ValueError("routed decisions require a route")
        if self.status is DecisionStatus.NO_SUITABLE_ROUTE and self.route is not None:
            raise ValueError("no_suitable_route decisions must not contain a route")
        if not self.required_capabilities:
            raise ValueError("semantic decisions require at least one capability")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.task_type.strip():
            raise ValueError("task_type cannot be empty")
        if not self.reason.strip():
            raise ValueError("reason cannot be empty")


@dataclass(frozen=True, slots=True)
class JevDecision:
    """Authoritative decision returned by Jev."""

    decision_type: DecisionType
    status: DecisionStatus
    route: Route | None
    task_type: str
    required_capabilities: tuple[Capability, ...]
    confidence: float | None
    reason: str
    approval_required: bool
    allowed_routes: tuple[Route, ...]
    policy_reasons: tuple[str, ...]
