from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DecisionType(str, Enum):
    """Decision capabilities offered by Jev.

    V0 implements only ROUTE_TASK. Keeping the capability explicit prevents
    Jev from being architecturally reduced to a model router forever.
    """

    ROUTE_TASK = "route_task"


class Route(str, Enum):
    LOCAL_FAST = "local_fast"
    LOCAL_HEAVY = "local_heavy"
    CODEX = "codex"
    CLAUDE = "claude"
    API_FALLBACK = "api_fallback"


class PrivacyLevel(str, Enum):
    NORMAL = "normal"
    LOCAL_ONLY = "local_only"


class ActionRisk(str, Enum):
    NONE = "none"
    REVERSIBLE = "reversible"
    DESTRUCTIVE = "destructive"


@dataclass(frozen=True, slots=True)
class JevRequest:
    """Normalized request sent to the decision harness."""

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
    capabilities: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class PolicyResult:
    allowed_routes: frozenset[Route]
    approval_required: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Candidate semantic decision; policy/validator remain authoritative."""

    route: Route
    task_type: str
    confidence: float
    reason: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.task_type.strip():
            raise ValueError("task_type cannot be empty")
        if not self.reason.strip():
            raise ValueError("reason cannot be empty")


@dataclass(frozen=True, slots=True)
class JevDecision:
    """Authoritative decision returned by Jev."""

    decision_type: DecisionType
    route: Route
    task_type: str
    confidence: float
    reason: str
    approval_required: bool
    allowed_routes: tuple[Route, ...]
    policy_reasons: tuple[str, ...]
