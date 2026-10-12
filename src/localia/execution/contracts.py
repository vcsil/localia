from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from localia.jev import JevDecision, Route


class ExecutionStatus(str, Enum):
    """Terminal state of an execution attempt."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED_APPROVAL = "blocked_approval"
    NO_SUITABLE_ROUTE = "no_suitable_route"
    EXECUTOR_UNAVAILABLE = "executor_unavailable"


@dataclass(frozen=True, slots=True)
class ExecutorOutput:
    """Provider-level output before the harness adds routing metadata."""

    success: bool
    output: str = ""
    stderr: str = ""
    exit_code: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if self.success and self.error:
            raise ValueError("successful executor output cannot contain error")
        if not self.success and not self.error:
            raise ValueError("failed executor output requires error")


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Authoritative result returned by the LocalIA execution harness."""

    run_id: str
    task_id: str
    decision: JevDecision
    status: ExecutionStatus
    provider: str | None
    output: str
    stderr: str
    exit_code: int | None
    error: str | None
    approved: bool
    routing_latency_seconds: float
    execution_latency_seconds: float
    total_latency_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status is ExecutionStatus.SUCCEEDED

    @property
    def route(self) -> Route | None:
        return self.decision.route

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id cannot be empty")
        if not self.task_id.strip():
            raise ValueError("task_id cannot be empty")
        if self.routing_latency_seconds < 0:
            raise ValueError("routing latency cannot be negative")
        if self.execution_latency_seconds < 0:
            raise ValueError("execution latency cannot be negative")
        if self.total_latency_seconds < 0:
            raise ValueError("total latency cannot be negative")
        if self.success and self.decision.route is None:
            raise ValueError("successful execution requires a routed decision")
        if self.success and self.error:
            raise ValueError("successful execution cannot contain error")
        if self.status is ExecutionStatus.FAILED and not self.error:
            raise ValueError("failed execution requires error")
