from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from uuid import uuid4

from localia.execution.contracts import ExecutionResult, ExecutionStatus
from localia.execution.registry import ExecutorRegistry
from localia.execution.telemetry import JsonlTelemetry
from localia.jev import DecisionStatus, Jev, JevRequest


class ExecutionHarness:
    """Route with Jev, then execute through the selected concrete backend."""

    def __init__(
        self,
        *,
        jev: Jev,
        executors: ExecutorRegistry,
        telemetry: JsonlTelemetry | None = None,
    ) -> None:
        self.jev = jev
        self.executors = executors
        self.telemetry = telemetry

    def run(
        self,
        request: JevRequest,
        *,
        approved: bool = False,
        task_id: str | None = None,
    ) -> ExecutionResult:
        run_id = uuid4().hex
        task_id = task_id or run_id

        started = perf_counter()
        routing_started = perf_counter()
        decision = self.jev.decide(request)
        routing_latency = perf_counter() - routing_started

        if decision.status is DecisionStatus.NO_SUITABLE_ROUTE:
            result = ExecutionResult(
                run_id=run_id,
                task_id=task_id,
                decision=decision,
                status=ExecutionStatus.NO_SUITABLE_ROUTE,
                provider=None,
                output="",
                stderr="",
                exit_code=None,
                error=None,
                approved=approved,
                routing_latency_seconds=routing_latency,
                execution_latency_seconds=0.0,
                total_latency_seconds=perf_counter() - started,
            )
            self._record(request, result)
            return result

        if decision.approval_required and not approved:
            result = ExecutionResult(
                run_id=run_id,
                task_id=task_id,
                decision=decision,
                status=ExecutionStatus.BLOCKED_APPROVAL,
                provider=None,
                output="",
                stderr="",
                exit_code=None,
                error="human approval is required before execution",
                approved=False,
                routing_latency_seconds=routing_latency,
                execution_latency_seconds=0.0,
                total_latency_seconds=perf_counter() - started,
            )
            self._record(request, result)
            return result

        route = decision.route
        assert route is not None

        if not self.executors.has(route):
            result = ExecutionResult(
                run_id=run_id,
                task_id=task_id,
                decision=decision,
                status=ExecutionStatus.EXECUTOR_UNAVAILABLE,
                provider=None,
                output="",
                stderr="",
                exit_code=None,
                error=f"no executor registered for route: {route.value}",
                approved=approved,
                routing_latency_seconds=routing_latency,
                execution_latency_seconds=0.0,
                total_latency_seconds=perf_counter() - started,
            )
            self._record(request, result)
            return result

        executor = self.executors.get(route)
        execution_started = perf_counter()
        provider_output = executor.execute(request, decision)
        execution_latency = perf_counter() - execution_started

        result = ExecutionResult(
            run_id=run_id,
            task_id=task_id,
            decision=decision,
            status=(
                ExecutionStatus.SUCCEEDED
                if provider_output.success
                else ExecutionStatus.FAILED
            ),
            provider=executor.provider,
            output=provider_output.output,
            stderr=provider_output.stderr,
            exit_code=provider_output.exit_code,
            error=provider_output.error,
            approved=approved,
            routing_latency_seconds=routing_latency,
            execution_latency_seconds=execution_latency,
            total_latency_seconds=perf_counter() - started,
            metadata=provider_output.metadata,
        )
        self._record(request, result)
        return result

    def _record(self, request: JevRequest, result: ExecutionResult) -> None:
        if self.telemetry is not None:
            self.telemetry.record(request, result)
