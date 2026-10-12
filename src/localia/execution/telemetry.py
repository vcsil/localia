from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

from localia.execution.contracts import ExecutionResult
from localia.jev import JevRequest


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def task_fingerprint(task: str) -> str:
    return hashlib.sha256(task.encode("utf-8")).hexdigest()[:16]


class JsonlTelemetry:
    """Append-only telemetry sink without raw prompt/response content by default."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def record(self, request: JevRequest, result: ExecutionResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        decision = result.decision
        record = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": result.run_id,
            "task_id": result.task_id,
            "task_fingerprint": task_fingerprint(request.task),
            "status": result.status.value,
            "success": result.success,
            "route": decision.route.value if decision.route else None,
            "provider": result.provider,
            "task_type": decision.task_type,
            "required_capabilities": [
                item.value for item in decision.required_capabilities
            ],
            "approval_required": decision.approval_required,
            "approved": result.approved,
            "routing_latency_seconds": result.routing_latency_seconds,
            "execution_latency_seconds": result.execution_latency_seconds,
            "total_latency_seconds": result.total_latency_seconds,
            "exit_code": result.exit_code,
            "error": result.error,
            "metadata": _jsonable(result.metadata),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
