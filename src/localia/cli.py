from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from enum import Enum
from typing import Any

from .execution import (
    CodexExecutor,
    ExecutionHarness,
    ExecutionStatus,
    ExecutorRegistry,
    JsonlTelemetry,
    LocalFastExecutor,
)
from .jev import ActionRisk, Jev, JevRequest, PrivacyLevel, Route
from .jev.semantic import (
    LMStudioSemanticRouter,
    SemanticRouterError,
    StubSemanticRouter,
)
from .jev.validator import DecisionValidationError
from .providers import (
    LMStudioBackend,
    LMStudioConfigurationError,
    LMStudioRestBackend,
    LMStudioUnavailableError,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LocalIA / Jev decision harness"
    )
    parser.add_argument("task", nargs="?", help="task to route")
    parser.add_argument(
        "--router",
        choices=("lmstudio", "stub"),
        default="lmstudio",
        help="semantic router backend (default: lmstudio REST)",
    )
    parser.add_argument(
        "--lmstudio-audit",
        action="store_true",
        help="inspect the exact loaded Qwen router model and exit",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="forbid all cloud routes",
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=0,
        help="estimated context size",
    )
    parser.add_argument(
        "--destructive",
        action="store_true",
        help="mark the requested action as destructive",
    )
    parser.add_argument(
        "--enable",
        action="append",
        choices=[route.value for route in Route],
        default=[],
        help="enable an additional route for this run; repeatable",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute the routed task instead of returning only the decision",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="approve a destructive execution for this invocation",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=Path.cwd(),
        help="working directory for tool/repository executors",
    )
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=Path("data/runs.jsonl"),
        help="JSONL telemetry path used with --execute",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.lmstudio_audit:
        try:
            print(
                json.dumps(
                    _jsonable(LMStudioBackend().audit()),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        except (LMStudioUnavailableError, LMStudioConfigurationError) as exc:
            print(f"LM Studio audit failed: {exc}", file=sys.stderr)
            return 2

    if not args.task:
        print(
            "error: task is required unless --lmstudio-audit is used",
            file=sys.stderr,
        )
        return 2

    lmstudio_backend = None
    if args.router == "lmstudio":
        lmstudio_backend = LMStudioRestBackend()
        semantic_router = LMStudioSemanticRouter(lmstudio_backend)
    else:
        semantic_router = StubSemanticRouter()

    jev = Jev(semantic_router=semantic_router)

    for route_name in args.enable:
        jev.registry.set_enabled(Route(route_name), True)

    local_only = bool(args.local_only)

    request = JevRequest(
        task=args.task,
        privacy=(
            PrivacyLevel.LOCAL_ONLY
            if local_only
            else PrivacyLevel.NORMAL
        ),
        cloud_allowed=not local_only,
        context_tokens_estimate=args.context_tokens,
        action_risk=(
            ActionRisk.DESTRUCTIVE
            if args.destructive
            else ActionRisk.NONE
        ),
    )

    if args.approve and not args.execute:
        print("error: --approve requires --execute", file=sys.stderr)
        return 2

    if args.execute:
        execution_backend = lmstudio_backend or LMStudioRestBackend()
        executors = ExecutorRegistry(
            (
                LocalFastExecutor(backend=execution_backend),
                CodexExecutor(cwd=args.cwd),
            )
        )
        harness = ExecutionHarness(
            jev=jev,
            executors=executors,
            telemetry=JsonlTelemetry(args.telemetry),
        )
        try:
            result = harness.run(request, approved=bool(args.approve))
        except (LMStudioUnavailableError, LMStudioConfigurationError) as exc:
            print(f"LM Studio router failed: {exc}", file=sys.stderr)
            return 2
        except (SemanticRouterError, DecisionValidationError) as exc:
            print(f"Jev routing failed: {exc}", file=sys.stderr)
            return 3

        print(
            json.dumps(
                _jsonable(asdict(result)),
                ensure_ascii=False,
                indent=2,
            )
        )
        if result.status is ExecutionStatus.SUCCEEDED:
            return 0
        if result.status is ExecutionStatus.BLOCKED_APPROVAL:
            return 4
        if result.status is ExecutionStatus.NO_SUITABLE_ROUTE:
            return 5
        return 6

    try:
        decision = jev.decide(request)
    except (LMStudioUnavailableError, LMStudioConfigurationError) as exc:
        print(f"LM Studio router failed: {exc}", file=sys.stderr)
        return 2
    except (SemanticRouterError, DecisionValidationError) as exc:
        print(f"Jev routing failed: {exc}", file=sys.stderr)
        return 3

    payload = _jsonable(asdict(decision))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
