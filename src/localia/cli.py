from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from enum import Enum
from typing import Any

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

    if args.router == "lmstudio":
        semantic_router = LMStudioSemanticRouter(LMStudioRestBackend())
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
