from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from enum import Enum
from typing import Any

from .jev import ActionRisk, Jev, JevRequest, PrivacyLevel, Route


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
    parser = argparse.ArgumentParser(description="LocalIA / Jev Core V0 decision harness")
    parser.add_argument("task", help="task to route")
    parser.add_argument("--local-only", action="store_true", help="forbid all cloud routes")
    parser.add_argument("--context-tokens", type=int, default=0, help="estimated context size")
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

    jev = Jev()
    for route_name in args.enable:
        jev.registry.set_enabled(Route(route_name), True)

    local_only = bool(args.local_only)
    request = JevRequest(
        task=args.task,
        privacy=PrivacyLevel.LOCAL_ONLY if local_only else PrivacyLevel.NORMAL,
        cloud_allowed=not local_only,
        context_tokens_estimate=args.context_tokens,
        action_risk=ActionRisk.DESTRUCTIVE if args.destructive else ActionRisk.NONE,
    )

    decision = jev.decide(request)
    print(json.dumps(_jsonable(asdict(decision)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
