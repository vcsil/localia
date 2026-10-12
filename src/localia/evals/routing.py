from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable, Mapping

from localia.jev import (
    ActionRisk,
    Jev,
    JevDecision,
    JevRequest,
    PrivacyLevel,
    Route,
)
from localia.jev.registry import RouteRegistry
from localia.jev.semantic import LMStudioSemanticRouter, StubSemanticRouter
from localia.providers import LMStudioRestBackend


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES = PROJECT_ROOT / "evals" / "routing_cases_v1.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"


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


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("routing eval corpus must be a JSON array")

    ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, Mapping):
            raise ValueError(f"case #{index} is not an object")
        case = dict(item)
        case_id = str(case.get("id", "")).strip()
        task = str(case.get("task", "")).strip()
        category = str(case.get("category", "")).strip()
        expected = case.get("expected")
        if not case_id or case_id in ids:
            raise ValueError(f"invalid/duplicate case id: {case_id!r}")
        if not task:
            raise ValueError(f"case {case_id} has empty task")
        if not category:
            raise ValueError(f"case {case_id} has empty category")
        if not isinstance(expected, Mapping):
            raise ValueError(f"case {case_id} has invalid expected object")
        ids.add(case_id)
        validated.append(case)
    return validated


def build_request(case: Mapping[str, Any]) -> JevRequest:
    local_only = bool(case.get("local_only", False))
    return JevRequest(
        task=str(case["task"]),
        privacy=(PrivacyLevel.LOCAL_ONLY if local_only else PrivacyLevel.NORMAL),
        cloud_allowed=not local_only,
        context_tokens_estimate=int(case.get("context_tokens", 0)),
        action_risk=(
            ActionRisk.DESTRUCTIVE
            if bool(case.get("destructive", False))
            else ActionRisk.NONE
        ),
        metadata={"eval_case_id": str(case["id"])},
    )


def build_registry(case: Mapping[str, Any]) -> RouteRegistry:
    registry = RouteRegistry()
    for route_name in case.get("enable_routes", []):
        registry.set_enabled(Route(str(route_name)), True)
    for route_name in case.get("disable_routes", []):
        registry.set_enabled(Route(str(route_name)), False)
    return registry


def decision_to_dict(decision: JevDecision) -> dict[str, Any]:
    return _jsonable(asdict(decision))


def _expected_checks(
    case: Mapping[str, Any],
    decision: JevDecision,
) -> dict[str, bool]:
    expected = dict(case["expected"])
    actual_caps = {item.value for item in decision.required_capabilities}
    actual_allowed = {item.value for item in decision.allowed_routes}

    checks: dict[str, bool] = {}

    if "status" in expected:
        checks["status"] = decision.status.value == expected["status"]

    if "route" in expected:
        actual_route = decision.route.value if decision.route else None
        checks["route"] = actual_route == expected["route"]

    must_include = set(expected.get("capabilities_must_include", []))
    must_not_include = set(expected.get("capabilities_must_not_include", []))
    if must_include or must_not_include:
        checks["capabilities"] = (
            must_include <= actual_caps
            and not (must_not_include & actual_caps)
        )

    if "approval_required" in expected:
        checks["approval"] = (
            decision.approval_required
            is bool(expected["approval_required"])
        )

    if "allowed_routes_exact" in expected:
        checks["policy_allowed_routes"] = (
            actual_allowed == set(expected["allowed_routes_exact"])
        )

    if "policy_reasons_must_include" in expected:
        reasons = "\n".join(decision.policy_reasons).casefold()
        checks["policy_reasons"] = all(
            str(fragment).casefold() in reasons
            for fragment in expected["policy_reasons_must_include"]
        )

    # A routed result outside the authoritative allowed set is always a safety
    # failure, even if a malformed corpus forgot to assert it explicitly.
    checks["no_policy_violation"] = (
        decision.route is None or decision.route in decision.allowed_routes
    )

    return checks


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * percentile
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    fraction = index - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    successful_calls = [r for r in records if r.get("error") is None]
    latencies = [float(r["latency_seconds"]) for r in successful_calls]

    check_names = sorted(
        {
            name
            for record in successful_calls
            for name in record.get("checks", {})
        }
    )
    check_accuracy: dict[str, float | None] = {}
    for name in check_names:
        applicable = [
            bool(record["checks"][name])
            for record in successful_calls
            if name in record.get("checks", {})
        ]
        check_accuracy[name] = (
            sum(applicable) / len(applicable) if applicable else None
        )

    passed = sum(bool(record.get("passed")) for record in records)
    policy_violations = sum(
        1
        for record in successful_calls
        if record.get("checks", {}).get("no_policy_violation") is False
    )

    case_signatures: dict[str, set[str]] = {}
    for record in successful_calls:
        decision = record["decision"]
        signature = json.dumps(
            {
                "status": decision.get("status"),
                "route": decision.get("route"),
                "required_capabilities": decision.get("required_capabilities"),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        case_signatures.setdefault(record["case_id"], set()).add(signature)

    consistency_rate = None
    if case_signatures:
        consistency_rate = (
            sum(len(signatures) == 1 for signatures in case_signatures.values())
            / len(case_signatures)
        )

    return {
        "runs": total,
        "passed": passed,
        "failed": total - passed,
        "overall_accuracy": passed / total if total else None,
        "errors": total - len(successful_calls),
        "policy_violations": policy_violations,
        "check_accuracy": check_accuracy,
        "consistency_rate": consistency_rate,
        "latency_seconds": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "min": min(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
    }


def _make_semantic_router(name: str):
    if name == "stub":
        return StubSemanticRouter()
    if name == "lmstudio":
        return LMStudioSemanticRouter(LMStudioRestBackend())
    raise ValueError(f"unsupported router: {name}")


def run_suite(
    cases: Iterable[Mapping[str, Any]],
    *,
    router_name: str,
    repeat: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    semantic_router = _make_semantic_router(router_name)
    records: list[dict[str, Any]] = []

    for case in cases:
        for iteration in range(1, repeat + 1):
            started = time.perf_counter()
            try:
                registry = build_registry(case)
                jev = Jev(
                    registry=registry,
                    semantic_router=semantic_router,
                )
                decision = jev.decide(build_request(case))
                elapsed = time.perf_counter() - started
                checks = _expected_checks(case, decision)
                passed = bool(checks) and all(checks.values())
                records.append(
                    {
                        "case_id": case["id"],
                        "category": case["category"],
                        "iteration": iteration,
                        "task": case["task"],
                        "latency_seconds": elapsed,
                        "decision": decision_to_dict(decision),
                        "checks": checks,
                        "passed": passed,
                        "error": None,
                    }
                )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                records.append(
                    {
                        "case_id": case["id"],
                        "category": case["category"],
                        "iteration": iteration,
                        "task": case["task"],
                        "latency_seconds": elapsed,
                        "decision": None,
                        "checks": {},
                        "passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    return records, summarize_records(records)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the LocalIA/Jev routing evaluation suite"
    )
    parser.add_argument(
        "--router",
        choices=("lmstudio", "stub"),
        default="lmstudio",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="run only this category; repeatable",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.repeat < 1:
        print("error: --repeat must be >= 1", file=sys.stderr)
        return 2
    if args.limit < 0:
        print("error: --limit must be >= 0", file=sys.stderr)
        return 2

    try:
        cases = load_cases(args.cases)
    except Exception as exc:
        print(f"error loading cases: {exc}", file=sys.stderr)
        return 2

    if args.category:
        selected = set(args.category)
        cases = [case for case in cases if case["category"] in selected]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("error: no evaluation cases selected", file=sys.stderr)
        return 2

    print(
        f"Running {len(cases)} routing cases x {args.repeat} "
        f"with router={args.router}..."
    )

    records, summary = run_suite(
        cases,
        router_name=args.router,
        repeat=args.repeat,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output
    if output is None:
        DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output = DEFAULT_RESULTS_DIR / f"routing_eval_{args.router}_{timestamp}.json"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "router": args.router,
        "cases_file": str(args.cases),
        "selected_cases": len(cases),
        "repeat": args.repeat,
        "summary": summary,
        "records": records,
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        "Result: "
        f"{summary['passed']}/{summary['runs']} passed "
        f"({summary['overall_accuracy']:.1%})"
    )
    checks = summary["check_accuracy"]
    for name in ("status", "route", "capabilities", "approval", "policy_allowed_routes", "policy_reasons", "no_policy_violation"):
        value = checks.get(name)
        if value is not None:
            print(f"  {name}: {value:.1%}")
    print(f"  policy_violations: {summary['policy_violations']}")
    print(f"  errors: {summary['errors']}")
    print(f"  consistency: {summary['consistency_rate']:.1%}")
    latency = summary["latency_seconds"]
    if latency["mean"] is not None:
        print(
            "  latency: "
            f"mean={latency['mean']:.3f}s "
            f"p50={latency['p50']:.3f}s "
            f"p95={latency['p95']:.3f}s"
        )
    print(f"Report: {output}")

    failed_records = [record for record in records if not record["passed"]]
    if failed_records:
        print("Failures:")
        for record in failed_records[:20]:
            detail = record["error"] or ", ".join(
                name
                for name, passed in record.get("checks", {}).items()
                if not passed
            )
            print(
                f"  - {record['case_id']}#{record['iteration']}: {detail}"
            )
        if len(failed_records) > 20:
            print(f"  ... and {len(failed_records) - 20} more")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
