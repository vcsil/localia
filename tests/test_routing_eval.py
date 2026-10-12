import json
from pathlib import Path
import tempfile
import unittest

from localia.evals.routing import (
    PROJECT_ROOT,
    _expected_checks,
    load_cases,
    summarize_records,
)
from localia.jev.contracts import (
    Capability,
    DecisionStatus,
    JevDecision,
    DecisionType,
    Route,
)


class RoutingEvalTests(unittest.TestCase):
    def test_corpus_is_large_unique_and_category_diverse(self):
        cases = load_cases(PROJECT_ROOT / "evals" / "routing_cases_v1.json")

        self.assertGreaterEqual(len(cases), 50)
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        self.assertGreaterEqual(len({case["category"] for case in cases}), 7)

    def test_expected_checks_score_route_capability_policy_and_approval(self):
        case = {
            "id": "x",
            "category": "test",
            "task": "x",
            "expected": {
                "status": "routed",
                "route": "codex",
                "capabilities_must_include": ["repository_edit"],
                "capabilities_must_not_include": ["general"],
                "approval_required": True,
                "allowed_routes_exact": ["codex"],
                "policy_reasons_must_include": ["approval"],
            },
        }
        decision = JevDecision(
            decision_type=DecisionType.ROUTE_TASK,
            status=DecisionStatus.ROUTED,
            route=Route.CODEX,
            task_type="repository_edit",
            required_capabilities=(
                Capability.REPOSITORY_EDIT,
                Capability.REPOSITORY_READ,
            ),
            confidence=None,
            reason="deterministic",
            approval_required=True,
            allowed_routes=(Route.CODEX,),
            policy_reasons=("destructive action requires human approval",),
        )

        checks = _expected_checks(case, decision)

        self.assertTrue(all(checks.values()))

    def test_summary_reports_accuracy_policy_violations_and_consistency(self):
        records = [
            {
                "case_id": "a",
                "latency_seconds": 1.0,
                "decision": {
                    "status": "routed",
                    "route": "local_fast",
                    "required_capabilities": ["general"],
                },
                "checks": {"route": True, "no_policy_violation": True},
                "passed": True,
                "error": None,
            },
            {
                "case_id": "a",
                "latency_seconds": 2.0,
                "decision": {
                    "status": "routed",
                    "route": "local_fast",
                    "required_capabilities": ["general"],
                },
                "checks": {"route": True, "no_policy_violation": True},
                "passed": True,
                "error": None,
            },
        ]

        summary = summarize_records(records)

        self.assertEqual(summary["overall_accuracy"], 1.0)
        self.assertEqual(summary["policy_violations"], 0)
        self.assertEqual(summary["consistency_rate"], 1.0)
        self.assertEqual(summary["latency_seconds"]["mean"], 1.5)

    def test_loader_rejects_duplicate_ids(self):
        content = [
            {"id": "dup", "category": "x", "task": "one", "expected": {}},
            {"id": "dup", "category": "x", "task": "two", "expected": {}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text(json.dumps(content), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(path)


if __name__ == "__main__":
    unittest.main()
