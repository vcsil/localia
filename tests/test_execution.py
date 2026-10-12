import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from localia.execution import (
    CodexExecutor,
    ExecutionHarness,
    ExecutionStatus,
    ExecutorOutput,
    ExecutorRegistry,
    JsonlTelemetry,
    LocalFastExecutor,
)
from localia.jev import (
    Capability,
    DecisionStatus,
    DecisionType,
    JevDecision,
    JevRequest,
    Route,
)


def decision(
    *,
    route=Route.LOCAL_FAST,
    status=DecisionStatus.ROUTED,
    capabilities=(Capability.GENERAL,),
    approval_required=False,
):
    return JevDecision(
        decision_type=DecisionType.ROUTE_TASK,
        status=status,
        route=route,
        task_type="general",
        required_capabilities=tuple(capabilities),
        confidence=None,
        reason="test decision",
        approval_required=approval_required,
        allowed_routes=(() if route is None else (route,)),
        policy_reasons=(),
    )


class FakeJev:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def decide(self, request):
        self.calls += 1
        return self.value


class FakeExecutor:
    def __init__(self, route=Route.LOCAL_FAST, output=None):
        self.route = route
        self.provider = "fake"
        self.output = output or ExecutorOutput(success=True, output="done")
        self.calls = 0

    def execute(self, request, routed_decision):
        self.calls += 1
        return self.output


class FakeTextBackend:
    def __init__(self):
        self.calls = []
        self.last_metadata = {"model": "test-model", "usage": {"total_tokens": 3}}

    def respond_text(self, *, system_prompt, user_prompt, config):
        self.calls.append((system_prompt, user_prompt, config))
        return "local answer"


class ExecutionHarnessTests(unittest.TestCase):
    def test_successful_execution_uses_selected_executor(self):
        routed = decision()
        executor = FakeExecutor()
        harness = ExecutionHarness(
            jev=FakeJev(routed),
            executors=ExecutorRegistry((executor,)),
        )

        result = harness.run(JevRequest(task="Explain DNS"))

        self.assertEqual(result.status, ExecutionStatus.SUCCEEDED)
        self.assertTrue(result.success)
        self.assertEqual(result.output, "done")
        self.assertEqual(result.provider, "fake")
        self.assertEqual(executor.calls, 1)

    def test_destructive_task_is_blocked_before_executor_without_approval(self):
        routed = decision(approval_required=True)
        executor = FakeExecutor()
        harness = ExecutionHarness(
            jev=FakeJev(routed),
            executors=ExecutorRegistry((executor,)),
        )

        result = harness.run(JevRequest(task="delete branch"))

        self.assertEqual(result.status, ExecutionStatus.BLOCKED_APPROVAL)
        self.assertEqual(executor.calls, 0)

    def test_approved_destructive_task_can_execute(self):
        routed = decision(approval_required=True)
        executor = FakeExecutor()
        harness = ExecutionHarness(
            jev=FakeJev(routed),
            executors=ExecutorRegistry((executor,)),
        )

        result = harness.run(
            JevRequest(task="delete branch"),
            approved=True,
        )

        self.assertEqual(result.status, ExecutionStatus.SUCCEEDED)
        self.assertEqual(executor.calls, 1)

    def test_no_suitable_route_never_calls_executor(self):
        routed = decision(
            route=None,
            status=DecisionStatus.NO_SUITABLE_ROUTE,
            capabilities=(),
        )
        executor = FakeExecutor()
        harness = ExecutionHarness(
            jev=FakeJev(routed),
            executors=ExecutorRegistry((executor,)),
        )

        result = harness.run(JevRequest(task="unsupported"))

        self.assertEqual(result.status, ExecutionStatus.NO_SUITABLE_ROUTE)
        self.assertEqual(executor.calls, 0)

    def test_missing_executor_is_reported_without_execution(self):
        routed = decision(route=Route.CODEX)
        harness = ExecutionHarness(
            jev=FakeJev(routed),
            executors=ExecutorRegistry(),
        )

        result = harness.run(JevRequest(task="repo task"))

        self.assertEqual(result.status, ExecutionStatus.EXECUTOR_UNAVAILABLE)
        self.assertIn("no executor registered", result.error)

    def test_failed_provider_becomes_failed_execution_result(self):
        routed = decision()
        executor = FakeExecutor(
            output=ExecutorOutput(success=False, error="provider failed")
        )
        harness = ExecutionHarness(
            jev=FakeJev(routed),
            executors=ExecutorRegistry((executor,)),
        )

        result = harness.run(JevRequest(task="task"))

        self.assertEqual(result.status, ExecutionStatus.FAILED)
        self.assertEqual(result.error, "provider failed")


class TelemetryTests(unittest.TestCase):
    def test_jsonl_omits_raw_task_and_output(self):
        raw_task = "private prompt that must not be logged"
        raw_output = "private model answer"
        routed = decision()
        executor = FakeExecutor(
            output=ExecutorOutput(success=True, output=raw_output)
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.jsonl"
            harness = ExecutionHarness(
                jev=FakeJev(routed),
                executors=ExecutorRegistry((executor,)),
                telemetry=JsonlTelemetry(path),
            )
            result = harness.run(JevRequest(task=raw_task))
            text = path.read_text(encoding="utf-8")
            record = json.loads(text)

        self.assertEqual(record["run_id"], result.run_id)
        self.assertNotIn(raw_task, text)
        self.assertNotIn(raw_output, text)
        self.assertEqual(record["route"], "local_fast")
        self.assertTrue(record["success"])


class LocalFastExecutorTests(unittest.TestCase):
    def test_local_executor_uses_public_text_backend(self):
        backend = FakeTextBackend()
        executor = LocalFastExecutor(backend=backend)

        result = executor.execute(
            JevRequest(task="Explain DNS"),
            decision(),
        )

        self.assertTrue(result.success)
        self.assertEqual(result.output, "local answer")
        self.assertEqual(result.metadata["model"], "test-model")
        self.assertEqual(len(backend.calls), 1)


class CodexExecutorTests(unittest.TestCase):
    def test_read_only_capabilities_use_read_only_sandbox(self):
        executor = CodexExecutor(cwd=".")
        routed = decision(
            route=Route.CODEX,
            capabilities=(Capability.REPOSITORY_READ,),
        )

        command = executor._command(JevRequest(task="inspect repo"), routed)

        self.assertIn("read-only", command)
        self.assertIn("--ask-for-approval", command)
        self.assertIn("never", command)

    def test_edit_capability_uses_workspace_write_sandbox(self):
        executor = CodexExecutor(cwd=".")
        routed = decision(
            route=Route.CODEX,
            capabilities=(
                Capability.REPOSITORY_READ,
                Capability.REPOSITORY_EDIT,
            ),
        )

        command = executor._command(JevRequest(task="edit repo"), routed)

        self.assertIn("workspace-write", command)

    @patch("localia.execution.executors.subprocess.run")
    def test_codex_success_returns_stdout(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=["codex"],
            returncode=0,
            stdout="completed",
            stderr="",
        )
        executor = CodexExecutor(cwd=".")
        routed = decision(
            route=Route.CODEX,
            capabilities=(Capability.REPOSITORY_READ,),
        )

        result = executor.execute(JevRequest(task="inspect repo"), routed)

        self.assertTrue(result.success)
        self.assertEqual(result.output, "completed")
        self.assertNotIn("command", result.metadata)

    @patch("localia.execution.executors.subprocess.run")
    def test_codex_nonzero_exit_is_failure(self, run):
        run.return_value = subprocess.CompletedProcess(
            args=["codex"],
            returncode=2,
            stdout="",
            stderr="bad",
        )
        executor = CodexExecutor(cwd=".")
        routed = decision(
            route=Route.CODEX,
            capabilities=(Capability.REPOSITORY_READ,),
        )

        result = executor.execute(JevRequest(task="inspect repo"), routed)

        self.assertFalse(result.success)
        self.assertEqual(result.exit_code, 2)


if __name__ == "__main__":
    unittest.main()
