from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Protocol

from localia.execution.contracts import ExecutorOutput
from localia.jev import Capability, JevDecision, JevRequest, Route
from localia.providers.lmstudio import (
    LMStudioRestBackend,
)


class ExecutorUnavailableError(RuntimeError):
    """Raised when a configured execution backend is not available."""


class Executor(Protocol):
    route: Route
    provider: str

    def execute(
        self,
        request: JevRequest,
        decision: JevDecision,
    ) -> ExecutorOutput:
        ...


class LocalFastExecutor:
    """Execute ordinary local tasks through LM Studio chat completions."""

    route = Route.LOCAL_FAST
    provider = "lmstudio:qwen3.5-9b-q4_k_m"

    def __init__(
        self,
        *,
        backend: LMStudioRestBackend | None = None,
    ) -> None:
        self.backend = backend or LMStudioRestBackend()

    def execute(
        self,
        request: JevRequest,
        decision: JevDecision,
    ) -> ExecutorOutput:
        if decision.route is not self.route:
            raise ValueError("LocalFastExecutor received a non-local_fast decision")

        try:
            content = self.backend.respond_text(
                system_prompt=(
                    "You are LocalIA's local execution model. "
                    "Perform the user's task directly. "
                    "Do not discuss routing or internal capabilities."
                ),
                user_prompt=request.task,
                config={"temperature": 0.2, "maxTokens": 1024},
            )
        except Exception as exc:
            return ExecutorOutput(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        return ExecutorOutput(
            success=True,
            output=content,
            metadata=dict(self.backend.last_metadata or {}),
        )


class CodexExecutor:
    """Execute repository/tool tasks through the Codex CLI."""

    route = Route.CODEX
    provider = "codex_cli"

    def __init__(
        self,
        *,
        executable: str = "codex",
        cwd: str | Path | None = None,
        timeout_seconds: float = 900.0,
    ) -> None:
        self.executable = executable
        self.cwd = Path(cwd).resolve() if cwd is not None else Path.cwd()
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _sandbox(decision: JevDecision) -> str:
        capabilities = set(decision.required_capabilities)
        if (
            Capability.REPOSITORY_EDIT in capabilities
            or Capability.CODE_EXECUTION in capabilities
        ):
            return "workspace-write"
        return "read-only"

    def _command(
        self,
        request: JevRequest,
        decision: JevDecision,
    ) -> list[str]:
        return [
            self.executable,
            "exec",
            "--sandbox",
            self._sandbox(decision),
            "--ask-for-approval",
            "never",
            "--skip-git-repo-check",
            request.task,
        ]

    def execute(
        self,
        request: JevRequest,
        decision: JevDecision,
    ) -> ExecutorOutput:
        if decision.route is not self.route:
            raise ValueError("CodexExecutor received a non-codex decision")

        command = self._command(request, decision)
        try:
            completed = subprocess.run(
                command,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            return ExecutorOutput(
                success=False,
                error=(
                    f"Codex CLI executable {self.executable!r} was not found. "
                    "Install/configure Codex CLI before enabling execution."
                ),
                metadata={"executable": self.executable},
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutorOutput(
                success=False,
                output="" if exc.stdout is None else str(exc.stdout),
                stderr="" if exc.stderr is None else str(exc.stderr),
                error=f"Codex CLI timed out after {self.timeout_seconds:.0f}s",
                metadata={"executable": self.executable},
            )
        except OSError as exc:
            return ExecutorOutput(
                success=False,
                error=f"Could not start Codex CLI: {exc}",
                metadata={"executable": self.executable},
            )

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if completed.returncode != 0:
            return ExecutorOutput(
                success=False,
                output=stdout.strip(),
                stderr=stderr.strip(),
                exit_code=completed.returncode,
                error=f"Codex CLI exited with code {completed.returncode}",
                metadata={
                    "executable": self.executable,
                    "sandbox": self._sandbox(decision),
                    "cwd": str(self.cwd),
                },
            )

        return ExecutorOutput(
            success=True,
            output=stdout.strip(),
            stderr=stderr.strip(),
            exit_code=completed.returncode,
            metadata={
                "executable": self.executable,
                "sandbox": self._sandbox(decision),
                "cwd": str(self.cwd),
            },
        )
