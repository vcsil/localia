# LocalIA / Jev v0.6.0 — Execution Layer

v0.6 turns Jev from a routing-only harness into a routing + execution harness.

The v0.5.2 routing baseline remains unchanged:

- 55/55 routing cases passed;
- 100% route/capability/status/approval/policy checks;
- 0 policy violations.

## Architecture

```text
JevRequest
    |
    v
Deterministic policy
    |
    v
Semantic routing + capability hardening
    |
    v
JevDecision
    |
    v
ExecutionHarness
    |
    +--> approval gate
    |
    v
ExecutorRegistry
    |
    +--> local_fast -> LM Studio
    |
    +--> codex      -> Codex CLI
    |
    v
ExecutionResult
    |
    v
JSONL telemetry
```

Routing and execution are deliberately separate. `Jev.decide()` was not
changed by this milestone.

## Execution contracts

The execution layer introduces:

- `ExecutorOutput`: provider-level outcome;
- `ExecutionResult`: authoritative harness outcome;
- `ExecutionStatus`:
  - `succeeded`
  - `failed`
  - `blocked_approval`
  - `no_suitable_route`
  - `executor_unavailable`
- `ExecutorRegistry`;
- `ExecutionHarness`.

## LocalFastExecutor

`local_fast` executes through LM Studio using a plain text chat completion.

Routing still uses schema-constrained structured output. Execution does not
reuse the routing schema.

This keeps two responsibilities separate:

```text
Qwen as router     -> structured classification
Qwen as executor   -> normal task response
```

## CodexExecutor

Codex tasks are run non-interactively through:

```text
codex exec
```

The sandbox is selected from the authoritative capabilities:

```text
repository_read only
    -> read-only

repository_edit or code_execution
    -> workspace-write
```

The harness owns user approval. Codex therefore runs with:

```text
--ask-for-approval never
```

after the LocalIA approval gate has already allowed the task.

No `danger-full-access` mode is used.

Network access is not enabled by v0.6. Remote/network actions may therefore
fail at execution time until network access becomes an explicit capability and
policy decision.

## Human approval

A destructive decision is never executed merely because an executor exists.

Without explicit approval:

```text
decision.approval_required = true
approved = false

=> blocked_approval
=> executor is never called
```

CLI approval is per invocation:

```powershell
localia "Exclua..." --execute --destructive --approve
```

## Telemetry

Execution telemetry defaults to:

```text
data/runs.jsonl
```

Raw prompts and raw model responses are NOT stored in telemetry by default.

Instead the record contains:

- run/task IDs;
- SHA-256-derived task fingerprint;
- route/provider;
- task type;
- required capabilities;
- approval state;
- routing/execution/total latency;
- success/failure;
- exit code;
- sanitized provider metadata.

`data/runs.jsonl` is git-ignored.

## CLI

Routing-only behavior remains backward compatible:

```powershell
localia "Explique DNS"
```

Execute locally:

```powershell
localia "Explique DNS" --execute
```

Allow Codex as a route and execute in the current repository:

```powershell
localia "Analise este repositório" --enable codex --execute --cwd .
```

Repository edit:

```powershell
localia "Corrija os testes deste repositório" --enable codex --execute --cwd .
```

Destructive task, first without approval:

```powershell
localia "Apague a branch..." --enable codex --execute --destructive --cwd .
```

This returns `blocked_approval`.

Run only after explicit approval:

```powershell
localia "Apague a branch..." --enable codex --execute --destructive --approve --cwd .
```

## Exit codes

```text
0  success
2  CLI / LM Studio availability/configuration error
3  routing / validation error
4  execution blocked awaiting approval
5  no suitable route
6  executor unavailable or execution failed
```

## Tests

Run:

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
```

The suite includes:

- successful execution;
- approval blocking before executor invocation;
- approved destructive execution;
- no-suitable-route short circuit;
- unavailable executors;
- provider failures;
- telemetry privacy;
- local LM Studio text execution;
- Codex read-only/workspace-write sandbox selection;
- Codex subprocess success/failure;
- plain-text LM Studio REST completions.

## Deliberately deferred

v0.6 does not add:

- Claude executor;
- OpenRouter executor;
- local_heavy executor;
- dynamic context;
- automatic route escalation after execution failure;
- network capability/policy;
- end-to-end quality grading.

Those belong in subsequent milestones after real local_fast/Codex execution is
validated.
