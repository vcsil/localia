# LocalIA / Jev v0.5.2 — semantic edge-case hardening

v0.5.2 is a focused follow-up to v0.5.1.

## Baseline from v0.5.1

The unchanged 55-case LM Studio benchmark produced:

- 52/55 strict passes (94.55%);
- 52/54 valid decisions (96.30%) when excluding one transient LM Studio transport failure;
- 100% route accuracy;
- 100% status, approval and policy checks;
- 0 policy violations.

Two semantic capability failures remained:

1. read-only repository analysis was incorrectly marked as `repository_edit`;
2. project implementation work failed to recover `repository_edit`.

A third failed record came from a transient LM Studio engine fetch failure.

## Changes

### Explicit negative constraints can remove capabilities

Phrases such as:

- `sem modificar nada`
- `não altere`
- `read-only`
- `without modifying`

can now remove a spurious `repository_edit` capability inferred by the model.

This is intentionally narrow: only explicit negative constraints can remove an
operational capability.

### Project context is repository context when operational

`projeto` / `project` now count as repository context for explicit
operational requests such as implementation, editing, refactoring or adding
code/tests.

This lets tasks such as:

`Implemente uma nova função neste projeto...`

recover `repository_edit` even if the model omitted it.

### Narrow LM Studio transient retry

Jev retries exactly once when LM Studio reports:

`Engine protocol predict request failed: fetch failed`

Other HTTP/configuration failures are not retried.

## Benchmark discipline

`evals/routing_cases_v1.json` remains unchanged.

The acceptance test for v0.5.2 is the same 55-case benchmark:

```powershell
localia-routing-eval --router lmstudio --repeat 1
```

No repeat=3 run is required for this milestone.

Target:

- >= 54/55 strict passes;
- ideally 55/55 if the LM Studio request succeeds;
- 100% route accuracy;
- 0 policy violations.
