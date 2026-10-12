# LocalIA / Jev v0.5.1 — capability hardening

v0.5.1 hardens the capability classifier after the first full 55-case routing
benchmark of v0.5.0.

## Baseline

The v0.5.0 LM Studio evaluation produced:

- 55 runs;
- 43 passed;
- 78.18% strict overall accuracy;
- 98.18% route accuracy;
- 79.63% capability accuracy;
- 100% status, approval and policy checks;
- 0 policy violations.

The failures were concentrated in capability inference rather than policy or
route safety.

## What changed

### Deterministic capability hardening

The semantic model remains the primary classifier, but Jev now adds
high-confidence operational facts when they are explicitly stated by the task:

- explicit JSON/YAML/XML/CSV output adds `structured_output`;
- explicit repository edits add `repository_edit`;
- explicit test/git execution adds `code_execution`;
- clear chat-only code markers add `code_reasoning`;
- `long_context` and `structured_output` cannot replace the underlying
  functional capability.

Negated execution requests such as "não rode nada" are explicitly preserved.

### Stable task types

Operational capabilities canonicalize obvious task types such as
`repository_edit`, `code_execution`, `repository_read`, and
`code_reasoning`.

### Chat-only code stays local

When the required capability is ordinary `code_reasoning` and a capable local
route exists, Jev deterministically keeps the task local instead of allowing a
semantic preference to escalate it to Codex.

This does not apply to repository access, repository edits, or code execution.

### Benchmark corpus unchanged

`evals/routing_cases_v1.json` is intentionally unchanged. v0.5.1 should be
measured against exactly the same 55 expectations.

## Validation

Run unit tests:

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
```

Run the same live LM Studio benchmark:

```powershell
localia-routing-eval --router lmstudio --repeat 1
```

Then measure repeatability:

```powershell
localia-routing-eval --router lmstudio --repeat 3
```

Target for v0.5.1:

- >= 95% strict overall accuracy;
- 100% policy / no-policy-violation checks;
- no regression in route accuracy.
