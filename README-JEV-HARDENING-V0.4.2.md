# LocalIA / Jev v0.4.2 — capability/reason hardening

This release hardens two findings from the first live capability-aware runs.

## 1. Capability implications are deterministic

`repository_edit` now implies `repository_read` through a transitive
normalization step owned by deterministic code. A model no longer has to
remember to emit both capabilities.

Normalization is applied to:

- semantic decisions;
- route specifications;
- external calls to `RouteRegistry.suitable_routes()`.

## 2. The model no longer writes Jev's authoritative reason

The live Qwen correctly classified `general`, but its free-form reason then
incorrectly claimed that `local_fast` lacked the capability. That prose was
never authoritative, but it was misleading.

The structured model contract now contains only:

```text
route
task_type
required_capabilities
```

Jev itself generates the final `reason` from deterministic policy/registry
facts. Model prose is neither requested nor surfaced.

This also reduces the semantic-router output budget from 160 to 96 tokens.

## Commit

```text
fix(jev): normalize capabilities and derive authoritative reasons
```
