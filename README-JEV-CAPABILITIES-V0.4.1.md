# LocalIA / Jev v0.4.1 — deterministic capability resolution

## Bug fixed

v0.4.0 asked Qwen to generate both:

- `status`
- `route`

The JSON schema constrained each field independently but did not encode the
cross-field rule:

```text
status=no_suitable_route  <=>  route=none
```

A real Qwen call therefore produced a structurally valid but internally
inconsistent pair, causing:

```text
SemanticRouterError: no_suitable_route must use route='none'
```

## Architecture correction

v0.4.1 removes `status` from the model-generated schema entirely.

Qwen now returns only:

```text
route preference
task_type
required_capabilities
reason
```

Jev derives suitability deterministically:

```text
semantic required_capabilities
        |
        v
allowed routes + RouteRegistry
        |
        +-- 0 suitable routes -> no_suitable_route
        |
        +-- 1 suitable route  -> choose it deterministically
        |
        +-- 2+ suitable routes -> use Qwen route preference
```

This is stronger than merely adding a JSON-schema conditional because
`no_suitable_route` is not fundamentally a probabilistic judgment once the
required capabilities have been classified. It is a set-membership result.

The CLI also catches semantic/validator errors and prints a concise failure
instead of exposing a Python traceback.

## Apply

```powershell
Expand-Archive .\localia_jev_capabilities_v0.4.1_fix.zip -DestinationPath . -Force
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Live tests

```powershell
localia "Explique cache L3"
```

Expected: `routed -> local_fast` with `general`.

```powershell
localia "Edite os arquivos deste repositório Python e rode os testes" --enable codex
```

Expected: `routed -> codex`.

```powershell
localia "Edite os arquivos deste repositório Python e rode os testes" --enable codex --local-only
```

Expected: `no_suitable_route -> route: null`.

## Commit message

```text
fix(jev): derive route status from capability suitability
```
