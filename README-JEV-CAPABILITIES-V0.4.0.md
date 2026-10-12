# LocalIA / Jev v0.4.0 — capabilities + no_suitable_route

This milestone fixes a structural bug in routing:

> A route being permitted does not mean it is capable of doing the task.

## New flow

```text
JevRequest
   |
PolicyEngine                 deterministic: what is allowed?
   |
policy-allowed routes
   |
SemanticRouter / Qwen        semantic: what does the task require?
   |                         + which allowed route can do all of it?
   v
RouteDecision
   |-- routed(route, required_capabilities)
   `-- no_suitable_route(required_capabilities)
   |
DecisionValidator            deterministic consistency check
   |
JevDecision
```

## Typed capabilities

Current operational capabilities are:

- general
- classification
- extraction
- summarization
- structured_output
- code_reasoning
- repository_read
- repository_edit
- code_execution
- long_context
- reasoning

The distinction between `code_reasoning` and repository/execution capability is
intentional. Qwen can explain code locally without having filesystem or command
execution access.

## Important behavioral change

v0.3.x short-circuited when policy left one route. v0.4.0 removes that unsafe
optimization. A single allowed route must still be checked semantically for
capability suitability.

Example:

```text
cloud policy removes Codex
+ task requires repository_edit + code_execution
+ local_fast lacks those capabilities
= no_suitable_route
```

This is a normal decision result, not an exception.

## Apply

```powershell
Expand-Archive .\localia_jev_capabilities_v0.4.0.zip -DestinationPath . -Force
python -m pip install -e .
python -m unittest discover -s tests -v
```

Keep LM Studio Q4 loaded with API identifier:

```text
qwen/qwen3.5-9b@q4_k_m
```

and the server running.

## Live tests

General task:

```powershell
localia "Explique cache L3"
```

Expected: `status=routed`, `route=local_fast`, capability `general`.

Repository task with Codex enabled:

```powershell
localia "Edite os arquivos deste repositório Python e rode os testes" --enable codex
```

Expected: `status=routed`, `route=codex`, with repository/execution capability.

Same task with cloud forbidden:

```powershell
localia "Edite os arquivos deste repositório Python e rode os testes" --enable codex --local-only
```

Expected: `status=no_suitable_route`, `route=null`.

## Git commit

```text
feat(jev): add capability-aware no-suitable-route decisions
```
