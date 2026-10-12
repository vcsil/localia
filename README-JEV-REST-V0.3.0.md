# LocalIA / Jev v0.3.0 — REST semantic router

The live PowerShell test established that the LM Studio native REST endpoint
can run Qwen3.5-9B Q4_K_M with:

- `reasoning: "off"`
- `reasoning_output_tokens: 0`
- a single `message` containing only the requested JSON

So v0.3.0 separates responsibilities:

```text
LM Studio Python SDK
  -> audit the loaded Q4 instance

LM Studio native REST /api/v1/chat
  -> semantic routing with reasoning explicitly OFF
```

Jev also now short-circuits when policy leaves only one route:

```text
1 allowed route  -> deterministic selection, no LLM call
2+ allowed routes -> Qwen semantic decision
```

## Apply

```powershell
Expand-Archive .\localia_jev_rest_v0.3.0.zip -DestinationPath . -Force
python -m pip install -e .
python -m unittest discover -s tests -v
```

## Keep LM Studio ready

```powershell
lms server start --port 1234
localia --lmstudio-audit
```

## Single-route path

This should no longer invoke Qwen:

```powershell
localia "Explique cache L3"
```

Expected:

```text
route = local_fast
task_type = deterministic_single_route
confidence = null
```

## First real semantic choice

```powershell
localia "Corrija os testes deste repositório Python" --enable codex
```

Now `local_fast` and `codex` are both allowed, so Qwen must decide between
them via REST with reasoning disabled.

Codex is still only a route candidate at this milestone; its executor is not
connected yet.
