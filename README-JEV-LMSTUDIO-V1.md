# Jev + LM Studio Semantic Router — milestone 0.2

This update replaces the user-facing CLI's semantic stub with the real
LM Studio/Qwen router while keeping deterministic policy authoritative.

## Architecture

```text
JevRequest
   |
PolicyEngine          <- deterministic
   |
allowed routes only
   |
LMStudioSemanticRouter
   |
Qwen3.5-9B Q4_K_M     <- semantic judgment only
   |
RouteDecision
   |
DecisionValidator     <- deterministic
   |
JevDecision
```

## Important design decisions

### Qwen is not Jev

Qwen is only Jev's semantic routing engine. It cannot override privacy,
availability, approval, context, or other deterministic constraints.

### Structured output limits the model

The JSON schema sent to LM Studio dynamically enumerates only the routes
that policy already allowed. The validator checks the result again.

### No fake confidence

The LM Studio semantic router returns `confidence: null` for now. The old
stub values such as 0.70 and 0.85 were placeholders. A probability-like
confidence value will only be reintroduced after empirical calibration.

### Exact model, reuse-only

LocalIA intentionally does not silently JIT-load the router model. The exact
Q4 instance must already be loaded in LM Studio. This prevents an accidental
load with different GPU/context/thread settings.

Expected profile:

- model key: `qwen/qwen3.5-9b@q4_k_m`
- Q4_K_M
- CPU AVX2 runtime
- CPU thread pool: 8
- prediction CPU threads: 8
- GPU offload: 0
- context: 4096
- temperature: 0
- Thinking: OFF

Some LM Studio SDK versions do not expose every llama.cpp load parameter.
`--lmstudio-audit` validates what can be observed through the SDK and states
what remains to confirm in LM Studio/logs.

## Install / refresh editable package

Inside the activated `.venv`:

```powershell
python -m pip install -e .
```

This installs the pinned `lmstudio==1.5.0` SDK as a project dependency.

## Run unit tests

```powershell
python -m unittest discover -s tests -v
```

These tests do not require LM Studio. They use fake backends to verify the
trust boundaries and provider behavior.

## Live preflight

Keep LM Studio open and load the exact Q4 model with the validated profile.

Then:

```powershell
localia --lmstudio-audit
```

## First real Jev routing call

To make route selection meaningful, enable another candidate route:

```powershell
localia "Corrija os testes deste repositório Python" --enable codex
```

The command uses the real local Qwen semantic router. `codex` is only a
candidate route at this stage; its executor is not connected yet.

For an offline comparison with the old heuristic:

```powershell
localia "Corrija os testes deste repositório Python" --enable codex --router stub
```

## Next milestone

After the live semantic-routing smoke test succeeds, build a small routing
evaluation set before connecting the real Codex/Claude executors.
