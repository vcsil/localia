# LocalIA / Jev v0.3.1 — schema-constrained semantic routing

## Why v0.3.0 failed

The native LM Studio endpoint `/api/v1/chat` successfully disabled reasoning,
but it does not provide the JSON-schema structured-output contract used by
LM Studio's OpenAI-compatible `/v1/chat/completions` endpoint.

So the v0.3.0 model was only *asked* to return JSON. On a real routing prompt
Qwen instead returned YAML-like text:

```text
task_type: codex
reason: ...
```

That output was correctly rejected by LocalIA.

## v0.3.1 design

- SDK: audit the already-loaded calibrated Q4.
- `/v1/chat/completions`: semantic routing.
- `response_format.type = json_schema`: grammar-constrained JSON.
- model API identifier: `qwen/qwen3.5-9b@q4_k_m`.
- temperature: 0.
- max_tokens: 128.
- exact three-field decision contract.
- arbitrary prose/YAML/partial JSON is rejected.

Qwen/LM Studio versions have sometimes put schema-constrained JSON in
`reasoning_content` with `content` empty. LocalIA accepts that workaround only
when the entire channel is exactly one valid JSON object. It never extracts
arbitrary embedded JSON from prose.

The single-route short-circuit from v0.3.0 remains unchanged.

## Apply

```powershell
Expand-Archive .\localia_jev_structured_v0.3.1.zip -DestinationPath . -Force
python -m pip install -e .
python -m unittest discover -s tests -v
```

Keep the loaded model's LM Studio API Identifier set to:

```text
qwen/qwen3.5-9b@q4_k_m
```

and keep the server running:

```powershell
lms server status
```

## Tests

Single route — no semantic LLM call:

```powershell
localia "Explique cache L3"
```

Real semantic choice:

```powershell
localia "Corrija os testes deste repositório Python" --enable codex
```

Counter-test:

```powershell
localia "Explique de forma simples o que é fotossíntese" --enable codex
```

Policy dominance:

```powershell
localia "Corrija este código Python" --enable codex --local-only
```

## Commit message

```text
fix(jev): enforce schema-constrained LM Studio routing
```
