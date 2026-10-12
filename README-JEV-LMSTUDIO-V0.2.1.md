# LocalIA / Jev — LM Studio identity fix v0.2.1

LM Studio can expose a loaded quantized GGUF under the generic loaded-model
identifier `qwen/qwen3.5-9b`, even when the concrete file is
`Qwen3.5-9B-Q4_K_M.gguf`.

Version 0.2.0 incorrectly required the loaded `modelKey` itself to equal
`qwen/qwen3.5-9b@q4_k_m`.

Version 0.2.1 verifies the concrete Q4 using the strongest model metadata
available from the LM Studio SDK:

1. exact quantized model key, when exposed;
2. concrete GGUF path containing `Q4_K_M`;
3. the exact Q4 file size established during the benchmark phase, used as a
   fallback when LM Studio exposes only a generic path.

A generic identifier alone is NOT accepted.

After extracting this overlay into the repository:

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
localia --lmstudio-audit
```

Optional raw LM Studio diagnostic:

```powershell
lms ps --json
```
