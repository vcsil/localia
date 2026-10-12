# LocalIA / Jev — structured-output compatibility fix v0.2.2

This release hardens the LM Studio provider against a known class of
structured-output failures seen with reasoning-capable Qwen models.

Normal path:

1. Prefer `result.parsed` when the SDK returns a mapping.
2. Otherwise accept content only when it is itself a complete JSON object.
3. Recover a JSON suffix after LM Studio's explicit synthetic-reasoning
   separator.
4. Recover a JSON suffix after one closed Qwen `<think>...</think>` prefix.
5. Reject arbitrary prose containing embedded JSON.

The provider still returns only a candidate semantic decision. Policies and
DecisionValidator remain authoritative.

If recovery still fails, the error now includes diagnostics such as:

- SDK `structured` flag;
- parsed/content types;
- stop reason;
- predicted token count;
- a bounded content preview.

This makes it possible to distinguish reasoning leakage from a token-budget
cutoff without accepting loose model output.

Apply:

```powershell
Expand-Archive .\localia_jev_lmstudio_v0.2.2_structured_fix.zip -DestinationPath . -Force
python -m pip install -e .
python -m unittest discover -s tests -v
localia --lmstudio-audit
localia "Explique cache L3"
```
