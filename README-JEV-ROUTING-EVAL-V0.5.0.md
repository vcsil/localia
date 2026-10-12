# LocalIA / Jev v0.5.0 — Routing Evaluation Suite

This milestone adds the first repeatable routing benchmark for the real Jev
pipeline.

## Corpus

`evals/routing_cases_v1.json` contains 55 cases across:

- general knowledge;
- summarization/classification/extraction/structured output;
- code reasoning in conversation;
- repository read/edit/execution;
- local-only policy behavior;
- context-limit behavior;
- destructive-action approval;
- prompt-injection/adversarial routing attempts;
- ambiguous boundary cases.

Each case can assert:

- expected status;
- expected route;
- capabilities that must be present/absent;
- approval requirement;
- exact allowed-route set after policy;
- required policy-reason fragments.

## Metrics

The runner reports:

- overall accuracy;
- status accuracy;
- route accuracy;
- capability-constraint accuracy;
- approval accuracy;
- policy allowed-route/reason accuracy;
- policy-violation count;
- errors;
- repeat consistency;
- mean/P50/P95 routing latency.

Every run is also written to JSON under `benchmarks/results/` by default.

## First real baseline

Keep Qwen Q4 loaded and LM Studio server running, then run:

```powershell
localia-routing-eval --router lmstudio
```

For a smaller first pass:

```powershell
localia-routing-eval --router lmstudio --limit 10
```

For a category:

```powershell
localia-routing-eval --router lmstudio --category repository
```

For repeatability testing:

```powershell
localia-routing-eval --router lmstudio --repeat 3
```

The command exits with code `1` when any evaluated run misses its expected
contract, while still writing the complete report.

## Interpretation target

The first run is a baseline, not a pass/fail judgment of the project. A useful
initial target before connecting real executors is:

```text
policy violations: 0
invalid/unavailable routes: 0
status/route accuracy: >= 95%
```

Capability accuracy is intentionally measured separately because it is the
semantic boundary we expect to tune most during this phase.

## Commit

```text
feat(jev): add routing evaluation suite
```
