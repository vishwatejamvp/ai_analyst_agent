# Eval Harness — How to Use & Extend

A minimal, layered eval system for the AI Analyst Agent. Three principles:

1. **Cases as data** — `golden.jsonl` is the dataset. Add cases without
   touching Python.
2. **Layered execution** — three levels (cheap → expensive). Run only
   what you have infra for.
3. **Pure scorers** — each scorer is a side-effect-free function in
   `scorers.py`. Easy to test in isolation.

## Quick start

```bash
# Cheapest layer only — no DB, no API key needed.
python -m tests.eval.runner --level intent

# Routing layer — needs Mongo to probe schema.
python -m tests.eval.runner --level routing

# Full pipeline — needs Mongo + ANTHROPIC_API_KEY.
python -m tests.eval.runner --level e2e

# Everything (default).
python -m tests.eval.runner

# Filter by tag(s).
python -m tests.eval.runner --tag follow-up
python -m tests.eval.runner --tag greeting,oos

# Run a single case by id (great for debugging one regression).
python -m tests.eval.runner --only intent-greet-001

# Save JSON for diffing across runs.
python -m tests.eval.runner --save tests/eval/results/$(date +%Y%m%d-%H%M).json

# Include the FULL trace dict per case (large; off by default).
python -m tests.eval.runner --traces --save tests/eval/results/full.json

# CI mode — exit non-zero on any non-skipped failure.
python -m tests.eval.runner --strict
```

## Observability output (build #2)

Every case runs inside its own `TraceContext`. The runner records:

* `latency_ms` — end-to-end wall-clock for the case
* `input_tokens` / `output_tokens` — sum across all Claude calls
* `cost_usd` — computed from `utils.observability.pricing.PRICING`
* `llm_calls` — number of Claude calls (helpful when you expect a single
  call but accidentally introduce a loop)

These appear in both the terminal report and the saved JSON. Pass
`--traces` to additionally save the per-span timing tree under
`Result.trace`.

## Levels

| Level     | Calls                              | Needs                  |
|-----------|------------------------------------|------------------------|
| `intent`  | `question_intent.classify`         | nothing                |
| `routing` | `routing_service.decide`           | Mongo (schema probe)   |
| `e2e`     | `analyst_orchestrator.answer`      | Mongo + Anthropic key  |

If a level's infra is missing, those cases are **skipped**, not failed —
the cheap layers above still produce a clean report.

## Adding a case

Append one line to `golden.jsonl`. Required: `id`, `level`, `question`.
Everything in `expect` is optional — assert only what matters for your case.

```json
{"id": "route-zakat-sum-001", "level": "routing",
 "question": "total zakat disbursed in 2024",
 "expect": {"route": "ANALYTICAL", "operation": "sum", "time_present": true},
 "tags": ["analytical", "zakat", "time"]}
```

### Available `expect` fields

* `intent` — `DISCOVERY|OUT_OF_SCOPE|COMPARISON|ANALYTICAL`
* `route` — `ANALYTICAL|SEMANTIC|HYBRID|DISCOVERY|OUT_OF_SCOPE`
* `target` — exact collection/table the router must pick
* `operation` — `sum|avg|count|min|max`
* `metric` *or* `metric_in: [list]`
* `group_by`
* `time_present: true|false`
* `text_contains: ["substring", ...]` — case-insensitive AND across the list
* `text_not_contains: [...]` — hallucination guards
* `warning_codes_include / warning_codes_exclude: ["stale_data", ...]`
* `data_nonempty: true|false`
* `chart_type: "BAR|LINE|PIE|KPI|SKIP"` — assert the rendered chart type
  (use `SKIP` when no chart should be produced)
* `llm_calls_min / llm_calls_max` — cost-budget regression guard
* `reranked: true|false` — whether the cross-encoder rerank stage ran
* `vector_hits_min` — minimum vector_context hits returned
* `span_names_include / span_names_exclude` — assert specific spans
  appeared (or didn't) in the case's trace; useful as positive /
  negative controls for opt-in subsystems like the summariser
* `routing_refined: true|false` — whether the LLM router fallback
  patched the rule decision (true = at least one `llm-refined:`
  marker present in `matched_keywords`)
* `refined_fields_include: ["target", "group_by", ...]` — which
  fields the refiner must have patched (matches the comma-list
  inside the `llm-refined:` marker)
* `critic_action: "approve|flag|fallback"` — verdict from the
  insight critic (read from `meta.critic.action`)
* `critic_max_blocking_issues: N` — cap on medium/high-severity
  critic findings; set `0` as a false-positive guard against an
  over-eager critic prompt
* `intent_source: "rule|distiller|distiller-agree"` — Build #8;
  pin which classifier (rule, distilled student override, or
  distilled student concurrence) decided the intent. Read from
  `IntentResult.source`. Use `intent_source: "distiller"` to assert
  the trained model overrode the rule on a known-gap question.
* `intent_source_in: [list]` — same field, set membership instead
  of exact match. Useful for "either rule or distiller-agree is
  fine, but never an override".

### Follow-up cases (e2e only)

Set both `previous_question` and `session_id`. The runner sends the
prior question first to seed session state, then sends `question` so
the session-patch path is exercised.

```json
{"id": "e2e-followup-001", "level": "e2e",
 "question": "by emirate", "previous_question": "average occupancy in 2024",
 "session_id": "eval-followup-1",
 "expect": {"group_by": "emirate", "data_nonempty": true},
 "tags": ["follow-up"]}
```

### Multi-turn cases (build #5 — conversation memory)

Use `previous_questions: [...]` with the same `session_id` to seed
several prior turns. The runner fires every seed turn under the
session before sending `question`; scoring + tracing apply ONLY to
the final turn so the case's reported tokens / latency reflect that
turn under test, not the warm-up.

Run with the summariser explicitly enabled and a low trigger so a
short case still trips compression:

```bash
python -m tests.eval.runner \
  --tag session-summary \
  --session-summary on \
  --summary-trigger 3 \
  --summary-keep-last 2 \
  --traces \
  --save tests/eval/results/run_summary.json
```

```json
{"id": "e2e-session-summary-001", "level": "e2e",
 "session_id": "eval-summary-1",
 "previous_questions": ["q1", "q2", "q3"],
 "question": "compare those two years",
 "expect": {"span_names_include": ["session.summarise"]},
 "tags": ["session-summary"]}
```

When the case passes, inspect the trace with
`python -m tests.eval._inspect_results` — you should see the
`session.summarise` span nested inside the final turn's
`request.answer`, and a Claude usage entry with low input/output
token counts (the summariser prompt is small).

### Catalog-first routing (build #9)

`RoutingService` scores the question against `awqaf_datasets_metadata`
(slug, display name, purpose, key metrics) before collection-name token
overlap. The chosen method is exposed on `routing.resolution` and in
`meta.routing_debug` when `include_details=true`.

Env: `CATALOG_ROUTING_ENABLED` (default true), `CATALOG_ROUTING_MIN_SCORE`
(default 3.0). Dataset choice is server-side only (catalog routing + session);
the chat UI sends `question` + `session_id` only. Integrators may use
`GET /api/v1/datasets` or `collection` on `POST /analyze` when needed.

### LLM router fallback (build #6 — uncertain rule cases)

Enabled by default (`ROUTER_LLM_FALLBACK_ENABLED=true`). `decide()` consults
an LLM second-opinion when uncertainty flags fire (no_target,
missing_metric_for_op, group_by_intent_unmet, op_via_default,
low_target_overlap, pure_semantic_with_year). Clean queries skip
the LLM entirely (~$0); only ambiguous ones pay the ~$0.007 / 3 s
extra round trip.

Run eval with `--llm-router off` to assert rule-only baselines.

```bash
python -m tests.eval.runner \
  --tag llm-router \
  --llm-router on \
  --traces \
  --save tests/eval/results/run_router.json
```

```json
{"id": "route-llmfallback-groupby-001", "level": "routing",
 "question": "top 5 emirates by total revenue from occupancy and revenues",
 "expect": {"route": "ANALYTICAL", "operation": "sum", "group_by": "emirate",
            "routing_refined": true, "refined_fields_include": ["group_by"],
            "span_names_include": ["router.llm_fallback"]},
 "tags": ["llm-router", "known-gap-fix"]}
```

Cases tagged `llm-router` are **feature-gated**: they assert the
refiner ran and will fail under default `--llm-router off`. This is
the same convention as the `chart-llm` and `reranker` cases — run
them with the matching flag.

### Insight critic (build #7 — verifier loop over the analyst draft)

Run with `--critic on` to verify each analyst draft against
STRUCTURED DATA / TRUST PANEL / QUALITY WARNINGS. The critic flags
fabricated numbers, wrong periods, wrong scope, contradictions of
the trust panel, and silently-dropped warnings. Medium/high-severity
findings trigger one bounded revise round (unless
`--critic-revise off` puts the critic in shadow / annotate-only
mode).

```bash
python -m tests.eval.runner --tag critic --critic on --traces
python -m tests.eval.runner --tag critic-off    # negative control
```

Critic verdicts are non-deterministic across runs (the analyst's
draft itself varies even at temperature=0). Eval cases therefore
assert mechanism — that the critic ran and `llm_calls` stays in
{2 (approve), 3 (revise)} — rather than pinning a specific verdict.

To watch the critic on a specific question and see its findings:

```bash
python -m tests.eval._inspect_critic "average occupancy rate in 2024"
```

The inspector prints the draft, the critic's verdict, each
flagged issue with severity / type / quote / evidence /
suggested_fix, and the per-call cost breakdown.

### Distilled intent classifier (build #8 — small student replaces big LLM)

Run with `--distill on` to let `question_intent.classify` consult
a TF-IDF + LogReg student model trained from the rule classifier.
The student NEVER overrides the rule unless three conditions hold
simultaneously:

1. the model artifact (`data/intent_distiller.joblib`) loaded
2. its confidence is ≥ `--distill-threshold`
3. it disagrees with the rule's verdict

Otherwise the rule wins. The student is therefore a strict upgrade
path: it can only fix rule misses, never make them worse silently.

```bash
# Train the model from data/intent_labels.jsonl.
python -m scripts.train_intent_distiller

# Run the distillation cases (override + agreement + gate test).
python -m tests.eval.runner \
  --tag distill \
  --distill on \
  --distill-threshold 0.4 \
  --traces

# Negative control: distillation off, rule must stand.
python -m tests.eval.runner --tag distill-off
```

The threshold default is `0.75` — appropriate when the model has
been trained on hundreds+ of labels. The bootstrap dataset shipped
with the repo (`data/intent_labels.jsonl`, ~60 rows) produces
confidences in the 0.4-0.55 band, which is why the override case
needs `--distill-threshold 0.4` to fire. As you collect more labels
via `services.intent_distiller.collect_label`, retrain and raise
the threshold back to the conservative default.

```json
{"id": "intent-distill-override-001", "level": "intent",
 "question": "okay so what is in here",
 "expect": {"intent": "DISCOVERY", "intent_source": "distiller"},
 "tags": ["distill", "override"]}
```

Cases tagged `distill` are **feature-gated**: they assert the
distilled student fired and will fail under default `--distill off`.
Same convention as `llm-router` and `critic` cases. The `distill-off`
tagged case is the inverse — asserts the rule path stands when
distillation is disabled, fails if you accidentally leave
distillation on while running the baseline.

## When to add a case

* **You fixed a bug** → add the failing input as a regression case.
* **You added a new feature / route** → add a case that triggers it.
* **A user complained** → freeze their question as a permanent eval case.

Goal: every reported issue lives forever in `golden.jsonl`. The dataset
grows with the system's history.

## CI integration

```yaml
# .github/workflows/eval.yml (sketch)
- run: python -m tests.eval.runner --level intent,routing --strict
```

The cheap levels run on every PR. The e2e level runs on a nightly job
where the full stack is available.
