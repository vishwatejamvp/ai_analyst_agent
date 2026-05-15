"""Pure scoring functions.

Each scorer:

* takes the *expected* value (from :class:`Expect`) and the *actual*
  value (extracted from the agent output);
* returns a :class:`Score` capturing the verdict + a short human-readable
  message.

Why pure functions?
-------------------
Scorers are the unit of testability for the harness itself. A pure
``(expected, actual) -> Score`` is trivial to import in a Python REPL
and reason about — no fixtures, no I/O, no mocks. When the harness
disagrees with your intuition on a case, you can replay one scorer in
isolation.

All scorers are *None-safe* on the expected side: passing ``None``
means "the case did not assert on this dimension" and the scorer
returns ``None`` instead of a Score, so the runner can skip it.
"""

from __future__ import annotations

from typing import Any

from .cases import Score


def _ok(name: str, expected: Any, actual: Any, msg: str = "") -> Score:
    return Score(name=name, passed=True, expected=expected, actual=actual, message=msg)


def _fail(name: str, expected: Any, actual: Any, msg: str) -> Score:
    return Score(name=name, passed=False, expected=expected, actual=actual, message=msg)


# ---------------------------------------------------------------------------
# Equality / membership scorers
# ---------------------------------------------------------------------------
def score_intent(expected: str | None, actual: str) -> Score | None:
    if expected is None:
        return None
    if actual.upper() == expected.upper():
        return _ok("intent", expected, actual)
    return _fail("intent", expected, actual, f"intent mismatch: {actual} != {expected}")


def score_route(expected: str | None, actual: str) -> Score | None:
    if expected is None:
        return None
    if actual.upper() == expected.upper():
        return _ok("route", expected, actual)
    return _fail("route", expected, actual, f"route mismatch: {actual} != {expected}")


def score_target(expected: str | None, actual: str | None) -> Score | None:
    if expected is None:
        return None
    if actual == expected:
        return _ok("target", expected, actual)
    return _fail("target", expected, actual, f"target mismatch: {actual} != {expected}")


def score_operation(expected: str | None, actual: str | None) -> Score | None:
    if expected is None:
        return None
    if actual == expected:
        return _ok("operation", expected, actual)
    return _fail("operation", expected, actual, f"op mismatch: {actual} != {expected}")


def score_metric(
    expected: str | None,
    expected_in: list[str] | None,
    actual: str | None,
) -> Score | None:
    """Either an exact metric, or membership in an acceptable set.

    Use ``metric_in`` when several columns would be a reasonable answer
    (e.g. when the dataset has both ``revenue`` and ``total_revenue``).
    """
    if expected is None and expected_in is None:
        return None
    if expected_in is not None:
        if actual in expected_in:
            return _ok("metric", expected_in, actual)
        return _fail(
            "metric", expected_in, actual,
            f"metric `{actual}` not in expected set {expected_in}",
        )
    if actual == expected:
        return _ok("metric", expected, actual)
    return _fail("metric", expected, actual, f"metric mismatch: {actual} != {expected}")


def score_group_by(expected: str | None, actual: str | None) -> Score | None:
    if expected is None:
        return None
    if actual == expected:
        return _ok("group_by", expected, actual)
    return _fail("group_by", expected, actual, f"group_by mismatch: {actual} != {expected}")


def score_time_present(expected: bool | None, actual: bool) -> Score | None:
    if expected is None:
        return None
    if actual == expected:
        return _ok("time_present", expected, actual)
    return _fail(
        "time_present", expected, actual,
        "expected time-bucket present" if expected else "expected NO time bucket",
    )


# ---------------------------------------------------------------------------
# Text scorers (case-insensitive substring; AND across the list)
# ---------------------------------------------------------------------------
def score_text_contains(
    expected: list[str] | None, actual: str
) -> Score | None:
    if not expected:
        return None
    text = (actual or "").lower()
    missing = [s for s in expected if s.lower() not in text]
    if not missing:
        return _ok("text_contains", expected, "<insight>")
    return _fail(
        "text_contains", expected, "<insight>",
        f"missing substrings: {missing}",
    )


def score_text_not_contains(
    expected: list[str] | None, actual: str
) -> Score | None:
    if not expected:
        return None
    text = (actual or "").lower()
    found = [s for s in expected if s.lower() in text]
    if not found:
        return _ok("text_not_contains", expected, "<insight>")
    return _fail(
        "text_not_contains", expected, "<insight>",
        f"forbidden substrings present: {found}",
    )


# ---------------------------------------------------------------------------
# Warning + data scorers
# ---------------------------------------------------------------------------
def score_warnings_include(
    expected: list[str] | None, actual: list[str]
) -> Score | None:
    if not expected:
        return None
    actual_set = set(actual)
    missing = [w for w in expected if w not in actual_set]
    if not missing:
        return _ok("warnings_include", expected, sorted(actual_set))
    return _fail(
        "warnings_include", expected, sorted(actual_set),
        f"missing required warnings: {missing}",
    )


def score_warnings_exclude(
    expected: list[str] | None, actual: list[str]
) -> Score | None:
    if not expected:
        return None
    actual_set = set(actual)
    found = [w for w in expected if w in actual_set]
    if not found:
        return _ok("warnings_exclude", expected, sorted(actual_set))
    return _fail(
        "warnings_exclude", expected, sorted(actual_set),
        f"forbidden warnings present: {found}",
    )


def score_data_nonempty(
    expected: bool | None, actual_rows: int
) -> Score | None:
    if expected is None:
        return None
    is_nonempty = actual_rows > 0
    if is_nonempty == expected:
        return _ok("data_nonempty", expected, actual_rows)
    return _fail(
        "data_nonempty", expected, actual_rows,
        f"expected nonempty={expected} but got rows={actual_rows}",
    )


def score_chart_type(expected: str | None, actual: str | None) -> Score | None:
    """Compare the rendered chart type — accepts ``'SKIP'`` for "no chart".

    Use this with ``chart_decider=llm`` to verify the LLM tool's choices.
    """
    if expected is None:
        return None
    norm = (actual or "SKIP").upper()
    if norm == expected.upper():
        return _ok("chart_type", expected, norm)
    return _fail(
        "chart_type", expected, norm,
        f"chart_type mismatch: {norm} != {expected}",
    )


def score_reranked(
    expected: bool | None, hits: list
) -> Score | None:
    """Verify whether the cross-encoder rerank stage actually ran.

    A rerank-enabled run sets ``rerank_score`` on every hit it returned;
    a rerank-disabled run leaves the field as ``None``. So the test is
    "did at least one hit come back with rerank_score?".
    """
    if expected is None:
        return None
    has_reranked = any(getattr(h, "rerank_score", None) is not None for h in hits)
    if has_reranked == expected:
        return _ok("reranked", expected, has_reranked)
    return _fail(
        "reranked", expected, has_reranked,
        "expected reranking to have run" if expected
        else "expected reranking to NOT have run",
    )


def score_vector_hits_min(
    expected: int | None, actual: int
) -> Score | None:
    if expected is None:
        return None
    if actual >= expected:
        return _ok("vector_hits_min", expected, actual)
    return _fail(
        "vector_hits_min", expected, actual,
        f"expected at least {expected} vector hits, got {actual}",
    )


def score_critic_action(
    expected: str | None, critic_meta: dict | None
) -> Score | None:
    """Verify the critic returned the expected verdict.

    Reads ``response.meta['critic']['action']``. ``None`` for the
    actual value (e.g. critic disabled, no LLM ran) is reported as
    ``'<not-run>'`` so the failure message is unambiguous.
    """
    if expected is None:
        return None
    actual = (critic_meta or {}).get("action", "<not-run>")
    if actual == expected:
        return _ok("critic_action", expected, actual)
    return _fail(
        "critic_action", expected, actual,
        f"critic action mismatch: {actual} != {expected}",
    )


def score_critic_blocking_issues(
    expected_max: int | None, critic_meta: dict | None
) -> Score | None:
    """Cap on medium/high-severity critic findings.

    The most useful form is ``critic_max_blocking_issues=0`` on a
    clean analytical case — that asserts the critic did not invent
    issues against a faithful draft (a false-positive control).
    """
    if expected_max is None:
        return None
    issues = (critic_meta or {}).get("issues", []) or []
    blocking = [
        i for i in issues
        if i.get("severity") in ("high", "medium")
    ]
    if len(blocking) <= expected_max:
        return _ok("critic_blocking_issues", f"<= {expected_max}", len(blocking))
    return _fail(
        "critic_blocking_issues", f"<= {expected_max}", len(blocking),
        f"expected at most {expected_max} blocking findings, got {len(blocking)}",
    )


def score_routing_refined(
    expected: bool | None, matched_keywords: list[str]
) -> Score | None:
    """Verify whether the LLM router fallback actually patched the decision.

    The refiner appends a marker like ``llm-refined:target,group_by`` to
    ``RoutingDecision.matched_keywords`` whenever it applies one or more
    patches. Checking for that prefix is a tiny, exact-string contract
    that the eval can assert without re-implementing the patching logic.
    """
    if expected is None:
        return None
    has_marker = any(
        isinstance(kw, str) and kw.startswith("llm-refined:")
        for kw in (matched_keywords or [])
    )
    if has_marker == expected:
        return _ok("routing_refined", expected, has_marker)
    return _fail(
        "routing_refined", expected, has_marker,
        "expected the LLM refiner to have patched the decision"
        if expected
        else "expected the rule decision to stand without LLM refinement",
    )


def score_refined_fields_include(
    expected: list[str] | None, matched_keywords: list[str]
) -> Score | None:
    """Verify each named field appears in the refiner's audit marker.

    The marker is shaped like ``llm-refined:target,group_by`` — we
    parse the suffix and check membership. Useful to assert *what*
    the refiner fixed, not just *that* it ran.
    """
    if not expected:
        return None
    actual: set[str] = set()
    for kw in matched_keywords or []:
        if isinstance(kw, str) and kw.startswith("llm-refined:"):
            actual.update(
                f.strip() for f in kw[len("llm-refined:"):].split(",") if f.strip()
            )
    missing = [f for f in expected if f not in actual]
    if not missing:
        return _ok("refined_fields_include", expected, sorted(actual))
    return _fail(
        "refined_fields_include", expected, sorted(actual),
        f"refiner did not patch required fields: {missing}",
    )


def score_span_names_include(
    expected: list[str] | None, span_names: list[str]
) -> Score | None:
    """Verify each expected span name appears at least once in the trace.

    The runner extracts ``span_names`` from the case's ``TraceContext``
    after the run. Useful as a positive control: e.g. asserting the
    summariser actually ran (``session.summarise``) when conversation
    memory was enabled and the buffer crossed the trigger.
    """
    if not expected:
        return None
    actual_set = set(span_names)
    missing = [name for name in expected if name not in actual_set]
    if not missing:
        return _ok("span_names_include", expected, sorted(actual_set))
    return _fail(
        "span_names_include", expected, sorted(actual_set),
        f"missing required spans: {missing}",
    )


def score_span_names_exclude(
    expected: list[str] | None, span_names: list[str]
) -> Score | None:
    """Verify none of the listed span names appear in the trace.

    Negative control mirror of :func:`score_span_names_include` — useful
    to assert an opt-in subsystem stayed off in a baseline case.
    """
    if not expected:
        return None
    actual_set = set(span_names)
    found = [name for name in expected if name in actual_set]
    if not found:
        return _ok("span_names_exclude", expected, sorted(actual_set))
    return _fail(
        "span_names_exclude", expected, sorted(actual_set),
        f"forbidden spans present: {found}",
    )


def score_llm_calls(
    expected_min: int | None,
    expected_max: int | None,
    actual: int,
) -> Score | None:
    """Verify the case made the expected number of LLM calls.

    Useful as a *cost-budget regression test* — if a refactor accidentally
    introduces an extra LLM call, this scorer trips.
    """
    if expected_min is None and expected_max is None:
        return None
    if expected_min is not None and actual < expected_min:
        return _fail(
            "llm_calls", f">= {expected_min}", actual,
            f"expected at least {expected_min} LLM calls, got {actual}",
        )
    if expected_max is not None and actual > expected_max:
        return _fail(
            "llm_calls", f"<= {expected_max}", actual,
            f"expected at most {expected_max} LLM calls, got {actual}",
        )
    return _ok("llm_calls", (expected_min, expected_max), actual)


def score_intent_source(
    expected: str | None, actual: str | None
) -> Score | None:
    """Build #8: verify which classifier produced the intent.

    ``actual`` is read from ``IntentResult.source``. Cases that pin
    ``intent_source: "distiller"`` assert the trained student
    OVERRODE the rule (the high-leverage case). Cases that pin
    ``intent_source: "rule"`` assert the rule path stood, which is
    how baseline regression cases lock in "the model did not fire
    on routine inputs".
    """
    if expected is None:
        return None
    actual_norm = actual or "<missing>"
    if actual_norm == expected:
        return _ok("intent_source", expected, actual_norm)
    return _fail(
        "intent_source", expected, actual_norm,
        f"intent source mismatch: {actual_norm} != {expected}",
    )


def score_intent_source_in(
    expected: list[str] | None, actual: str | None
) -> Score | None:
    """Build #8: verify intent source falls in an allowed set.

    Useful when you want "either rule or distiller-agree is fine,
    but distiller (override) is not" — tighter than ``intent_source``
    alone but more permissive than pinning a single value.
    """
    if not expected:
        return None
    actual_norm = actual or "<missing>"
    if actual_norm in expected:
        return _ok("intent_source_in", expected, actual_norm)
    return _fail(
        "intent_source_in", expected, actual_norm,
        f"intent source {actual_norm!r} not in allowed set {expected}",
    )
