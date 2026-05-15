"""Terminal + JSON reporting for eval results.

Two outputs:

* ``render_terminal`` — human-friendly pass/fail summary with per-case
  diffs on failures. Uses ANSI colour by default; pass ``color=False``
  for CI logs that would otherwise show escape codes.
* ``write_json`` — typed JSON dump (via Pydantic) suitable for diffing
  between runs and for archival in CI artifacts.

Aggregate metrics reported in the terminal:

* Per-level pass / fail / skip counts and pass-rate.
* Latency p50 + p95 per level.
* Overall verdict line at the bottom.

We intentionally compute p50/p95 by hand rather than pulling numpy in
just for two percentiles — the harness should stay zero-extra-deps.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .cases import EvalLevel, Result


# ---------------------------------------------------------------------------
# Colour helpers (ANSI; degrade to plain when color=False)
# ---------------------------------------------------------------------------
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"


def _c(text: str, code: str, *, color: bool) -> str:
    return f"{code}{text}{_RESET}" if color else text


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile. Empty input returns 0.0."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


# ---------------------------------------------------------------------------
# Terminal renderer
# ---------------------------------------------------------------------------
def render_terminal(results: list[Result], *, color: bool = True) -> str:
    by_level: dict[EvalLevel, list[Result]] = defaultdict(list)
    for r in results:
        by_level[r.level].append(r)

    lines: list[str] = []
    lines.append(_c("AI Analyst Agent — Eval Report", _BOLD, color=color))
    lines.append(_c("=" * 60, _DIM, color=color))

    for level in EvalLevel:
        level_results = by_level.get(level, [])
        if not level_results:
            continue
        lines.append("")
        lines.append(_c(f"[{level.value.upper()}]", _CYAN + _BOLD, color=color))
        for r in level_results:
            lines.append(_format_case_line(r, color=color))
            for s in r.scores:
                if not s.passed:
                    lines.append(
                        "    " + _c("✗", _RED, color=color)
                        + f" {s.name}: {s.message}"
                    )
            if r.error:
                tag = "skip" if r.skipped else "error"
                lines.append("    " + _c(f"[{tag}] {r.error}", _YELLOW, color=color))

        lines.append(_format_level_summary(level, level_results, color=color))

    lines.append("")
    lines.append(_c("=" * 60, _DIM, color=color))
    lines.append(_format_overall(results, color=color))
    return "\n".join(lines)


def _format_case_line(r: Result, *, color: bool) -> str:
    if r.skipped:
        marker = _c("○", _YELLOW, color=color)
        verdict = _c("SKIP", _YELLOW, color=color)
    elif r.passed:
        marker = _c("✓", _GREEN, color=color)
        verdict = _c("PASS", _GREEN, color=color)
    else:
        marker = _c("✗", _RED, color=color)
        verdict = _c("FAIL", _RED, color=color)
    cost_part = ""
    if r.llm_calls > 0:
        cost_part = _c(
            f"  · {r.input_tokens}/{r.output_tokens} tok  ${r.cost_usd:.4f}",
            "\033[2m",  # _DIM
            color=color,
        )
    return f"  {marker} {verdict}  {r.case_id:<32}  {r.latency_ms:7.1f} ms{cost_part}"


def _format_level_summary(
    level: EvalLevel, results: list[Result], *, color: bool
) -> str:
    total = len(results)
    skipped = sum(1 for r in results if r.skipped)
    runnable = total - skipped
    passed = sum(1 for r in results if r.passed and not r.skipped)
    failed = runnable - passed
    pass_rate = (passed / runnable * 100.0) if runnable else 0.0
    latencies = [r.latency_ms for r in results if not r.skipped]
    p50 = _percentile(latencies, 0.5)
    p95 = _percentile(latencies, 0.95)
    in_tok = sum(r.input_tokens for r in results)
    out_tok = sum(r.output_tokens for r in results)
    cost = sum(r.cost_usd for r in results)

    parts = [
        f"  {level.value} summary:",
        _c(f"{passed} pass", _GREEN, color=color),
        _c(f"{failed} fail", _RED if failed else _DIM, color=color),
        _c(f"{skipped} skip", _YELLOW if skipped else _DIM, color=color),
        f"({pass_rate:.1f}% of runnable)",
        f"p50 {p50:.0f} ms / p95 {p95:.0f} ms",
    ]
    if in_tok or out_tok:
        parts.append(f"in {in_tok} / out {out_tok} tok")
        parts.append(f"${cost:.4f}")
    return "  " + " · ".join(parts)


def _format_overall(results: list[Result], *, color: bool) -> str:
    total = len(results)
    skipped = sum(1 for r in results if r.skipped)
    runnable = total - skipped
    passed = sum(1 for r in results if r.passed and not r.skipped)
    failed = runnable - passed
    pass_rate = (passed / runnable * 100.0) if runnable else 0.0
    in_tok = sum(r.input_tokens for r in results)
    out_tok = sum(r.output_tokens for r in results)
    cost = sum(r.cost_usd for r in results)

    if failed == 0 and runnable > 0:
        verdict = _c("ALL GREEN", _GREEN + _BOLD, color=color)
    elif runnable == 0:
        verdict = _c("NOTHING RUN", _YELLOW + _BOLD, color=color)
    else:
        verdict = _c("REGRESSIONS PRESENT", _RED + _BOLD, color=color)

    cost_line = ""
    if in_tok or out_tok:
        cost_line = (
            f"  ·  total in {in_tok} / out {out_tok} tok  ·  "
            f"${cost:.4f} estimated"
        )

    return (
        f"{verdict}  —  total {total} · runnable {runnable} · "
        f"passed {passed} · failed {failed} · skipped {skipped} · "
        f"pass-rate {pass_rate:.1f}%{cost_line}"
    )


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------
def write_json(path: Path, results: list[Result]) -> None:
    """Dump results as a JSON array. Stable shape — diff-friendly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.model_dump(mode="json") for r in results]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
