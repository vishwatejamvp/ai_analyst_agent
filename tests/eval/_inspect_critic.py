"""Inspect the latest critic findings for one question.

Runs one orchestrator request with the critic enabled and prints the
draft narrative + each flagged issue + the revised narrative.
Useful for deciding whether the critic is catching real errors or
false-positiving on faithful drafts.

Usage::

    python -m tests.eval._inspect_critic "average occupancy rate in 2024"
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m tests.eval._inspect_critic <question>", file=sys.stderr)
        return 1
    question = sys.argv[1]

    from models.config import get_settings
    get_settings().critic_enabled = True

    from models.schemas import QueryRequest
    from services.analyst_service import analyst_orchestrator
    from utils.observability import TraceContext, use_trace

    ctx = TraceContext.start(request_id="inspect_critic")
    with use_trace(ctx):
        resp = analyst_orchestrator.answer(
            QueryRequest(question=question, include_details=False)
        )

    critic = (resp.meta or {}).get("critic") or {}
    print("=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)
    print("\n--- DRAFT (revised, what the user would see) ---")
    print(resp.insight)
    print("\n--- CRITIC ACTION ---")
    print(f"action       : {critic.get('action')}")
    print(f"issue_count  : {critic.get('issue_count')}")
    print(f"summary      : {critic.get('summary')}")
    issues = critic.get("issues") or []
    if issues:
        print("\n--- CRITIC FINDINGS ---")
        for i, issue in enumerate(issues, start=1):
            print(f"\n[{i}] severity={issue.get('severity')} type={issue.get('type')}")
            print(f"    quote        : {issue.get('quote')}")
            print(f"    evidence     : {issue.get('evidence')}")
            print(f"    suggested_fix: {issue.get('suggested_fix')}")
    print("\n--- LLM CALLS ---")
    for u in ctx.llm_usages:
        print(
            f"  {u.span_name or '<unnamed>':22}  "
            f"in={u.input_tokens:>5} out={u.output_tokens:>5}  "
            f"${u.cost_usd:.4f}  {u.latency_ms:.0f}ms"
        )
    print(f"\nstructured_data rows: {len(resp.structured_data)}")
    if resp.structured_data:
        print("first 3 rows:")
        print(json.dumps(resp.structured_data[:3], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
