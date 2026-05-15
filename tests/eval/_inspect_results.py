"""Throwaway debug helper: print the latest saved eval Result + trace.

Usage::

    python -m tests.eval._inspect_results
"""

from __future__ import annotations

import glob
import json
import sys


def main() -> int:
    import os
    files = sorted(
        glob.glob("tests/eval/results/*.json"),
        key=os.path.getmtime,
    )
    if not files:
        print("no result files in tests/eval/results/", file=sys.stderr)
        return 1
    latest = files[-1]
    print(f"reading: {latest}\n")
    data = json.loads(open(latest).read())
    if not data:
        print(f"{latest} is empty", file=sys.stderr)
        return 1
    r = data[0]
    summary = {
        k: r.get(k)
        for k in [
            "case_id",
            "passed",
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "cost_usd",
            "llm_calls",
        ]
    }
    print("Result summary:")
    print(json.dumps(summary, indent=2))

    t = r.get("trace") or {}
    spans = t.get("spans", [])
    print(f"\nTrace spans ({len(spans)}):")
    for s in spans:
        name = s["name"]
        dur = s["duration_ms"]
        parent = s["parent"]
        print(f"  {name:32} {dur:7.1f}ms  parent={parent}")

    usages = t.get("llm_usages", [])
    print(f"\nTrace LLM usages ({len(usages)}):")
    for u in usages:
        model = u["model"]
        cost = u["cost_usd"]
        lat = u["latency_ms"]
        print(
            f"  model={model} in={u['input_tokens']} out={u['output_tokens']} "
            f"cost=${cost:.6f} latency={lat:.1f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
