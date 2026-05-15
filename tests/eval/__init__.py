"""Evaluation harness for the AI Analyst Agent.

Why this package exists
-----------------------
Every prompt change, every router rule, every glossary tweak silently
shifts the agent's behaviour across hundreds of dimensions. Without a
golden dataset and an automated runner, "did this change make things
better or worse?" becomes a coin flip.

The harness is intentionally tiny:

* :mod:`tests.eval.cases`    — typed Case/Expect/Result schemas.
* :mod:`tests.eval.scorers`  — pure ``(expected, actual) -> Score`` fns.
* :mod:`tests.eval.runner`   — dispatches a Case to the right layer
  (intent / routing / e2e) and collects scores + latency.
* :mod:`tests.eval.report`   — terminal + JSON output.
* ``golden.jsonl``           — the dataset. One case per line, JSON.

Run::

    python -m tests.eval.runner                       # all cases
    python -m tests.eval.runner --level intent        # cheapest layer only
    python -m tests.eval.runner --tag follow-up
    python -m tests.eval.runner --strict --save out.json
"""
