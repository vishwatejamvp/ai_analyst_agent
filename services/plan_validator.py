"""Plan validator — checks an :class:`AggregationSpec` against the live target
schema BEFORE the database is asked to execute it.

The router's job is to translate a question into an analytical *plan*. The
plan is then handed to Mongo / MySQL. Historically there has been no step
between those two: a wrong target or a wrong field would silently propagate
and explode inside the database driver (e.g. ``$dateToString`` on a non-date
field), producing a confusing "your terms aren't part of <wrong-target>"
message to the user.

This module is the missing step. Given a :class:`RoutingDecision` and a
schema snapshot (column names + a small document sample) it returns a
:class:`PlanValidationResult` with one of two effective verdicts:

* ``should_execute = True``  — the spec is consistent with the target (or has
  been *downgraded* into a still-meaningful smaller spec, e.g. time clause
  dropped because the chosen field is not date-coercible). Execution may
  proceed using ``result.decision``.
* ``should_execute = False`` — the spec cannot be salvaged (e.g. the chosen
  target is a catalog/glossary collection that doesn't carry facts, or the
  required metric isn't on the target at all). Execution must be skipped;
  the warnings are surfaced to the user.

Design rules:

* The validator never reads the database itself. The orchestrator passes in
  the schema snapshot so caching, error handling, and TTL stay in one place.
* All warnings use existing :class:`WarningCode` values so the trust panel
  can render them without enum churn.
* Downgrades return a *new* :class:`AggregationSpec` (via ``model_copy``);
  the original spec is never mutated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from models.enums import WarningCode
from models.schemas import AnalystWarning, RoutingDecision

# Catalog collections — never carry facts. Routing should not pick them for
# analytical questions; if it does, the validator refuses cleanly so the
# user gets a helpful message instead of a Mongo crash.
NON_FACT_COLLECTIONS: frozenset[str] = frozenset(
    {"awqaf_datasets_metadata", "awqaf_datasets_glossary"}
)

# String shapes accepted by the time-bucket expression in ``mongo_service``.
# Only the ``period`` field is permitted to use these (it is handled with
# ``$substr``, not ``$dateToString``). Other date fields must be real BSON
# Date instances; ISO strings on those fields will crash Mongo.
_YYYY_MM_DD_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_YYYY_MM_RE = re.compile(r"^\d{4}-\d{2}$")


@dataclass(frozen=True)
class PlanValidationResult:
    """Verdict + (possibly downgraded) decision + typed warnings."""

    decision: RoutingDecision
    warnings: list[AnalystWarning] = field(default_factory=list)
    should_execute: bool = True
    notes: list[str] = field(default_factory=list)


class PlanValidator:
    """Sanity-check an analytical plan against the live target schema.

    Call sequence on a "good" plan:
        validate() → returns result with ``should_execute=True`` and the
        original ``decision`` unchanged (no warnings).

    On a "salvageable" plan (e.g. time field on a target that has no usable
    date column):
        validate() → returns result with ``should_execute=True``, a *new*
        decision whose ``aggregation`` has the offending pieces dropped, and
        a typed warning explaining what was downgraded.

    On an "unsalvageable" plan (catalog target, or metric absent):
        validate() → ``should_execute=False`` plus a warning that the
        orchestrator surfaces to the user verbatim.
    """

    def __init__(
        self,
        *,
        non_fact_collections: frozenset[str] = NON_FACT_COLLECTIONS,
    ) -> None:
        self._non_fact = non_fact_collections

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def validate(
        self,
        decision: RoutingDecision,
        *,
        columns: list[str],
        sample: list[dict[str, Any]],
    ) -> PlanValidationResult:
        spec = decision.aggregation
        if spec is None or not decision.target:
            return PlanValidationResult(decision=decision)

        target = decision.target
        warnings: list[AnalystWarning] = []
        notes: list[str] = []

        # Rule 1 — Catalog / glossary collections can never carry facts.
        # This is the structural guard against the "router picked metadata"
        # failure mode: regardless of why the router landed there, we
        # refuse to ship an aggregation pipeline at it.
        if target in self._non_fact:
            warnings.append(
                AnalystWarning(
                    code=WarningCode.TARGET_AMBIGUOUS,
                    message=(
                        f"`{target}` is a catalog collection and does not "
                        "carry analytical facts. Please name a specific "
                        "dataset (e.g. `occupancy-rates-and-revenues`) or "
                        "rephrase the question."
                    ),
                    details={"target": target, "rule": "non_fact_collection"},
                )
            )
            return PlanValidationResult(
                decision=decision,
                warnings=warnings,
                should_execute=False,
                notes=[f"reject: target `{target}` is a catalog collection"],
            )

        if not columns:
            # No schema visible (collection empty / unreadable). Stay quiet
            # and let the regular empty-result path produce its own message.
            return PlanValidationResult(decision=decision)

        col_set = set(columns)
        new_spec = spec

        # Rule 2 — Metric must exist on the target.
        # Skip when the operation has no metric (count) or when the spec
        # didn't pin one (open-ended sum / avg fall through to executor).
        if (
            spec.operation != "count"
            and spec.metric
            and spec.metric not in col_set
        ):
            warnings.append(
                AnalystWarning(
                    code=WarningCode.METRIC_EMPTY,
                    message=(
                        f"Metric `{spec.metric}` does not exist on "
                        f"`{target}`. Showing the available data shape "
                        "instead so you can pick a metric the dataset "
                        "actually carries."
                    ),
                    details={"target": target, "missing_metric": spec.metric},
                )
            )
            notes.append(f"reject: metric `{spec.metric}` not in schema")
            return PlanValidationResult(
                decision=decision,
                warnings=warnings,
                should_execute=False,
                notes=notes,
            )

        # Rule 3 — group_by must exist; if not, drop it (downgrade).
        if spec.group_by and spec.group_by not in col_set:
            notes.append(f"downgrade: dropped group_by `{spec.group_by}`")
            warnings.append(
                AnalystWarning(
                    code=WarningCode.TARGET_AMBIGUOUS,
                    message=(
                        f"Group-by field `{spec.group_by}` is not present "
                        f"on `{target}`; dropped the group-by and reporting "
                        "the overall total instead."
                    ),
                    details={"dropped_group_by": spec.group_by},
                )
            )
            new_spec = new_spec.model_copy(update={"group_by": None})

        # Rule 4 — time field must exist AND be date-coercible.
        # This is the structural guard against the "$dateToString on a
        # non-date field" Mongo crash. Both sub-cases produce a downgrade,
        # not a refusal: the user still gets a non-time-bucketed total.
        if new_spec.time is not None:
            t_field = new_spec.time.field
            if t_field not in col_set:
                notes.append(f"downgrade: time.field `{t_field}` not in schema")
                warnings.append(
                    AnalystWarning(
                        code=WarningCode.TARGET_AMBIGUOUS,
                        message=(
                            f"Time field `{t_field}` is not on `{target}`; "
                            "falling back to a non-time-bucketed total."
                        ),
                        details={"dropped_time_field": t_field},
                    )
                )
                new_spec = new_spec.model_copy(update={"time": None})
            elif not _field_is_date_coercible(t_field, sample):
                notes.append(
                    f"downgrade: time.field `{t_field}` not date-coercible"
                )
                warnings.append(
                    AnalystWarning(
                        code=WarningCode.TARGET_AMBIGUOUS,
                        message=(
                            f"Field `{t_field}` on `{target}` is not a date "
                            "or year-month string, so a trend cannot be "
                            "drawn. Showing a non-time-bucketed total "
                            "instead."
                        ),
                        details={"dropped_time_field": t_field},
                    )
                )
                new_spec = new_spec.model_copy(update={"time": None})

        if new_spec is spec:
            return PlanValidationResult(decision=decision, notes=notes)

        new_decision = decision.model_copy(update={"aggregation": new_spec})
        return PlanValidationResult(
            decision=new_decision, warnings=warnings, notes=notes
        )


def _field_is_date_coercible(
    field_name: str, sample: list[dict[str, Any]]
) -> bool:
    """Return True iff at least one sample document has a value compatible
    with the way ``mongo_service._time_bucket_expr`` will handle ``field_name``.

    The mongo expression is **name-dispatched**, so coercibility is also
    name-dispatched here:

    * ``field_name == "period"`` → executor uses ``$substr``; a
      ``YYYY-MM`` (or longer ISO) **string** is acceptable.
    * ``field_name == "year"``   → executor uses ``$toString`` (or the
      AWQAF year-month bucket); an **integer** year in [1900, 2100] is
      acceptable.
    * any other field name       → executor uses **``$dateToString``**,
      which only accepts a real BSON Date. Strings that *look* like
      dates ("2024-01-15", ``ingested_at`` ISO datetimes, etc.) are
      **NOT** coercible here — Mongo raises ``Location4997901`` on them.

    The function is intentionally conservative: if no sample doc has the
    field at all, it returns False rather than guessing. A sparse field
    isn't useful for trends anyway, and a downgrade is far better UX
    than a database crash.
    """
    if not sample:
        return False
    is_period = field_name == "period"
    is_year = field_name == "year"
    for doc in sample:
        if field_name not in doc:
            continue
        value = doc.get(field_name)
        if value is None:
            continue
        if isinstance(value, bool):
            return False
        if isinstance(value, (datetime, date)):
            return True
        if is_year and isinstance(value, int):
            if 1900 <= value <= 2100:
                return True
            continue
        if is_period and isinstance(value, str):
            v = value.strip()
            if _YYYY_MM_RE.match(v) or _YYYY_MM_DD_RE.match(v):
                return True
            continue
        return False
    return False


plan_validator = PlanValidator()
