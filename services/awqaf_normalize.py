"""AWQAF JSON → flat row normalization (pure, no I/O).

Only **three** kinds of inputs are accepted now:

* ``all_records.json``  → facts (one Mongo doc per row, no record_kind mix)
* ``glossary.json``     → exactly one Mongo doc per dataset (fields + terms)
* ``contents.json``     → one Mongo doc per dataset (catalog metadata)

Per-year ``YYYY.json`` and other nested wrappers are intentionally ignored —
``all_records.json`` is the single canonical source for each service so we
never double-count monthly observations.

The flow stays:

    raw JSON
        │
        ▼
    services.awqaf_normalize.*           # this module (pure transforms)
        │
        ▼
    services.awqaf_ingest.*              # Mongo + FAISS writes
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Collection naming
# ---------------------------------------------------------------------------
_NAME_CLEAN_RE = re.compile(r"[^a-zA-Z0-9_]+")

DATASETS_METADATA_COLLECTION = "awqaf_datasets_metadata"
DATASETS_GLOSSARY_COLLECTION = "awqaf_datasets_glossary"


def _slug(name: str) -> str:
    cleaned = _NAME_CLEAN_RE.sub("_", str(name).strip().lower()).strip("_")
    return cleaned or "unknown"


def collection_name_from_service(service: str) -> str:
    """``hajj-package-service`` → ``awqaf_hajj_package_service_facts``.

    The ``_facts`` suffix signals: "this collection holds only measurement
    rows; no glossary, no metadata".
    """
    return f"awqaf_{_slug(service)}_facts"


def service_slug(service: str) -> str:
    """Public slug helper (used as the ``dataset`` join key everywhere)."""
    return _slug(service)


# ---------------------------------------------------------------------------
# Shared envelope
# ---------------------------------------------------------------------------
_MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_MONTH_TO_NUM = {name: i + 1 for i, name in enumerate(_MONTHS)}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_year(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit() and len(value) == 4:
        return int(value)
    return None


def _coerce_month(value: Any) -> tuple[str | None, int | None]:
    """Return ``(lower_name, month_num)`` if recognizable, else ``(None, None)``."""
    if value is None:
        return None, None
    s = str(value).strip().lower()
    if not s:
        return None, None
    if s in _MONTH_TO_NUM:
        return s, _MONTH_TO_NUM[s]
    if s.isdigit() and 1 <= int(s) <= 12:
        n = int(s)
        return _MONTHS[n - 1], n
    return s, None


# Priority order for promoting a natural slice column into ``dimension``.
# A row keeps the original column too — ``dimension`` is just a stable handle
# the router can group by without knowing the dataset's specific schema.
_DIMENSION_PRIORITY = (
    "emirate",
    "channel",
    "project",
    "country",
    "campaign_name",
    "campaign",
    "center",
    "mosque_name",
    "mosque",
    "area",
    "region",
    "category",
)


def _pick_dimension(row: dict[str, Any]) -> Any:
    for key in _DIMENSION_PRIORITY:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


# ---------------------------------------------------------------------------
# Schema consistency: ensure ``total_<X>`` exists when the channel components
# do. Without this, services that publish only ``website_transactions`` +
# ``smart_app_transactions`` (e.g. hajj-permit-service) cannot answer
# "total transactions" queries — the router resolves metric=None and the
# aggregation fails. After this backfill every transaction collection has a
# uniform ``total_transactions`` field so "total" questions just work.
# ---------------------------------------------------------------------------
_TOTAL_BACKFILLS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("total_transactions", ("website_transactions", "smart_app_transactions")),
)


def _backfill_totals(row: dict[str, Any]) -> None:
    for total_field, components in _TOTAL_BACKFILLS:
        existing = row.get(total_field)
        if isinstance(existing, (int, float)):
            continue  # don't overwrite a value the source already provided
        parts: list[float] = []
        for comp in components:
            value = row.get(comp)
            if isinstance(value, (int, float)):
                parts.append(float(value))
        if not parts:
            continue  # no components to sum — leave the field absent
        total = sum(parts)
        row[total_field] = int(total) if total.is_integer() else total


# ---------------------------------------------------------------------------
# 1. Facts (all_records.json)
# ---------------------------------------------------------------------------
def flatten_facts(
    records: list[Any],
    *,
    service: str,
    source_file: str | None = "all_records.json",
) -> list[dict[str, Any]]:
    """One Mongo doc per row in ``all_records.json``.

    The envelope is small and stable:

    * ``dataset``      — machine slug (folder name), used by metadata join
    * ``year``, ``month``, ``month_num``, ``period``  — present only when row is time-keyed
    * ``dimension``    — currently ``None``; reserved for future
                          ``emirate`` / ``channel`` / ``project`` slices that
                          may surface in non-time-series datasets
    * ``source_file``, ``ingested_at`` — provenance

    Domain metric fields from the original row are kept at the top level so
    aggregation pipelines and routing can address them directly.
    """
    if not isinstance(records, list):
        return []

    dataset = _slug(service)
    ingested_at = _now_iso()
    rows: list[dict[str, Any]] = []

    for raw in records:
        if not isinstance(raw, dict):
            continue

        row = dict(raw)
        row.pop("dataset", None)  # human label lives in metadata, not facts

        year = _coerce_year(row.pop("year", None))
        month_name, month_num = _coerce_month(row.pop("month", None))

        envelope: dict[str, Any] = {
            "dataset": dataset,
            "source_file": source_file,
            "ingested_at": ingested_at,
        }
        if year is not None:
            envelope["year"] = year
        if month_name is not None:
            envelope["month"] = month_name
        if month_num is not None:
            envelope["month_num"] = month_num
        if year is not None and month_num is not None:
            envelope["period"] = f"{year:04d}-{month_num:02d}"

        envelope["dimension"] = _pick_dimension(row)

        _backfill_totals(row)
        rows.append({**envelope, **row})

    return rows


# ---------------------------------------------------------------------------
# 2. Glossary (glossary.json) — one doc per dataset
# ---------------------------------------------------------------------------
def flatten_dataset_glossary(
    data: dict[str, Any],
    *,
    service: str,
) -> dict[str, Any] | None:
    """Return a single ``awqaf_datasets_glossary`` row for one service."""
    if not isinstance(data, dict):
        return None

    dataset = _slug(service)
    terms: list[dict[str, Any]] = []
    for t in data.get("terms") or []:
        if isinstance(t, dict):
            terms.append(
                {
                    "term": t.get("term"),
                    "definition": t.get("definition"),
                    "calculation": t.get("calculation"),
                    "unit": t.get("unit"),
                    "detail_level": t.get("detail_level"),
                    "data_source": t.get("data_source"),
                }
            )

    fields_raw = data.get("fields")
    fields: dict[str, Any] = {}
    if isinstance(fields_raw, dict):
        for fk, meta in fields_raw.items():
            if isinstance(meta, dict):
                fields[str(fk)] = {
                    "original_name": meta.get("original_name"),
                    "parent_group": meta.get("parent_group"),
                    "description": meta.get("description"),
                    "unit": meta.get("unit"),
                }
            else:
                fields[str(fk)] = {"description": str(meta) if meta is not None else None}

    indicator = data.get("indicator") if isinstance(data.get("indicator"), dict) else None
    source = data.get("source") if isinstance(data.get("source"), dict) else None
    coverage = data.get("coverage") if isinstance(data.get("coverage"), dict) else None

    return {
        "dataset": dataset,
        "dataset_name": data.get("dataset"),
        "indicator": indicator,
        "fields": fields,
        "terms": terms,
        "source": source,
        "coverage": coverage,
        "ingested_at": _now_iso(),
    }


# ---------------------------------------------------------------------------
# 3. Catalog metadata (contents.json) — one doc per dataset
# ---------------------------------------------------------------------------
def flatten_contents(data: Any) -> list[dict[str, Any]]:
    """Return ``awqaf_datasets_metadata`` rows from ``AWQAF-DATA/contents.json``.

    One row per ``datasets[]`` entry. The catalog header (title, source
    organization, etc.) is intentionally dropped — metadata reads target the
    per-dataset rows, never a "catalog root" sentinel.
    """
    if not isinstance(data, dict):
        return []

    rows: list[dict[str, Any]] = []
    ingested_at = _now_iso()

    for ds in data.get("datasets") or []:
        if not isinstance(ds, dict):
            continue
        folder = ds.get("folder")
        if not folder:
            continue
        dataset = _slug(folder)
        rows.append(
            {
                "dataset": dataset,
                "dataset_name": ds.get("dataset_name") or ds.get("original_name"),
                "folder": folder,
                "facts_collection": collection_name_from_service(folder),
                "department": ds.get("department"),
                "purpose": ds.get("purpose"),
                "data_type": ds.get("data_type"),
                "granularity": ds.get("granularity"),
                "years": ds.get("years") or [],
                "key_metrics": ds.get("key_metrics") or [],
                "coverage": ds.get("coverage"),
                "ingested_at": ingested_at,
            }
        )
    return rows
