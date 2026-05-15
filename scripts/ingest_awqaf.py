"""CLI ingest of ``AWQAF-DATA/`` from disk.

Disk traversal only — flatten rules live in :mod:`services.awqaf_normalize`
and writes go through :mod:`services.awqaf_ingest` so the same logic applies
to a future HTTP ingest endpoint.

Sources used (per service folder):

* ``all_records.json``  → ``awqaf_<service>_facts``    (canonical observations)
* ``glossary.json``     → ``awqaf_datasets_glossary``  (one doc per dataset)

Plus, once at the root:

* ``contents.json``     → ``awqaf_datasets_metadata``  (one doc per dataset)

Per-year ``YYYY.json`` files and directory wrappers are **ignored** —
``all_records.json`` is the single canonical source.

Usage (from repo root, venv active)::

    python -m scripts.ingest_awqaf
    python -m scripts.ingest_awqaf --replace
    python -m scripts.ingest_awqaf --drop-all
    python -m scripts.ingest_awqaf --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.awqaf_ingest import (  # noqa: E402
    dataset_label_for,
    insert_dataset_glossary,
    insert_datasets_metadata,
    insert_facts,
)
from services.mongo_service import mongo_service  # noqa: E402
from services.vector_service import vector_service  # noqa: E402
from utils.logger import logger  # noqa: E402


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _drop_awqaf_collections() -> list[str]:
    """Drop every ``awqaf_*`` Mongo collection. Returns names that were dropped."""
    existing = mongo_service.list_collections()
    awqaf = [c for c in existing if c.startswith("awqaf_")]
    for name in awqaf:
        mongo_service.drop_collection(name)
        logger.warning(f"dropped collection {name}")
    return awqaf


def _ingest_service(
    service_dir: Path,
    *,
    replace: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    service = service_dir.name
    summaries: list[dict[str, Any]] = []

    ar = service_dir / "all_records.json"
    gl = service_dir / "glossary.json"

    if ar.is_file():
        data = _load_json(ar)
        if isinstance(data, list):
            if dry_run:
                summaries.append(
                    {
                        "collection": f"awqaf_{service.replace('-', '_')}_facts",
                        "rows_normalized": len(data),
                        "documents_inserted": 0,
                        "vectors_indexed": 0,
                        "dry_run": True,
                    }
                )
            else:
                summaries.append(
                    insert_facts(
                        service,
                        data,
                        replace=replace,
                        dataset_label=dataset_label_for(service),
                    )
                )
        else:
            logger.warning(f"{ar}: expected JSON array, got {type(data).__name__}")
    else:
        logger.info(f"[{service}] no all_records.json, skipping facts")

    if gl.is_file():
        data = _load_json(gl)
        if isinstance(data, dict):
            if dry_run:
                summaries.append(
                    {
                        "collection": "awqaf_datasets_glossary",
                        "documents_inserted": 0,
                        "vectors_indexed": 0,
                        "dry_run": True,
                    }
                )
            else:
                summaries.append(insert_dataset_glossary(service, data))
        else:
            logger.warning(f"{gl}: expected JSON object, got {type(data).__name__}")

    return summaries


def _ingest_root_contents(
    data_dir: Path,
    *,
    replace: bool,
    dry_run: bool,
) -> dict[str, Any]:
    path = data_dir / "contents.json"
    if not path.is_file():
        return {
            "collection": "awqaf_datasets_metadata",
            "rows_normalized": 0,
            "documents_inserted": 0,
        }
    data = _load_json(path)
    if dry_run:
        n = len((data or {}).get("datasets") or []) if isinstance(data, dict) else 0
        return {
            "collection": "awqaf_datasets_metadata",
            "rows_normalized": n,
            "documents_inserted": 0,
            "dry_run": True,
        }
    return insert_datasets_metadata(data, replace=replace)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest AWQAF-DATA into Mongo + FAISS")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_ROOT / "AWQAF-DATA",
        help="Path to AWQAF-DATA folder",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Drop each facts/metadata collection before insert (no glossary drop; glossary is always upserted)",
    )
    parser.add_argument(
        "--drop-all",
        action="store_true",
        help="Drop ALL awqaf_* Mongo collections and reset FAISS before ingest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse only; no Mongo/FAISS writes",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    if args.drop_all and not args.dry_run:
        dropped = _drop_awqaf_collections()
        vector_service.reset()
        logger.warning(
            f"Dropped {len(dropped)} awqaf_* collection(s) and reset FAISS index"
        )

    summaries: list[dict[str, Any]] = []
    summaries.append(
        _ingest_root_contents(data_dir, replace=args.replace, dry_run=args.dry_run)
    )

    for entry in sorted(data_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        summaries.extend(
            _ingest_service(entry, replace=args.replace, dry_run=args.dry_run)
        )

    total_rows = sum(s.get("rows_normalized", 0) for s in summaries)
    total_docs = sum(s.get("documents_inserted", 0) for s in summaries)
    total_vecs = sum(s.get("vectors_indexed", 0) for s in summaries)
    collections_touched = sorted(
        {s["collection"] for s in summaries if s.get("collection")}
    )

    print(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "dry_run": args.dry_run,
                "rows_normalized": total_rows,
                "documents_inserted": total_docs,
                "vectors_indexed": total_vecs,
                "faiss_size": 0 if args.dry_run else vector_service.size,
                "collections_touched": collections_touched,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
