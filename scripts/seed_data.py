"""Generate a sample Excel file and run the full ingestion pipeline.

This is the easiest way to see the system end-to-end without an Excel of
your own. Produces ``data/uploads/sample_sales.xlsx`` and ingests it into
Mongo + FAISS under the ``sample_sales`` collection.

Usage:

    python -m scripts.seed_data
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from services.ingestion_service import ingestion_service  # noqa: E402
from models.config import settings  # noqa: E402
from utils.logger import logger  # noqa: E402

ROWS = [
    {"region": "North", "product": "Widget A", "category": "Hardware",
     "amount": 1200.00, "quantity": 5,  "sold_at": "2024-01-15",
     "notes": "Strong start in northern markets, repeat customer."},
    {"region": "North", "product": "Widget B", "category": "Hardware",
     "amount": 2500.50, "quantity": 8,  "sold_at": "2024-01-22",
     "notes": "Bulk order from a long-time enterprise client."},
    {"region": "South", "product": "Widget A", "category": "Hardware",
     "amount": 900.00,  "quantity": 3,  "sold_at": "2024-02-05",
     "notes": "Slow uptake; pricing pressure from local competitors."},
    {"region": "South", "product": "Gizmo X", "category": "Software",
     "amount": 4500.00, "quantity": 2,  "sold_at": "2024-02-18",
     "notes": "Highest software deal of Q1; renewal expected."},
    {"region": "East",  "product": "Gizmo X", "category": "Software",
     "amount": 4200.00, "quantity": 2,  "sold_at": "2024-03-04",
     "notes": "Procurement cycle finally closed; champion at customer."},
    {"region": "East",  "product": "Service Pro", "category": "Service",
     "amount": 3300.00, "quantity": 1,  "sold_at": "2024-03-19",
     "notes": "Onboarding services package; expansion likely."},
    {"region": "West",  "product": "Widget B", "category": "Hardware",
     "amount": 1800.00, "quantity": 6,  "sold_at": "2024-04-02",
     "notes": "Channel partner pull; demo drove conversion."},
    {"region": "West",  "product": "Service Pro", "category": "Service",
     "amount": 2700.00, "quantity": 1,  "sold_at": "2024-04-21",
     "notes": "Quarterly retainer; stable but flat."},
]


def main() -> None:
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = upload_dir / "sample_sales.xlsx"

    df = pd.DataFrame(ROWS)
    df.to_excel(xlsx_path, index=False)
    logger.info(f"Wrote sample workbook -> {xlsx_path}")

    response = ingestion_service.ingest_file(
        xlsx_path, collection="sample_sales", replace=True
    )
    logger.info(
        f"Ingested {response.rows_ingested} rows, "
        f"{response.vectors_indexed} vectors into '{response.collection}'"
    )


if __name__ == "__main__":
    main()
