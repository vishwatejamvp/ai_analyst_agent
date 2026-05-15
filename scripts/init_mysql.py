"""Bootstrap a MySQL schema for demos / staging.

Creates a small `sales` table and inserts a few rows so the analytical
path has data to query against. Safe to re-run (uses ``IF NOT EXISTS``
and replaces seed rows).

Usage:

    python -m scripts.init_mysql
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from services.mysql_service import mysql_service  # noqa: E402
from utils.logger import logger  # noqa: E402

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS `sales` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `region` VARCHAR(64) NOT NULL,
    `product` VARCHAR(128) NOT NULL,
    `category` VARCHAR(64) NOT NULL,
    `amount` DECIMAL(12,2) NOT NULL,
    `quantity` INT NOT NULL,
    `sold_at` DATE NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

SEED_ROWS = [
    ("North", "Widget A", "Hardware", 1200.00, 5, "2024-01-15"),
    ("North", "Widget B", "Hardware", 2500.50, 8, "2024-01-22"),
    ("South", "Widget A", "Hardware", 900.00, 3, "2024-02-05"),
    ("South", "Gizmo X", "Software", 4500.00, 2, "2024-02-18"),
    ("East",  "Gizmo X", "Software", 4200.00, 2, "2024-03-04"),
    ("East",  "Service Pro", "Service", 3300.00, 1, "2024-03-19"),
    ("West",  "Widget B", "Hardware", 1800.00, 6, "2024-04-02"),
    ("West",  "Service Pro", "Service", 2700.00, 1, "2024-04-21"),
]


def main() -> None:
    engine = mysql_service.engine
    with engine.begin() as conn:
        conn.execute(text(CREATE_SQL))
        conn.execute(text("DELETE FROM `sales`"))
        conn.execute(
            text(
                "INSERT INTO `sales` (region, product, category, amount, quantity, sold_at) "
                "VALUES (:region, :product, :category, :amount, :quantity, :sold_at)"
            ),
            [
                dict(
                    region=r,
                    product=p,
                    category=c,
                    amount=a,
                    quantity=q,
                    sold_at=d,
                )
                for (r, p, c, a, q, d) in SEED_ROWS
            ],
        )
    logger.info(f"Seeded `sales` with {len(SEED_ROWS)} rows")


if __name__ == "__main__":
    main()
