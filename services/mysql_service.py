"""MySQL service: read-only query execution + safe aggregation builder.

Two ways to use it:

1. ``run_aggregation(table, spec)`` — translate an :class:`AggregationSpec`
   into a SELECT (safe, identifier-quoted, parameterised).
2. ``run_sql(sql)`` — execute an arbitrary SQL string. Only ``SELECT``
   statements are allowed; everything else raises :class:`UnsafeQueryError`.

The LLM never writes SQL directly — the routing service uses the
aggregation builder to produce vetted queries.
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from models.enums import TimeBucket
from models.schemas import AggregationSpec, TimeSpec
from models.config import settings
from utils.exceptions import DatabaseError, UnsafeQueryError
from utils.logger import logger

_OP_TO_SQL = {
    "sum": "SUM",
    "avg": "AVG",
    "average": "AVG",
    "mean": "AVG",
    "min": "MIN",
    "max": "MAX",
    "count": "COUNT",
}

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|"
    r"replace|rename|merge|call|exec|execute|load|handler)\b",
    re.IGNORECASE,
)


def _quote_ident(name: str) -> str:
    if not _IDENT_RE.match(name or ""):
        raise UnsafeQueryError(f"Invalid SQL identifier: {name!r}")
    return f"`{name}`"


class MySQLService:
    """Lightweight SQLAlchemy wrapper, read-only by default."""

    def __init__(self, url: str | None = None) -> None:
        self.url = url or settings.mysql_url
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            try:
                self._engine = create_engine(
                    self.url,
                    pool_pre_ping=True,
                    pool_recycle=1800,
                    future=True,
                )
                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("Connected to MySQL")
            except SQLAlchemyError as exc:
                raise DatabaseError(f"MySQL connection failed: {exc}") from exc
        return self._engine

    def dispose(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def list_tables(self) -> list[str]:
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text("SHOW TABLES")).fetchall()
            return sorted(r[0] for r in rows)
        except SQLAlchemyError as exc:
            raise DatabaseError(f"MySQL list_tables failed: {exc}") from exc

    # ------------------------------------------------------------------
    # SQL execution (read-only)
    # ------------------------------------------------------------------
    def run_sql(
        self,
        sql: str,
        params: dict[str, Any] | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """Run a *read-only* SQL ``SELECT`` and return rows as dicts."""
        if not sql or not sql.strip():
            raise UnsafeQueryError("Empty SQL")

        cleaned = sql.strip().rstrip(";")
        if not re.match(r"^\s*select\b", cleaned, re.IGNORECASE):
            raise UnsafeQueryError("Only SELECT statements are allowed")
        if _FORBIDDEN.search(cleaned):
            raise UnsafeQueryError("Forbidden SQL keyword detected")
        if ";" in cleaned:
            raise UnsafeQueryError("Multiple statements are not allowed")

        if not re.search(r"\blimit\b", cleaned, re.IGNORECASE):
            cleaned = f"{cleaned} LIMIT {int(limit)}"

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(cleaned), params or {})
                return [dict(row._mapping) for row in result.fetchall()]
        except SQLAlchemyError as exc:
            raise DatabaseError(f"MySQL query failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Safe aggregation builder
    # ------------------------------------------------------------------
    def run_aggregation(
        self,
        table: str,
        spec: AggregationSpec,
    ) -> list[dict[str, Any]]:
        """Build and execute a parameterised aggregation SELECT.

        When :attr:`AggregationSpec.time` is set the SELECT is bucketed by
        the time field (label is the bucket string), sorted ascending.
        ``group_by`` is ignored in that case to keep a single-series row
        contract for the chart layer.
        """
        sql, params = self.build_sql(table, spec)
        logger.debug(f"MySQL aggregation: {sql} -- params={params}")
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql), params)
                return [dict(row._mapping) for row in result.fetchall()]
        except SQLAlchemyError as exc:
            raise DatabaseError(f"MySQL aggregation failed: {exc}") from exc

    def build_sql(
        self,
        table: str,
        spec: AggregationSpec,
    ) -> tuple[str, dict[str, Any]]:
        """Pure compile of an :class:`AggregationSpec` to (SQL, bound params).

        Used by :meth:`run_aggregation` and by provenance previews.
        """
        op = (spec.operation or "").lower().strip()
        if op not in _OP_TO_SQL:
            raise DatabaseError(f"Unsupported aggregation operation: {spec.operation}")

        table_q = _quote_ident(table)
        if op == "count":
            metric_expr = "COUNT(*)"
        else:
            if not spec.metric:
                raise DatabaseError(f"Aggregation '{op}' requires a metric column")
            metric_expr = f"{_OP_TO_SQL[op]}({_quote_ident(spec.metric)})"

        select_parts = [f"{metric_expr} AS `value`"]
        group_by = ""

        if spec.time is not None:
            time_label_expr = _time_bucket_sql(spec.time)
            select_parts.insert(0, f"{time_label_expr} AS `label`")
            group_by = f" GROUP BY {time_label_expr}"
        elif spec.group_by:
            group_q = _quote_ident(spec.group_by)
            select_parts.insert(0, f"{group_q} AS `label`")
            group_by = f" GROUP BY {group_q}"
        else:
            select_parts.insert(0, "'(all)' AS `label`")

        where_sql, params = _build_where(spec.filters)
        if spec.time is not None:
            time_where, time_params = _time_range_where(spec.time, len(params))
            if time_where:
                where_sql = (
                    f"{where_sql} AND {time_where}"
                    if where_sql
                    else f" WHERE {time_where}"
                )
                params.update(time_params)

        if spec.time is not None:
            order_by = " ORDER BY `label` ASC"
        elif spec.order_by and spec.order_by != "value":
            order_by = f" ORDER BY {_quote_ident(spec.order_by)} ASC"
        else:
            order_by = " ORDER BY `value` DESC"

        limit_sql = (
            "" if spec.time is not None
            else (f" LIMIT {int(spec.limit)}" if spec.limit else " LIMIT 1000")
        )

        sql = (
            f"SELECT {', '.join(select_parts)} FROM {table_q}"
            f"{where_sql}{group_by}{order_by}{limit_sql}"
        )
        return sql, params

    def latest_value(self, table: str, column: str) -> Any:
        """Return MAX(``column``) from ``table`` (None if empty / on error)."""
        col_q = _quote_ident(column)
        tbl_q = _quote_ident(table)
        sql = (
            f"SELECT {col_q} AS `v` FROM {tbl_q} "
            f"WHERE {col_q} IS NOT NULL "
            f"ORDER BY {col_q} DESC LIMIT 1"
        )
        try:
            with self.engine.connect() as conn:
                row = conn.execute(text(sql)).fetchone()
        except SQLAlchemyError as exc:
            raise DatabaseError(f"MySQL latest_value failed: {exc}") from exc
        if row is None:
            return None
        return row._mapping.get("v")

    def earliest_value(self, table: str, column: str) -> Any:
        """Return MIN(``column``) from ``table`` (None if empty / on error)."""
        col_q = _quote_ident(column)
        tbl_q = _quote_ident(table)
        sql = (
            f"SELECT {col_q} AS `v` FROM {tbl_q} "
            f"WHERE {col_q} IS NOT NULL "
            f"ORDER BY {col_q} ASC LIMIT 1"
        )
        try:
            with self.engine.connect() as conn:
                row = conn.execute(text(sql)).fetchone()
        except SQLAlchemyError as exc:
            raise DatabaseError(f"MySQL earliest_value failed: {exc}") from exc
        if row is None:
            return None
        return row._mapping.get("v")

    def distinct_years(self, table: str, column: str) -> list[int]:
        """Return sorted distinct calendar years present in ``table.column``.

        Counterpart to :meth:`MongoService.distinct_years` for the SQL
        path. Treats ``year`` / ``period`` columns specially (same
        AWQAF storage convention) and falls back to ``YEAR(<col>)`` for
        real dates. Fails *closed* (returns ``[]``) on any DB error so
        the calling "what years do you actually have?" probe degrades
        gracefully instead of misreporting coverage.
        """
        col_q = _quote_ident(column)
        tbl_q = _quote_ident(table)
        if column == "year":
            year_expr = col_q
        elif column == "period":
            year_expr = f"CAST(SUBSTRING({col_q}, 1, 4) AS UNSIGNED)"
        else:
            year_expr = f"YEAR({col_q})"
        sql = (
            f"SELECT DISTINCT {year_expr} AS `y` FROM {tbl_q} "
            f"WHERE {col_q} IS NOT NULL "
            f"ORDER BY `y` ASC LIMIT 50"
        )
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text(sql)).fetchall()
        except SQLAlchemyError as exc:
            logger.warning(
                f"MySQL distinct_years on {table}.{column} failed "
                f"({type(exc).__name__}: {exc}); returning empty."
            )
            return []
        years: list[int] = []
        for row in rows:
            val = row._mapping.get("y")
            try:
                years.append(int(val))
            except (TypeError, ValueError):
                continue
        return years


_BUCKET_SQL = {
    TimeBucket.DAY: "DATE_FORMAT({col}, '%Y-%m-%d')",
    TimeBucket.MONTH: "DATE_FORMAT({col}, '%Y-%m')",
    TimeBucket.YEAR: "DATE_FORMAT({col}, '%Y')",
    TimeBucket.WEEK: "DATE_FORMAT({col}, '%x-W%v')",
    TimeBucket.QUARTER: "CONCAT(YEAR({col}), '-Q', QUARTER({col}))",
}


def _time_bucket_sql(time: TimeSpec) -> str:
    if time.field == "year":
        y = _quote_ident(time.field)
        if time.bucket == TimeBucket.MONTH:
            mcol = _quote_ident("month")
            return (
                f"CONCAT(CAST({y} AS CHAR), '-', LPAD(CAST(MONTH(STR_TO_DATE("
                f"CONCAT('1 ', LOWER({mcol})), '%d %M')) AS CHAR), 2, '0'))"
            )
        return f"CAST({y} AS CHAR)"
    template = _BUCKET_SQL.get(time.bucket, _BUCKET_SQL[TimeBucket.MONTH])
    return template.format(col=_quote_ident(time.field))


def _time_range_where(
    time: TimeSpec, base_index: int
) -> tuple[str, dict[str, Any]]:
    parts: list[str] = []
    params: dict[str, Any] = {}
    col = _quote_ident(time.field)
    if time.field == "year":
        if time.range_from is not None:
            pname = f"p_t{base_index}_from"
            parts.append(f"{col} >= :{pname}")
            params[pname] = time.range_from.year
        if time.range_to is not None:
            pname = f"p_t{base_index}_to"
            parts.append(f"{col} <= :{pname}")
            params[pname] = time.range_to.year
        return (" AND ".join(parts), params) if parts else ("", {})

    if time.range_from is not None:
        pname = f"p_t{base_index}_from"
        parts.append(f"{col} >= :{pname}")
        params[pname] = time.range_from
    if time.range_to is not None:
        pname = f"p_t{base_index}_to"
        parts.append(f"{col} <= :{pname}")
        params[pname] = time.range_to
    return (" AND ".join(parts), params) if parts else ("", {})


def _build_where(filters: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Turn a flat filter dict into a parameterised WHERE clause."""
    if not filters:
        return "", {}

    clauses: list[str] = []
    params: dict[str, Any] = {}
    for i, (key, value) in enumerate(filters.items()):
        col = _quote_ident(key)
        if isinstance(value, dict):
            for op_key, op_val in value.items():
                op_norm = {
                    ">": ">",
                    ">=": ">=",
                    "<": "<",
                    "<=": "<=",
                    "!=": "<>",
                    "like": "LIKE",
                }.get(op_key.lower(), None)
                if op_norm is None:
                    raise UnsafeQueryError(f"Unsupported filter operator: {op_key}")
                pname = f"p_{i}_{op_key}"
                clauses.append(f"{col} {op_norm} :{pname}")
                params[pname] = op_val
        else:
            pname = f"p_{i}"
            clauses.append(f"{col} = :{pname}")
            params[pname] = value
    return " WHERE " + " AND ".join(clauses), params


mysql_service = MySQLService()
