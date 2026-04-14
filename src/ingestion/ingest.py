"""Ingest URA transactions into Postgres with data validation."""

from __future__ import annotations

from typing import Any

import pandas as pd
import structlog
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.ingestion.ura_client import URAClient

logger = structlog.get_logger()


def flatten_transactions(raw_projects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten nested URA response into one row per transaction."""
    rows: list[dict[str, Any]] = []
    for project in raw_projects:
        base: dict[str, Any] = {
            "project": project.get("project", ""),
            "street": project.get("street"),
            "x": _safe_float(project.get("x")),
            "y": _safe_float(project.get("y")),
            "market_segment": project.get("marketSegment"),
        }
        for txn in project.get("transaction", []):
            row = {
                **base,
                "area": _safe_float(txn.get("area")),
                "floor_range": txn.get("floorRange"),
                "no_of_units": _safe_int(txn.get("noOfUnits")),
                "contract_date": txn.get("contractDate"),
                "type_of_sale": txn.get("typeOfSale"),
                "price": _safe_float(txn.get("price")),
                "property_type": txn.get("propertyType"),
                "district": txn.get("district"),
                "type_of_area": txn.get("typeOfArea"),
                "tenure": txn.get("tenure"),
                "nett_price": _safe_float(txn.get("nettPrice")),
            }
            rows.append(row)
    return rows


def load_to_postgres(rows: list[dict[str, Any]], db: Session) -> int:
    """Validate and bulk insert flattened transactions into Postgres."""
    if not rows:
        return 0

    df = pd.DataFrame(rows)
    clean_rows = df.to_dict("records")

    if not clean_rows:
        logger.warning("no_valid_rows_after_validation")
        return 0

    insert_sql = text("""
        INSERT INTO ura_transactions
            (project, street, x, y, market_segment, area, floor_range,
             no_of_units, contract_date, type_of_sale, price, property_type,
             district, type_of_area, tenure, nett_price)
        VALUES
            (:project, :street, :x, :y, :market_segment, :area, :floor_range,
             :no_of_units, :contract_date, :type_of_sale, :price, :property_type,
             :district, :type_of_area, :tenure, :nett_price)
    """)

    db.execute(insert_sql, clean_rows)
    db.commit()
    logger.info("transactions_loaded", count=len(clean_rows))
    return len(clean_rows)


def run_ingestion(access_key: str, db: Session) -> int:
    """End-to-end: fetch from URA API and load into Postgres."""
    client = URAClient(access_key)
    raw = client.fetch_all_transactions()
    rows = flatten_transactions(raw)
    count = load_to_postgres(rows, db)
    logger.info("ingestion_complete", total_rows=count)
    return count


def _safe_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None
