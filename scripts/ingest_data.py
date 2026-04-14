#!/usr/bin/env python3
"""CLI script to ingest URA transactions into Postgres."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import get_settings
from src.database import get_db
from src.ingestion.ingest import run_ingestion


def main() -> None:
    settings = get_settings()
    if not settings.ura_access_key:
        print("ERROR: Set URA_ACCESS_KEY in .env or environment")
        sys.exit(1)

    with get_db() as db:
        count = run_ingestion(settings.ura_access_key, db)
        print(f"Ingested {count} validated transactions into Postgres.")


if __name__ == "__main__":
    main()
