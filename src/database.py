"""Database engine, session management, and prediction logging."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Generator

import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings

logger = structlog.get_logger()

engine = create_engine(
    get_settings().database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=5,
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_dependency() -> Generator[Session, None, None]:
    """FastAPI dependency for request-scoped sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def log_prediction(
    db: Session,
    model_version: str,
    input_features: dict[str, Any],
    predicted_price: float,
    latency_ms: float,
) -> None:
    """Write a prediction to the prediction_logs table for monitoring."""
    try:
        db.execute(
            text("""
                INSERT INTO prediction_logs (model_version, input_features, predicted_price, latency_ms)
                VALUES (:model_version, :input_features, :predicted_price, :latency_ms)
            """),
            {
                "model_version": model_version,
                "input_features": json.dumps(input_features),
                "predicted_price": predicted_price,
                "latency_ms": latency_ms,
            },
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.warning("prediction_log_failed", exc_info=True)
