"""Lightweight experiment tracking — logs every training run to a local JSONL file."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

DEFAULT_LOG_PATH = Path("./artifacts/experiments.jsonl")


def log_experiment(
    algorithm: str,
    parameters: dict[str, Any],
    metrics: dict[str, float],
    feature_columns: list[str],
    train_size: int,
    test_size: int,
    artifact_path: str,
    log_path: Path = DEFAULT_LOG_PATH,
) -> str:
    """Append a training run record to the experiment log.

    Returns:
        The generated run_id.
    """
    run_id = str(uuid.uuid4())[:8]
    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "algorithm": algorithm,
        "parameters": parameters,
        "metrics": metrics,
        "feature_columns": feature_columns,
        "train_size": train_size,
        "test_size": test_size,
        "artifact_path": artifact_path,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    logger.info("experiment_logged", run_id=run_id, algorithm=algorithm, r2=metrics.get("r2"))
    return run_id


def load_experiments(log_path: Path = DEFAULT_LOG_PATH) -> list[dict[str, Any]]:
    """Load all experiment records from the JSONL log."""
    if not log_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_best_run(
    metric: str = "r2",
    higher_is_better: bool = True,
    log_path: Path = DEFAULT_LOG_PATH,
) -> dict[str, Any] | None:
    """Return the experiment run with the best value for the given metric."""
    experiments = load_experiments(log_path)
    if not experiments:
        return None
    return sorted(
        experiments,
        key=lambda r: r["metrics"].get(metric, float("-inf")),
        reverse=higher_is_better,
    )[0]
