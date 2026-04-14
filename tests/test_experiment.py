"""Tests for experiment tracking."""

from __future__ import annotations

from pathlib import Path
import tempfile

from src.model.experiment import log_experiment, load_experiments, get_best_run


class TestExperimentTracking:
    def test_log_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "experiments.jsonl"
            run_id = log_experiment(
                algorithm="xgboost",
                parameters={"n_estimators": 100},
                metrics={"r2": 0.85, "mae": 50000},
                feature_columns=["a", "b"],
                train_size=800,
                test_size=200,
                artifact_path="/tmp/model.joblib",
                log_path=log_path,
            )
            assert len(run_id) == 8

            experiments = load_experiments(log_path)
            assert len(experiments) == 1
            assert experiments[0]["algorithm"] == "xgboost"
            assert experiments[0]["metrics"]["r2"] == 0.85

    def test_get_best_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "experiments.jsonl"
            for algo, r2 in [("xgboost", 0.85), ("gbr", 0.90), ("xgboost", 0.88)]:
                log_experiment(
                    algorithm=algo,
                    parameters={},
                    metrics={"r2": r2},
                    feature_columns=[],
                    train_size=800,
                    test_size=200,
                    artifact_path="/tmp/model.joblib",
                    log_path=log_path,
                )
            best = get_best_run(metric="r2", log_path=log_path)
            assert best is not None
            assert best["algorithm"] == "gbr"
            assert best["metrics"]["r2"] == 0.90

    def test_empty_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "nonexistent.jsonl"
            assert load_experiments(log_path) == []
            assert get_best_run(log_path=log_path) is None
