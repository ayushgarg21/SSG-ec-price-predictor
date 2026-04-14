"""Tests for the FastAPI prediction endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_model() -> dict:
    """Mock appreciation model bundle."""
    model = MagicMock()
    model.predict.return_value = np.array([1.25])  # appreciation ratio

    scaler = MagicMock()
    scaler.transform.return_value = np.zeros((1, 24))

    q_lo = MagicMock()
    q_lo.predict.return_value = np.array([1.15])
    q_hi = MagicMock()
    q_hi.predict.return_value = np.array([1.35])

    return {
        "scaler": scaler,
        "model": model,
        "model_needs_scaling": True,
        "feature_columns": ["f"] * 24,
        "quantile_lower": q_lo,
        "quantile_upper": q_hi,
        "model_type": "appreciation",
        "launch_psm_lookup": {"TEST PROJECT": 12000.0},
        "global_mean_ratio": 1.25,
    }


@pytest.fixture
def mock_lookups():
    return {
        "district_stats": {},
        "ec_district_stats": {19.0: 14000.0},
        "project_stats": {},
        "district_momentum": {},
        "market_lag_psm": 15000.0,
        "project_target_enc": {},
        "global_mean_ratio": 1.25,
        "launch_psm_lookup": {"TEST PROJECT": 12000.0},
        "global_defaults": {
            "all_dist_med_psm": 15000.0,
            "all_dist_vol": 500.0,
            "all_dist_std": 3000.0,
            "ec_dist_med_psm": 14000.0,
            "district_momentum": 0.05,
            "market_lag_psm": 15000.0,
            "proj_lag_psm": 13000.0,
            "proj_lag_vol": 30.0,
        },
        "area_quartile_bins": [30, 80, 95, 110, 300],
        "latest_year": 2025,
    }


@pytest.fixture
def client(mock_model, mock_lookups):
    with patch("src.api.app._model", mock_model), \
         patch("src.api.app._explainer", None), \
         patch("src.api.app._model_version", "test-v1"), \
         patch("src.features.serving._lookups", mock_lookups):
        from src.api.app import app
        yield TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["active_run_id"] == "test-v1"


class TestPredictEndpoint:
    def test_predict_success(self, client: TestClient) -> None:
        payload = {
            "district": 19,
            "area_sqm": 95.0,
            "floor": 8,
            "lease_commence_year": 2018,
            "years_from_launch": 5,
            "sale_type": "Resale",
            "market_segment": "OCR",
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_price" in data
        assert "prediction_interval" in data
        assert data["prediction_interval"]["confidence"] == "80%"
        assert data["currency"] == "SGD"

    def test_predict_invalid_district(self, client: TestClient) -> None:
        payload = {
            "district": 99,
            "area_sqm": 95.0,
            "floor": 8,
            "lease_commence_year": 2018,
            "years_from_launch": 5,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_missing_field(self, client: TestClient) -> None:
        payload = {"district": 19, "area_sqm": 95.0}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


class TestMilestoneEndpoint:
    def test_milestones(self, client: TestClient) -> None:
        payload = {
            "district": 19,
            "area_sqm": 95.0,
            "floor": 8,
            "lease_commence_year": 2018,
            "market_segment": "OCR",
        }
        resp = client.post("/predict/milestones", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "mop_5yr_price" in data
        assert "mop_5yr_interval" in data
        assert "privatised_10yr_price" in data
        assert "privatised_10yr_interval" in data
        assert "price_appreciation" in data
        assert data["currency"] == "SGD"
