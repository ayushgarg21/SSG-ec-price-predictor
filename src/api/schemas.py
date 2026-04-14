"""Request/response schemas for the prediction API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    district: int = Field(..., ge=1, le=28, description="Postal district (1-28)")
    area_sqm: float = Field(..., gt=0, description="Floor area in square metres")
    floor: int = Field(..., ge=1, le=70, description="Approximate floor level")
    lease_commence_year: int = Field(..., ge=1990, le=2030, description="Lease commencement year")
    years_from_launch: int = Field(
        ..., ge=0, le=99,
        description="Years since lease commencement (5 for MOP, 10 for privatisation)",
    )
    sale_type: Literal["New Sale", "Sub Sale", "Resale"] = "Resale"
    market_segment: Literal["OCR", "RCR", "CCR"] = "OCR"
    project: str | None = Field(None, description="EC project name (optional, improves accuracy)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "district": 19,
                    "area_sqm": 95.0,
                    "floor": 8,
                    "lease_commence_year": 2018,
                    "years_from_launch": 5,
                    "sale_type": "Resale",
                    "market_segment": "OCR",
                }
            ]
        }
    }


class PredictionInterval(BaseModel):
    lower_bound: float
    upper_bound: float
    confidence: str = "80%"


class FeatureContribution(BaseModel):
    base_value: float
    feature_contributions: dict[str, float]
    top_driver: str


class PredictionResponse(BaseModel):
    predicted_price: float
    prediction_interval: PredictionInterval
    currency: str = "SGD"
    input_features: PredictionRequest
    explanation: FeatureContribution | None = None


class MilestonePredictionRequest(BaseModel):
    district: int = Field(..., ge=1, le=28)
    area_sqm: float = Field(..., gt=0)
    floor: int = Field(..., ge=1, le=70)
    lease_commence_year: int = Field(..., ge=1990, le=2030)
    market_segment: Literal["OCR", "RCR", "CCR"] = "OCR"
    project: str | None = None


class MilestonePredictionResponse(BaseModel):
    mop_5yr_price: float
    mop_5yr_interval: PredictionInterval
    privatised_10yr_price: float
    privatised_10yr_interval: PredictionInterval
    price_appreciation: float
    appreciation_pct: float | None
    currency: str = "SGD"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    active_run_id: str | None = None
