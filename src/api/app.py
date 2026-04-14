"""FastAPI application for EC price prediction.

Integrates: prediction logging, SHAP explainability, caching, rate limiting,
prediction intervals, leakage-free feature serving.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from src.api.cache import PredictionCache
from src.api.rate_limit import SlidingWindowRateLimiter
from src.api.schemas import (
    FeatureContribution,
    HealthResponse,
    MilestonePredictionRequest,
    MilestonePredictionResponse,
    PredictionInterval,
    PredictionRequest,
    PredictionResponse,
)
from src.config import get_settings
from src.database import get_db_dependency, log_prediction
from src.features.serving import build_serving_features, load_lookups
from src.model.explain import ModelExplainer
from src.model.predict import predict_at_milestones, predict_price
from src.model.train import load_model

logger = structlog.get_logger()

_model: Any = None
_explainer: ModelExplainer | None = None
_model_version: str = "unknown"
_cache = PredictionCache(ttl_seconds=get_settings().cache_ttl_seconds)
_rate_limiter = SlidingWindowRateLimiter(max_requests=get_settings().rate_limit_rpm)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _model, _explainer, _model_version
    settings = get_settings()
    try:
        _model = load_model(settings.model_path)
        _explainer = ModelExplainer(_model)

        # Load serving lookups for feature construction
        lookups_path = settings.model_path.replace("model.joblib", "serving_lookups.joblib")
        try:
            load_lookups(lookups_path)
        except FileNotFoundError:
            logger.warning("serving_lookups_not_found", path=lookups_path)

        meta_path = settings.model_path.replace("model.joblib", "model_metadata.json")
        try:
            with open(meta_path) as f:
                meta = json.load(f)
                _model_version = meta.get("run_id", "unknown")
        except FileNotFoundError:
            _model_version = "unknown"
        logger.info("model_loaded", path=settings.model_path, version=_model_version)
    except FileNotFoundError:
        logger.warning("model_not_found", path=settings.model_path)
    yield
    _model = None
    _explainer = None


app = FastAPI(
    title="EC Price Prediction API",
    description="Predict Executive Condominium resale prices at MOP (5yr) and privatisation (10yr) with prediction intervals",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_model() -> Any:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    return _model


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        version="1.0.0",
        active_run_id=_model_version if _model else None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    req: Request = None,
    db: Session = Depends(get_db_dependency),
) -> PredictionResponse:
    _rate_limiter.check(req)
    model = _ensure_model()

    features_dict = request.model_dump()
    cached = _cache.get(features_dict)
    if cached is not None:
        return cached

    start = time.perf_counter()
    result = predict_price(
        model=model,
        district=request.district,
        area_sqm=request.area_sqm,
        floor=request.floor,
        lease_commence_year=request.lease_commence_year,
        years_from_launch=request.years_from_launch,
        sale_type=request.sale_type,
        market_segment=request.market_segment,
        project=request.project,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    # SHAP explanation
    explanation: FeatureContribution | None = None
    if _explainer is not None:
        try:
            feature_vec = build_serving_features(
                district=request.district, area_sqm=request.area_sqm,
                floor=request.floor, lease_commence_year=request.lease_commence_year,
                years_from_launch=request.years_from_launch,
                sale_type=request.sale_type, market_segment=request.market_segment,
                project=request.project,
            )
            shap_result = _explainer.explain(feature_vec)
            if "error" not in shap_result:
                explanation = FeatureContribution(**shap_result)
        except Exception:
            logger.warning("shap_explanation_failed", exc_info=True)

    response = PredictionResponse(
        predicted_price=result["predicted_price"],
        prediction_interval=PredictionInterval(
            lower_bound=result["lower_bound"],
            upper_bound=result["upper_bound"],
        ),
        input_features=request,
        explanation=explanation,
    )

    log_prediction(db, _model_version, features_dict, result["predicted_price"], latency_ms)
    _cache.set(features_dict, response)
    logger.info("prediction_served", latency_ms=round(latency_ms, 2))
    return response


@app.post("/predict/milestones", response_model=MilestonePredictionResponse)
async def predict_milestones_endpoint(
    request: MilestonePredictionRequest,
    req: Request = None,
    db: Session = Depends(get_db_dependency),
) -> MilestonePredictionResponse:
    _rate_limiter.check(req)
    model = _ensure_model()

    features_dict = request.model_dump()
    cached = _cache.get(features_dict)
    if cached is not None:
        return cached

    start = time.perf_counter()
    result = predict_at_milestones(
        model=model,
        district=request.district,
        area_sqm=request.area_sqm,
        floor=request.floor,
        lease_commence_year=request.lease_commence_year,
        market_segment=request.market_segment,
        project=request.project,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    response = MilestonePredictionResponse(
        mop_5yr_price=result["mop_5yr_price"],
        mop_5yr_interval=PredictionInterval(
            lower_bound=result["mop_5yr_lower"],
            upper_bound=result["mop_5yr_upper"],
        ),
        privatised_10yr_price=result["privatised_10yr_price"],
        privatised_10yr_interval=PredictionInterval(
            lower_bound=result["privatised_10yr_lower"],
            upper_bound=result["privatised_10yr_upper"],
        ),
        price_appreciation=result["price_appreciation"],
        appreciation_pct=result["appreciation_pct"],
    )

    log_prediction(db, _model_version, features_dict, result["mop_5yr_price"], latency_ms)
    _cache.set(features_dict, response)
    logger.info("milestone_prediction_served", latency_ms=round(latency_ms, 2))
    return response
