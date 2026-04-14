"""Inference module for EC price prediction.

Supports the appreciation model from notebooks/03_appreciation_model.py:
  target = appreciation_ratio = resale_psm / launch_psm
  predicted_price = predicted_ratio × launch_psm × area
"""

from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from src.features.serving import build_serving_features

logger = structlog.get_logger()


def _get_launch_psm(model_bundle: dict[str, Any], project: str | None, district: int) -> float:
    """Get launch PSM for a project from the model's lookup table."""
    lookup = model_bundle.get("launch_psm_lookup", {})
    if project and project in lookup:
        return lookup[project]
    # Fallback: use district EC median as proxy
    from src.features.serving import get_lookups
    lookups = get_lookups()
    ec_dist = lookups.get("ec_district_stats", {})
    if float(district) in ec_dist:
        return ec_dist[float(district)]
    return lookups.get("global_defaults", {}).get("ec_dist_med_psm", 14000.0)


def predict_price(
    model: Any,
    district: int,
    area_sqm: float,
    floor: int,
    lease_commence_year: int,
    years_from_launch: int,
    sale_type: str = "Resale",
    market_segment: str = "OCR",
    project: str | None = None,
) -> dict[str, float]:
    """Predict the price of an EC unit with prediction intervals."""
    features = build_serving_features(
        district=district, area_sqm=area_sqm, floor=floor,
        lease_commence_year=lease_commence_year,
        years_from_launch=years_from_launch,
        sale_type=sale_type, market_segment=market_segment,
        project=project,
    )

    if isinstance(model, dict) and model.get("model_type") == "appreciation":
        # Appreciation model: predict ratio, convert to price
        launch_psm = _get_launch_psm(model, project, district)
        inner = model["model"]

        if model.get("model_needs_scaling", False):
            pred_input = model["scaler"].transform(features)
        else:
            pred_input = features

        ratio = float(inner.predict(pred_input)[0])
        point = round(ratio * launch_psm * area_sqm, 2)

        # Quantile intervals
        lower_ratio = float(model["quantile_lower"].predict(features)[0])
        upper_ratio = float(model["quantile_upper"].predict(features)[0])
        lower = round(lower_ratio * launch_psm * area_sqm, 2)
        upper = round(upper_ratio * launch_psm * area_sqm, 2)

    elif isinstance(model, dict):
        # Legacy log-price or raw-price model
        is_log = model.get("log_target", False)
        scaled = model["scaler"].transform(features)
        raw = float(model["model"].predict(scaled)[0])
        point = round(float(np.expm1(raw)) if is_log else raw, 2)

        if "quantile_lower" in model:
            lr = float(model["quantile_lower"].predict(features)[0])
            ur = float(model["quantile_upper"].predict(features)[0])
            lower = round(float(np.expm1(lr)) if is_log else lr, 2)
            upper = round(float(np.expm1(ur)) if is_log else ur, 2)
        else:
            lower, upper = point, point
    else:
        point = round(float(model.predict(features)[0]), 2)
        lower, upper = point, point

    logger.info(
        "prediction_made",
        district=district, area_sqm=area_sqm, project=project,
        predicted_price=point, interval=f"[{lower}, {upper}]",
    )
    return {"predicted_price": point, "lower_bound": lower, "upper_bound": upper}


def predict_at_milestones(
    model: Any,
    district: int,
    area_sqm: float,
    floor: int,
    lease_commence_year: int,
    market_segment: str = "OCR",
    project: str | None = None,
) -> dict[str, Any]:
    """Predict EC prices at both 5-year (MOP) and 10-year (privatisation) marks."""
    r5 = predict_price(
        model, district, area_sqm, floor, lease_commence_year,
        years_from_launch=5, sale_type="Resale", market_segment=market_segment,
        project=project,
    )
    r10 = predict_price(
        model, district, area_sqm, floor, lease_commence_year,
        years_from_launch=10, sale_type="Resale", market_segment=market_segment,
        project=project,
    )

    p5, p10 = r5["predicted_price"], r10["predicted_price"]
    return {
        "mop_5yr_price": p5,
        "mop_5yr_lower": r5["lower_bound"],
        "mop_5yr_upper": r5["upper_bound"],
        "privatised_10yr_price": p10,
        "privatised_10yr_lower": r10["lower_bound"],
        "privatised_10yr_upper": r10["upper_bound"],
        "price_appreciation": round(p10 - p5, 2),
        "appreciation_pct": round((p10 - p5) / p5 * 100, 2) if p5 > 0 else None,
    }
