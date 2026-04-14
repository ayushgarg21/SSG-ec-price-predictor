"""Feature construction for serving (inference time).

Builds features matching the appreciation model (notebooks/03_appreciation_model.py).
Uses pre-computed lookup tables — no data leakage.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import joblib
import structlog

logger = structlog.get_logger()

_lookups: dict[str, Any] | None = None


def load_lookups(path: str = "./artifacts/serving_lookups.joblib") -> dict[str, Any]:
    global _lookups
    _lookups = joblib.load(path)
    logger.info("serving_lookups_loaded", latest_year=_lookups.get("latest_year"))
    return _lookups


def get_lookups() -> dict[str, Any]:
    if _lookups is None:
        raise RuntimeError("Serving lookups not loaded. Call load_lookups() first.")
    return _lookups


def build_serving_features(
    district: int,
    area_sqm: float,
    floor: int,
    lease_commence_year: int,
    years_from_launch: int,
    sale_type: str = "Resale",
    market_segment: str = "OCR",
    project: str | None = None,
) -> np.ndarray:
    """Build full feature vector for inference.

    Matches the FEATURE_COLUMNS order from 03_appreciation_model.py:
      district_num, area, floor_mid, lease_commence_year,
      years_from_launch, remaining_lease, segment_encoded,
      launch_psm, launch_vs_district,
      all_dist_med_psm, all_dist_vol, all_dist_std,
      ec_dist_med_psm, district_momentum, market_lag_psm,
      proj_lag_psm, proj_lag_vol, project_target_enc,
      area_quartile, is_post_mop, is_privatised,
      quarter, lease_age_bucket, is_high_floor
    """
    lookups = get_lookups()
    defaults = lookups["global_defaults"]
    area_bins = lookups.get("area_quartile_bins", [30, 80, 95, 110, 300])

    segment_map = {"OCR": 0, "RCR": 1, "CCR": 2}
    remaining_lease = 99 - years_from_launch

    # Launch PSM
    launch_lookup = lookups.get("launch_psm_lookup", {})
    if project and project in launch_lookup:
        launch_psm = launch_lookup[project]
    else:
        ec_dist = lookups.get("ec_district_stats", {})
        launch_psm = ec_dist.get(float(district), defaults.get("ec_dist_med_psm", 14000.0))

    # District stats (full market)
    d_stats = lookups.get("district_stats", {}).get(float(district), {})
    all_dist_med_psm = d_stats.get("median_psm", defaults["all_dist_med_psm"])
    all_dist_vol = d_stats.get("volume", defaults["all_dist_vol"])
    all_dist_std = d_stats.get("std_psm", defaults["all_dist_std"])

    # EC district
    ec_dist_med_psm = lookups.get("ec_district_stats", {}).get(
        float(district), defaults["ec_dist_med_psm"]
    )

    # Launch vs district
    launch_vs_district = launch_psm / all_dist_med_psm if all_dist_med_psm > 0 else 1.0

    # Momentum, market
    district_momentum = lookups.get("district_momentum", {}).get(
        float(district), defaults["district_momentum"]
    )
    market_lag_psm = lookups.get("market_lag_psm", defaults["market_lag_psm"])

    # Project lag
    p_stats = lookups.get("project_stats", {}).get(project, {}) if project else {}
    proj_lag_psm = p_stats.get("median_psm", defaults["proj_lag_psm"])
    proj_lag_vol = p_stats.get("volume", defaults["proj_lag_vol"])

    # Target-encoded project
    project_target_enc = lookups.get("project_target_enc", {}).get(
        project, lookups.get("global_mean_ratio", 1.26)
    ) if project else lookups.get("global_mean_ratio", 1.26)

    # Derived
    area_quartile = min(int(np.digitize(area_sqm, area_bins[1:-1])), 3)
    is_post_mop = int(years_from_launch >= 5)
    is_privatised = int(years_from_launch >= 10)
    quarter = 1
    lease_age_bucket = (
        0 if years_from_launch <= 2
        else 1 if years_from_launch <= 5
        else 2 if years_from_launch <= 10
        else 3 if years_from_launch <= 15
        else 4
    )
    is_high_floor = int(floor >= 15)

    return np.array([[
        district, area_sqm, floor, lease_commence_year,
        years_from_launch, remaining_lease, segment_map.get(market_segment, 0),
        launch_psm, launch_vs_district,
        all_dist_med_psm, all_dist_vol, all_dist_std,
        ec_dist_med_psm, district_momentum, market_lag_psm,
        proj_lag_psm, proj_lag_vol, project_target_enc,
        area_quartile, is_post_mop, is_privatised,
        quarter, lease_age_bucket, is_high_floor,
    ]])
