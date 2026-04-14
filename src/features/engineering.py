"""Feature engineering for EC price prediction."""

import pandas as pd
import numpy as np
import re
import structlog

logger = structlog.get_logger()


def extract_floor_mid(floor_range: str | None) -> int | None:
    """Convert '06-10' floor range to midpoint (8)."""
    if not floor_range or not isinstance(floor_range, str):
        return None
    match = re.match(r"(\d+)\s*-\s*(\d+)", floor_range)
    if match:
        return (int(match.group(1)) + int(match.group(2))) // 2
    return None


def extract_lease_commence_year(tenure: str | None) -> int | None:
    """Extract lease commencement year from tenure string.
    e.g. '99 yrs lease commencing from 2014' -> 2014
    """
    if not tenure or not isinstance(tenure, str):
        return None
    match = re.search(r"commencing from (\d{4})", tenure)
    if match:
        return int(match.group(1))
    return None


def parse_contract_date(contract_date: str | None) -> tuple[int | None, int | None]:
    """Parse MMYY format to (month, year). e.g. '0325' -> (3, 2025)."""
    if not contract_date or len(contract_date) != 4:
        return None, None
    try:
        month = int(contract_date[:2])
        year_suffix = int(contract_date[2:])
        year = 2000 + year_suffix if year_suffix < 80 else 1900 + year_suffix
        return month, year
    except ValueError:
        return None, None


def compute_years_from_launch(lease_year: int | None, txn_year: int | None) -> int | None:
    """Compute how many years after lease commencement the transaction occurred."""
    if lease_year is None or txn_year is None:
        return None
    return txn_year - lease_year


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw transactions into model-ready features.

    Filters for Executive Condominiums and engineers all features needed
    for price prediction at 5-year (MOP) and 10-year (privatisation) marks.
    """
    # Filter for EC only
    ec_df = df[df["property_type"].str.contains("Executive Condominium", case=False, na=False)].copy()
    logger.info("ec_transactions_filtered", count=len(ec_df))

    if ec_df.empty:
        return pd.DataFrame()

    # Parse contract date
    parsed = ec_df["contract_date"].apply(lambda x: pd.Series(parse_contract_date(x)))
    ec_df["txn_month"] = parsed[0]
    ec_df["txn_year"] = parsed[1]

    # Extract features
    ec_df["floor_mid"] = ec_df["floor_range"].apply(extract_floor_mid)
    ec_df["lease_commence_year"] = ec_df["tenure"].apply(extract_lease_commence_year)
    ec_df["years_from_launch"] = ec_df.apply(
        lambda r: compute_years_from_launch(r["lease_commence_year"], r["txn_year"]), axis=1
    )
    ec_df["price_psm"] = ec_df["price"] / ec_df["area"]

    # District as numeric
    ec_df["district_num"] = pd.to_numeric(ec_df["district"], errors="coerce")

    # Remaining lease years (99-year lease assumed for ECs)
    ec_df["remaining_lease"] = 99 - ec_df["years_from_launch"]

    # Sale type encoding (URA API uses "1"=New Sale, "2"=Sub Sale, "3"=Resale)
    sale_type_map = {"1": 0, "New Sale": 0, "2": 1, "Sub Sale": 1, "3": 2, "Resale": 2}
    ec_df["sale_type_encoded"] = ec_df["type_of_sale"].map(sale_type_map).fillna(-1).astype(int)

    # Market segment encoding
    segment_map = {"OCR": 0, "RCR": 1, "CCR": 2}
    ec_df["segment_encoded"] = ec_df["market_segment"].map(segment_map).fillna(-1).astype(int)

    # Select final features
    feature_cols = [
        "project", "district_num", "area", "floor_mid", "lease_commence_year",
        "years_from_launch", "remaining_lease", "sale_type_encoded",
        "segment_encoded", "txn_year", "txn_month", "price_psm", "price",
    ]

    result = ec_df[feature_cols].dropna(subset=["price", "area", "floor_mid", "years_from_launch"])
    logger.info("features_built", rows=len(result))
    return result


def create_prediction_features(
    district: int,
    area_sqm: float,
    floor: int,
    lease_commence_year: int,
    years_from_launch: int,
    sale_type: str = "Resale",
    market_segment: str = "OCR",
) -> np.ndarray:
    """Build a single feature vector for inference."""
    sale_type_map = {"New Sale": 0, "Sub Sale": 1, "Resale": 2}
    segment_map = {"OCR": 0, "RCR": 1, "CCR": 2}

    remaining_lease = 99 - years_from_launch

    return np.array([[
        district,
        area_sqm,
        floor,
        lease_commence_year,
        years_from_launch,
        remaining_lease,
        sale_type_map.get(sale_type, 2),
        segment_map.get(market_segment, 0),
    ]])
