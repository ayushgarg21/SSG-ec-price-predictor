# Known Limitations and Mitigations

## Data Limitations

### 1. Limited MOP (5-year) Resale Samples
The test set contains very few transactions at exactly the 5-year mark — the precise prediction target HDB cares most about. This is because most ECs in the dataset are either very new (< 3 years, still in developer sales phase) or mature (> 10 years).

**Impact**: Higher prediction error for the 3-5 year range (MAPE ~6-7%) compared to other segments (~2-4%).

**Mitigation**: The model captures the broader relationship between years-from-launch and price through the `is_post_mop` binary feature and `lease_age_bucket` ordinal feature. As more ECs reach MOP in 2026-2027 (the current wave of 2020-2021 launches), model accuracy at this critical point will improve with retraining.

### 2. Only 5 Years of Transaction History
URA provides only the last 5 years of transaction data. This means:
- The model has never seen a full property cycle (boom-bust-recovery)
- All training data comes from a broadly appreciating market (2021-2026)
- The model may underestimate downside risk

**Mitigation**: The prediction intervals (80% coverage) partially address this by providing a range rather than a point estimate. For production use, we recommend combining model predictions with macroeconomic indicators (interest rates, cooling measures) that aren't captured in the current feature set.

### 3. No Private Condo Comparison Features
After privatisation (10 years), ECs compete directly with private condominiums. The model doesn't include private condo prices as features because that would require a separate data pipeline.

**Mitigation**: The `district_lag_median_psm` feature captures neighbourhood-level pricing trends that implicitly include private condo market conditions. A future iteration could explicitly incorporate private condo PSM as a benchmark feature.

## Model Limitations

### 4. District 28 (Seletar/Yio Chu Kang) Under-Represented
District 28 has only 14 test samples and the highest MAPE (~9%). This is because very few ECs are located in this district.

**Mitigation**: The model falls back to district-level lag features, which helps. For districts with < 20 transactions, predictions should be treated with wider confidence bands. The API returns prediction intervals that naturally widen for less-represented segments.

### 5. No Spatial Features Beyond District
The model uses district number as a proxy for location, but doesn't use the SVY21 coordinates (x, y) available in the data. Two ECs in the same district can have very different valuations based on proximity to MRT stations, schools, or amenities.

**Mitigation**: District-level features capture the macro location effect. A future iteration could engineer features like distance-to-nearest-MRT using the SVY21 coordinates and publicly available MRT station locations.

### 6. Project-Level Features May Not Exist for New Launches
For a brand-new EC project not yet in the training data, the project-level lag features (`project_lag_median_psm`, etc.) fall back to global medians. This reduces prediction accuracy for new launches.

**Mitigation**: The model still has district-level features and base features (area, floor, lease year) which drive the majority of prediction power. Users should also provide the `project` field in the API request when available to get project-specific adjustments.

## Infrastructure Limitations

### 7. URA API Bot Protection
The URA API blocks standard Python HTTP clients (httpx, requests) with a JavaScript challenge. The ingestion pipeline uses `curl` subprocess as a workaround.

**Impact**: The ingestion pipeline requires `curl` to be installed on the host system.

**Mitigation**: For a production deployment, consider:
- Running ingestion from a whitelisted IP
- Using Selenium/Playwright for automated browser-based fetching
- Caching URA data in a data lake to decouple ingestion from model training

### 8. In-Memory Caching and Rate Limiting
The API uses in-memory data structures for caching and rate limiting. These are not shared across workers/instances and are lost on restart.

**Mitigation**: For production scale, replace with Redis-backed caching and rate limiting. The current implementation is suitable for single-instance deployment as specified in the case study.

## What Would Improve the Model Most

In priority order:
1. **More historical data** (10+ years) to capture a full property cycle
2. **Spatial features** (MRT distance, school proximity) using SVY21 coordinates
3. **Macroeconomic features** (SIBOR/SORA rates, property cooling measure dates, CPI)
4. **Private condo benchmark** (district-level private condo PSM as a comparison feature)
5. **More MOP-era transactions** as the 2020-2021 EC wave reaches 5 years
