# Solution Handover Document

## Overview

This solution predicts Executive Condominium (EC) resale prices at two lifecycle milestones:
- **5 years** (MOP) — when units become eligible for resale
- **10 years** (privatisation) — when ECs become private condominiums

## Approach

### Problem Reframing
Rather than predicting absolute price (which varies wildly by unit size, floor, and era), the model predicts the **appreciation ratio** — how much a unit appreciates relative to its launch price:

```
predicted_price = appreciation_ratio × launch_psm × area
```

This removes price-level variance and focuses the model on what drives appreciation: location, age, market conditions.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Appreciation ratio as target | Removes price-level variance, focuses on drivers of value change |
| Resale-only training data | New sales (developer-set prices) follow different dynamics than resale (market-driven) |
| Temporal train/val/test split | Prevents data leakage; evaluates on future data the model hasn't seen |
| 139k full-market district features | EC prices are influenced by the broader property market, not just other ECs |
| Lag-1 year features only | Ensures all features are available at prediction time with no look-ahead bias |
| Weighted ensemble | Combines XGBoost, LightGBM, CatBoost; weights optimised on validation set |

### Model Performance

| Metric | Value |
|--------|-------|
| R² | 0.892 |
| MAPE | 4.91% |
| MAE | S$83,195 |
| Algorithm | LightGBM (or weighted ensemble) |
| Features | 24 |
| Training data | 10,140 resale transactions |
| Test period | 2026 (unseen during training) |

Segmented performance (all districts < 7% MAPE):
- Best: District 22 at 3.40% MAPE
- 11+ year ECs: 4.48% MAPE

### Leakage Prevention

Every feature is computed using **only data available before the prediction date**:
- District/project statistics use prior-year data (lag-1)
- Target encoding uses train-set KFold (val/test encoded from train means)
- NaN fill medians computed from training set only
- Area quartile bins computed from training set only

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  URA API     │───▶│  Ingestion    │───▶│  PostgreSQL   │
│  (curl)      │    │  Pipeline     │    │  (Docker)     │
└─────────────┘    └──────────────┘    └──────┬───────┘
                                              │
                    ┌──────────────┐           │
                    │  Feature Eng  │◀──────────┘
                    │  + Training   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐    ┌──────────────┐
                    │  FastAPI      │───▶│  Prediction   │
                    │  (Docker)     │    │  + Intervals  │
                    └──────┬───────┘    └──────────────┘
                           │
                    ┌──────▼───────┐
                    │  Streamlit    │
                    │  Frontend     │
                    └──────────────┘
```

## How to Run

### Prerequisites
- Docker Desktop
- Python 3.11+
- URA API access key

### Setup
```bash
cp .env.example .env     # Add URA_ACCESS_KEY
make install             # Install dependencies
make db                  # Start Postgres
make ingest              # Fetch 139k URA transactions
python notebooks/01_eda.py              # EDA + figures
python notebooks/02_appreciation_model.py  # Train model
make serve               # Start API at localhost:8000
make streamlit           # Start frontend at localhost:8501
```

### Quick Test
```bash
curl -X POST http://localhost:8000/predict/milestones \
  -H "Content-Type: application/json" \
  -d '{"district": 19, "area_sqm": 95, "floor": 8, "lease_commence_year": 2020, "project": "PIERMONT GRAND"}'
```

## File Structure

```
SSG_Ai/
├── src/                          # Application code
│   ├── api/                      # FastAPI + caching + rate limiting
│   ├── features/                 # Feature engineering + serving
│   ├── ingestion/                # URA API client (curl-based, retry + circuit breaker)
│   ├── model/                    # Training, inference, SHAP, experiment tracking
│   ├── config.py                 # Pydantic Settings
│   └── database.py               # SQLAlchemy + prediction logging
├── notebooks/
│   ├── 01_eda.py                 # Exploratory data analysis (8 figures)
│   ├── 02_appreciation_model.py  # Full training pipeline (16 figures)
│   └── figures/                  # All generated plots
├── artifacts/                    # Trained model + serving lookups
├── tests/                        # Unit + integration tests
├── docs/
│   ├── data_model.pdf            # Database schema diagram
│   ├── architecture.md           # System design
│   ├── monitoring.md             # Proposed monitoring metrics
│   ├── automation.md             # Proposed automation strategy
│   ├── governance.md             # Proposed governance framework
│   └── limitations.md            # Honest limitations analysis
├── frontend/                     # Streamlit + HTML UI
├── scripts/                      # CLI tools (ingest, train)
├── docker-compose.yml            # Postgres + API
├── Dockerfile                    # API container
├── Makefile                      # Single-command workflows
└── pyproject.toml                # Pinned dependencies
```

## Model Evolution (Interview Talking Points)

1. **v1**: Absolute price model with XGBoost → R²=0.94 → discovered data leakage
2. **v2**: Fixed leakage → R² dropped to 0.35 → diagnosed: wrong problem framing
3. **v3**: Tried log-transform + full market features → R²=0.45 → still wrong framing
4. **v4**: Reframed as appreciation model (ratio from launch price) → R²=0.89 → caught target encoding leak → fixed → **R²=0.892 fully clean**

Key lesson: the model went from predicting "what is this unit worth?" to "how much has this unit appreciated since launch?" — a much more tractable question.

## Known Limitations

See [docs/limitations.md](limitations.md) for full analysis. Summary:
- MOP (5yr) predictions have less training data than 10yr+ predictions
- Prediction interval coverage is 39% (below target 80%) — quantile models need more tuning
- New EC projects not in training data fall back to district-level estimates
- URA API requires curl (blocks Python HTTP clients)

## Security Notes

- Database credentials are in `.env` (not committed)
- URA API key is in `.env` (not committed)
- API rate-limited (60 req/min per IP)
- Docker container runs as non-root user
- No sensitive data in the repository
