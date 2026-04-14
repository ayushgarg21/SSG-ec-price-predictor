# System Architecture

## Overview

This solution implements an end-to-end ML pipeline for predicting Executive Condominium (EC) resale prices at two critical lifecycle milestones:

1. **5 years post-lease** — Minimum Occupancy Period (MOP), when units first become eligible for resale
2. **10 years post-lease** — When ECs become privatised and are indistinguishable from private condos

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                         │
│                                                                      │
│   ┌──────────────┐     ┌──────────────────┐     ┌───────────────┐   │
│   │  URA API      │────▶│  Ingestion        │────▶│  PostgreSQL   │   │
│   │  (Batch 1-4)  │     │  Pipeline         │     │  (Raw Store)  │   │
│   └──────────────┘     └──────────────────┘     └───────┬───────┘   │
│                                                          │           │
└──────────────────────────────────────────────────────────┼───────────┘
                                                           │
┌──────────────────────────────────────────────────────────┼───────────┐
│                     FEATURE ENGINEERING LAYER             │           │
│                                                          ▼           │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │  Feature Pipeline                                            │   │
│   │  • EC filtering  • Floor midpoint  • Lease year extraction   │   │
│   │  • Years-from-launch  • Price PSM  • Segment encoding       │   │
│   └──────────────────────────────────┬───────────────────────────┘   │
│                                      │                               │
└──────────────────────────────────────┼───────────────────────────────┘
                                       │
┌──────────────────────────────────────┼───────────────────────────────┐
│                        MODEL LAYER   │                               │
│                                      ▼                               │
│   ┌────────────────┐    ┌────────────────┐    ┌──────────────────┐  │
│   │  Training       │───▶│  XGBoost /      │───▶│  Model Registry  │  │
│   │  Pipeline       │    │  GBR Pipeline   │    │  (artifacts/)    │  │
│   └────────────────┘    └────────────────┘    └──────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                        SERVING LAYER                                 │
│                                                                      │
│   ┌──────────────┐     ┌──────────────────┐     ┌───────────────┐   │
│   │  FastAPI       │────▶│  /predict         │────▶│  JSON         │   │
│   │  (Docker)      │    │  /predict/mile..  │     │  Response     │   │
│   │                │    │  /health          │     │               │   │
│   └──────────────┘     └──────────────────┘     └───────────────┘   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Ingestion (`src/ingestion/`)
- **URA API Client**: Authenticates with AccessKey + daily Token, fetches all 4 batches of private residential transactions
- **Flattener**: Normalises the nested JSON response (project → transactions) into flat rows
- **Loader**: Bulk inserts into PostgreSQL via SQLAlchemy

### 2. Feature Engineering (`src/features/`)
- Filters for Executive Condominiums only (`property_type`)
- Parses `contract_date` (MMYY format) into month/year
- Extracts `lease_commence_year` from tenure string via regex
- Computes `years_from_launch` (transaction year − lease commencement year)
- Derives `remaining_lease`, `price_psm`, encoded categoricals

### 3. Model Training (`src/model/`)
- **Algorithm**: XGBoost Regressor (default) or Gradient Boosting Regressor
- **Pipeline**: StandardScaler → Regressor (scikit-learn Pipeline for consistent preprocessing)
- **Evaluation**: MAE, RMSE, R², MAPE, 5-fold cross-validation
- **Artifacts**: Serialised pipeline (joblib) + metadata JSON

### 4. Serving Layer (`src/api/`)
- **Framework**: FastAPI with async endpoints
- **Endpoints**:
  - `POST /predict` — single prediction for any years_from_launch
  - `POST /predict/milestones` — compares 5yr vs 10yr prices
  - `GET /health` — liveness + model status
- **Validation**: Pydantic v2 schemas with field constraints

### 5. Infrastructure
- **Docker Compose**: PostgreSQL 16 + API service
- **Database**: PostgreSQL with UUID primary keys, proper indexes, prediction logging table
- **Configuration**: Pydantic Settings with `.env` support

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| XGBoost over deep learning | Tabular data with < 100k rows; tree-based models dominate this regime. Lower latency, more interpretable. |
| scikit-learn Pipeline | Ensures preprocessing (scaling) is always applied consistently during training and inference. |
| PostgreSQL over SQLite | Production-grade; supports concurrent access, JSON columns for metadata, proper indexing. |
| FastAPI over Flask | Native async support, automatic OpenAPI docs, Pydantic validation, better performance. |
| Joblib serialisation | Standard for sklearn pipelines; faster than pickle for numpy-heavy objects. |

## Data Flow

```
URA API → Ingest → Postgres → Feature Eng → Train → artifacts/model.joblib
                                                          │
                                              API Server ◄─┘
                                                  │
                                          Client Request
                                                  │
                                          Feature Vector → Model.predict → Response
```
