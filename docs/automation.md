# Automation Strategy

## 1. Automated Feature Engineering

### Current State
Feature engineering is implemented as deterministic transformations in `src/features/engineering.py`. These are manually defined based on domain knowledge of the EC market.

### Proposed Automation

**Feature Store with Versioning**
- Migrate feature definitions to a feature store (e.g., Feast or a lightweight Postgres-backed store)
- Each feature pipeline version is tagged and immutable
- Training and serving always reference the same feature version, eliminating train-serve skew

**Automated Feature Discovery**
- Schedule weekly jobs to compute candidate features from newly ingested data:
  - Rolling price averages by district (3-month, 6-month, 12-month windows)
  - Macro indicators (MAS interest rates, CPI) joined from external APIs
  - Spatial features derived from SVY21 coordinates (distance to MRT, schools, CBD)
- Use permutation importance or SHAP to automatically rank and select top features
- Promote features that improve CV R² by > 1% to the production feature set

**Pipeline Orchestration**
```
Airflow DAG (Weekly):
  1. Ingest new URA transactions
  2. Run feature pipeline (versioned)
  3. Store features in feature store
  4. Trigger model retraining if data volume threshold met
```

## 2. Automated Model Selection

### Current State
XGBoost is the default algorithm, manually configured with fixed hyperparameters.

### Proposed Automation

**Hyperparameter Optimisation**
- Integrate Optuna for Bayesian hyperparameter search
- Search space: learning rate, max depth, n_estimators, subsample, regularisation terms
- Objective: minimise cross-validated MAPE (most interpretable for price prediction)
- Budget: 100 trials per retraining run

**Multi-Algorithm Comparison**
- On each retraining, automatically train and evaluate:
  1. XGBoost
  2. LightGBM
  3. CatBoost
  4. Elastic Net (baseline)
  5. Stacking ensemble of top-3
- Select the model with best CV MAPE, subject to latency constraints (< 50ms p95)

**Champion-Challenger Framework**
```
On retrain:
  1. Train new candidate model (challenger)
  2. Evaluate on holdout set
  3. If challenger MAPE < champion MAPE - 0.5%:
     a. Deploy challenger as shadow model (receives traffic, predictions logged but not served)
     b. After 7 days of shadow evaluation, promote to champion if metrics hold
  4. Else: keep champion, log results for review
```

**Model Registry**
- Every trained model is registered in `model_registry` table with:
  - Algorithm, hyperparameters, training data version
  - All evaluation metrics
  - Active/inactive flag
- Supports instant rollback to any previous model version

## 3. Automated Monitoring

### Continuous Monitoring Pipeline
```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ Prediction   │────▶│  Drift        │────▶│  Alert       │
│ Logs         │     │  Detector     │     │  Manager     │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                     ┌──────────────┐            │
                     │  Auto         │◀───────────┘
                     │  Retrain      │
                     │  Trigger      │
                     └──────────────┘
```

### Automated Drift Response
| Drift Level | PSI Range | Automated Action |
|-------------|-----------|-----------------|
| None | < 0.1 | No action |
| Minor | 0.1 - 0.2 | Log warning, increase monitoring frequency |
| Moderate | 0.2 - 0.3 | Auto-trigger retraining pipeline |
| Severe | > 0.3 | Alert on-call, pause predictions, emergency retrain |

### Ground Truth Reconciliation
- Automated quarterly job after URA publishes new actuals:
  1. Match predictions to actual transaction prices
  2. Compute residuals by segment
  3. Update monitoring dashboard
  4. If MAE degradation > 20%, trigger retrain

### Self-Healing
- If API health check fails 3 consecutive times → auto-restart container
- If model load fails → fall back to previous model version from registry
- If database connection drops → exponential backoff with circuit breaker
