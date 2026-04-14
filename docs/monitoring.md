# Model Monitoring Strategy

## 1. Metrics to Track

### Performance Metrics (Real-time)
| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| **Prediction Latency (p50/p95/p99)** | Time from request to response | p95 > 200ms |
| **Request throughput** | Requests per second | Sustained > 80% capacity |
| **Error rate** | 4xx + 5xx / total requests | > 1% |
| **Model load time** | Time to deserialise model on startup | > 10s |

### Data Quality Metrics (Batch)
| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| **Feature drift (PSI)** | Population Stability Index per feature between training and serving distributions | PSI > 0.2 |
| **Input range violations** | % of requests with features outside training bounds | > 5% |
| **Missing value rate** | % of null/missing features in requests | > 2% |
| **Data freshness** | Time since last successful URA ingestion | > 100 days |

### Model Quality Metrics (Periodic)
| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| **Prediction drift** | Distribution shift in predicted prices over time (KS test) | p-value < 0.05 |
| **Residual analysis** | MAE/MAPE computed against actual resale prices as they become available | MAE > 1.5× training MAE |
| **Segment-level performance** | Accuracy broken down by district, market segment | Any segment R² < 0.6 |
| **Prediction confidence** | Variance across ensemble trees | High variance predictions flagged |

## 2. Monitoring Implementation

### Prediction Logging
Every prediction is logged to `prediction_logs` table:
- Input features (JSONB)
- Predicted price
- Model version
- Latency
- Timestamp

This creates a queryable audit trail for retroactive analysis.

### Drift Detection Pipeline
```
Weekly Cron Job:
1. Query last 7 days of prediction_logs
2. Compare feature distributions against training data baseline
3. Compute PSI for each feature
4. If PSI > 0.2 for any feature → trigger alert
5. Log results to monitoring dashboard
```

### Ground Truth Feedback Loop
- URA publishes actual transaction prices quarterly
- Automated pipeline compares predictions made 3-6 months ago against actual prices
- Computes rolling MAE/MAPE to detect model staleness

## 3. Suggested Metrics for Incorporation

### Business Metrics
- **Price prediction accuracy by EC project** — some developments may be systematically over/undervalued
- **Prediction volume by district** — understand usage patterns, identify underserved segments
- **User correction rate** — if a UI allows feedback, track how often users disagree with predictions

### Operational Metrics
- **Database connection pool utilisation** — early warning for scaling needs
- **Container memory/CPU usage** — capacity planning
- **API availability (uptime SLA)** — target 99.9%

## 4. Alerting Strategy

| Severity | Condition | Response |
|----------|-----------|----------|
| **P0 (Critical)** | Model serving errors > 5%, API down | Page on-call, rollback to previous model |
| **P1 (High)** | Significant drift detected (PSI > 0.25) | Retrain within 48 hours |
| **P2 (Medium)** | MAE degradation > 20% vs baseline | Schedule retrain in next sprint |
| **P3 (Low)** | Minor feature drift, data freshness warning | Monitor, include in weekly review |
