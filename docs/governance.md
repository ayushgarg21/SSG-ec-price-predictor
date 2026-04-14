# Model Governance Framework

## 1. Access Controls

### Role-Based Access Control (RBAC)

| Role | Permissions | Example Users |
|------|------------|---------------|
| **Admin** | Full access: deploy models, modify infrastructure, manage users | ML Platform team |
| **ML Engineer** | Train models, push to registry, promote to staging | Data science team |
| **Reviewer** | Approve model promotions to production, view all metrics | Senior DS / Tech Lead |
| **Consumer** | Call prediction API, view documentation | Application developers |
| **Auditor** | Read-only access to all logs, registry, and metrics | Compliance / risk team |

### Infrastructure Access
- Database credentials stored in HashiCorp Vault or AWS Secrets Manager (never in source code)
- API keys (URA, cloud services) rotated quarterly
- Service-to-service authentication via mTLS or JWT tokens
- All Postgres access via connection pooler (PgBouncer) with per-role connection limits

## 2. Version Control and Reproducibility

### Code Versioning
- All model code, feature pipelines, and configs in Git
- Every training run tagged with a Git commit SHA
- Infrastructure-as-Code (Docker Compose / Terraform) versioned alongside application code

### Data Versioning
- Raw ingestion snapshots stored with timestamp prefixes
- Feature datasets tagged with pipeline version + data date range
- Training/test splits stored deterministically (fixed random seed = 42)

### Model Versioning
- Every trained model registered in `model_registry` with:
  - Unique version identifier (semver: `1.0.0`, `1.1.0`)
  - Link to training data version
  - Link to Git commit
  - Full hyperparameter set
  - All evaluation metrics
  - Training environment details (Python version, library versions)

### Reproducibility Checklist
```
For any model version, you must be able to:
☐ Check out the exact code commit
☐ Access the exact training data snapshot
☐ Re-run training and get equivalent metrics (±1% tolerance)
☐ Explain every feature and its derivation
☐ Trace any prediction back to input features and model version
```

## 3. Model Validation and Testing

### Pre-Deployment Validation
Before any model is promoted to production:

1. **Statistical Tests**
   - Performance on holdout set must meet minimum thresholds:
     - R² ≥ 0.70
     - MAPE ≤ 15%
     - MAE ≤ $80,000
   - No statistically significant performance degradation vs. current champion (paired t-test on residuals)

2. **Bias and Fairness Audit**
   - Performance parity across districts (no district with MAE > 2× overall MAE)
   - No systematic over/under-prediction for specific market segments
   - Analysis documented in model card

3. **Robustness Testing**
   - Sensitivity analysis: how much does prediction change for ±10% input perturbation?
   - Edge case evaluation: minimum/maximum floor, smallest/largest units, newest/oldest leases
   - Adversarial inputs: out-of-distribution feature values handled gracefully

4. **Integration Testing**
   - API endpoint returns valid responses for all schemas
   - Latency within SLA (< 200ms p95)
   - Model loads correctly in containerised environment

### Approval Workflow
```
ML Engineer trains model
       │
       ▼
Automated validation suite runs
       │
       ▼ (all pass)
Reviewer receives model card + metrics comparison
       │
       ▼ (approved)
Shadow deployment (7 days)
       │
       ▼ (metrics hold)
Production promotion
```

## 4. Audit Trail and Compliance

### Prediction Logging
Every prediction is immutably logged with:
- Timestamp
- Model version
- Full input features
- Predicted output
- Inference latency
- Request source identifier

### Model Cards
Each production model has an accompanying model card documenting:
- Intended use and limitations
- Training data description and date range
- Performance metrics with confidence intervals
- Known biases or failure modes
- Maintenance schedule

### Change Log
All model promotions, demotions, and configuration changes recorded with:
- Who made the change
- What changed
- Why (linked to ticket/justification)
- Rollback procedure

## 5. Incident Response

| Scenario | Detection | Response | Recovery |
|----------|-----------|----------|----------|
| Model serving incorrect predictions | Monitoring alert (MAE spike) | Page on-call, investigate root cause | Rollback to previous model version |
| Data pipeline failure | Ingestion job fails | Alert data engineering | Re-run ingestion, extend data freshness SLA |
| Model bias discovered | Fairness audit | Pause affected predictions, investigate | Retrain with corrected features/data |
| Security breach | Access audit logs | Rotate all credentials, assess impact | Redeploy with patched configuration |

## 6. Regulatory Considerations

For property price predictions used in financial decision-making:
- **Transparency**: Predictions should be accompanied by confidence intervals, not point estimates alone
- **Explainability**: SHAP values available for each prediction to explain feature contributions
- **Disclaimer**: Predictions are estimates for informational purposes, not financial advice
- **Data retention**: Prediction logs retained for minimum 7 years per MAS guidelines for financial services data
