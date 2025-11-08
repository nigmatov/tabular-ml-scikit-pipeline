# tabular-ml-scikit-pipeline
End-to-end tabular ML with scikit-learn, Optuna tuning, SHAP, and FastAPI inference.

## Features

- Pipeline with numeric/ordinal/categorical branches.
- Stratified K-fold CV. Metrics: ROC-AUC, F1, calibration curve.
- Optuna trials (e.g., 50) with study artifact.
- SHAP summary and force plots saved to reports/.
- FastAPI server with input validation and example payloads.
- Unit tests for pipeline fit/predict and API contract.


## Quickstart
```bash
make setup
make train
make serve
```

## Example use case — Credit Risk Prediction

This notebook (`notebooks/01_credit_risk_walkthrough.ipynb`) demonstrates
how to train, evaluate, and explain a credit-risk model using scikit-learn and SHAP.

**Key outputs**
| File | Description |
|------|--------------|
| `reports/roc_curve.png` | ROC curve showing model discrimination |
| `reports/shap_summary.png` | SHAP feature importance |
| `models/credit_risk.joblib` | Serialized end-to-end pipeline |
