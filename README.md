# TrustVest AI

## Phase 1 — Synthetic Population Generator

### Purpose

Creates a reproducible, correlated synthetic population for educational and hackathon experimentation with explainable alternative-credit models. It is not suitable for real lending, eligibility, or other high-impact decisions.

### Generated Features

The dataset includes demographics, employment, income and expenses, UPI and wallet behaviour, bill-payment discipline, savings and loan signals, device trust, five deterministic engineered indices, and a deterministic `credit_likelihood` target (0–100).

### Dataset Size

The default configuration produces 100,000 Indian user records. Change `NUM_USERS` in [`ml/config.py`](ml/config.py) or pass `--num-users` at runtime.

### Output Files

- `ml/datasets/synthetic_users.csv`
- `ml/plots/income_distribution.png`
- `ml/plots/credit_distribution.png`
- `ml/plots/correlation_heatmap.png`
- `ml/plots/age_distribution.png`

### Usage Instructions

Install the dependencies, then run the generator from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m ml.generators.synthetic_generator
```

For a smaller development dataset with a fixed seed:

```powershell
python -m ml.generators.synthetic_generator --num-users 5000 --seed 2026
```

Running with the same seed and size produces the same dataset values.

## Phase 2 — Feature Engineering Pipeline

### Purpose

Builds a clean, reproducible, ML-ready dataset for explainable credit-likelihood regression. The pipeline only reads `ml/datasets/synthetic_users.csv`; it never modifies the source data.

### Engineered Features

The pipeline adds financial health, digital payment, credit behaviour, employment stability, financial cushion, device reliability, digital engagement, banking maturity, risk, debt burden, and seven transparent interaction features. Each score is derived from source fields only and does not use `credit_likelihood`.

### Scaling, Encoding, and Selection

Configured numeric and engineered features are scaled with `StandardScaler`. Occupation, employment type, education, state, gender, marital status, city tier, and wallet usage are one-hot encoded with `OneHotEncoder`. The pipeline removes constant, duplicate, near-zero variance, and highly correlated features (default absolute correlation threshold: 0.95).

### Usage Instructions

Install the dependencies and run from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m ml.preprocessing.preprocessing
```

### Artifacts Generated

- `ml/datasets/processed_credit_dataset.csv` — target-last numeric training dataset
- `ml/models/scaler.pkl` and `ml/models/encoder.pkl` — fitted preprocessing artifacts
- `ml/reports/validation_report.json` and `ml/reports/data_cleaning_corrections.csv` — validation and repair audit trail
- `ml/reports/feature_selection_report.json`, `feature_importance.csv`, and `correlation_report.md`
- `ml/plots/feature_importance.png`, `correlation_heatmap_processed.png`, `target_distribution_processed.png`, and `feature_distribution_summary.png`

The structured run log is written to `ml/reports/preprocessing_pipeline.jsonl`.
