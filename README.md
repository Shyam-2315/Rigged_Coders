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

## Phase 3 - Explainable Credit Scoring Model

### Purpose

Trains and exports an explainable XGBoost regression model that estimates the
synthetic `credit_likelihood` target. This phase consumes only the processed,
fully numeric dataset produced by Phase 2.

### Algorithms Compared

The hold-out comparison records RMSE, MAE, R2, MAPE, training time, and
prediction time for:

- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- Extra Trees Regressor
- HistGradientBoostingRegressor

XGBoost is then tuned with 5-fold `RandomizedSearchCV` across tree depth,
learning rate, estimators, row/column subsampling, minimum child weight, and
gamma. It is the selected deployment model because it provides regularised
nonlinear performance and exact tree-contribution explanations suitable for
the requested inference API.

### Training

Install the dependencies, then train from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m ml.training.train_credit_model
```

For a fast smoke run that retains the 5-fold workflow while using smaller
comparison and SHAP settings:

```powershell
python -m ml.training.train_credit_model --quick
```

All runtime settings, including paths, random seed, split, CV folds, tuning
iterations, output names, and explainability sample size are centralised in
`TrainingConfig` in [`ml/config.py`](ml/config.py).

### Explainability and Inference

TreeSHAP generates a global summary plot, global bar plot, dependence plots,
a local waterfall plot, optional force-plot HTML, a ranked feature CSV, and a
top-20 `feature_ranking.json`. The exported inference module exposes
`load_model()`, `predict()`, and `predict_with_explanation()`.

`predict_with_explanation()` accepts exactly one Phase 2 processed feature
record and returns a credit score, a hold-out/feature-coverage confidence
proxy, and signed top positive and negative TreeSHAP feature effects. It does
not accept raw user records; a backend must apply the saved Phase 2
preprocessing artifacts first.

### Generated Model Outputs

- `ml/models/credit_model.pkl` and `ml/models/best_model.pkl`
- `ml/models/model_metadata.json`
- `ml/reports/model_comparison.csv`, `training_report.md`,
  `evaluation_report.md`, `model_card.md`, `feature_importance.csv`, and
  `feature_ranking.json`
- `ml/plots/model_comparison.png`, `prediction_vs_actual.png`,
  `residual_distribution.png`, `residual_histogram.png`, `shap_summary.png`,
  `shap_bar.png`, `feature_importance.png`, and `learning_curve.png`

The model and all reports are educational artifacts for synthetic data. They
must not be used to make lending, eligibility, pricing, or other decisions
about real people.
