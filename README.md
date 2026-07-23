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

## Phase 4 — Behavioral AI Risk Profiling

### Purpose

Phase 4 adds a conversational-questionnaire classifier that estimates a synthetic investor risk profile (`Low`, `Medium`, or `High`) and returns a behavioral persona, class probabilities, model confidence, and local feature explanations. It is a research and product-prototyping component only—not personalised financial advice or an investment-suitability decision.

### Behavioral Dataset and Questionnaire

The generator creates **50,000 reproducible, correlated investor profiles**. Risk labels are deterministically derived from behavioral-finance characteristics—loss response, investment horizon, preparedness, experience, knowledge, liquidity preference, and confidence—not randomly assigned and not determined by income alone.

The questionnaire covers age, occupation, income stability, monthly income and savings, monthly investment budget, emergency-fund coverage, investment goal and horizon, expected annual return, reaction to a 20% market loss, prior experience, mutual-fund and stock knowledge, liquidity preference, dependents, financial confidence, investment frequency, and investment type.

Feature engineering produces transparent risk-tolerance, financial-preparedness, income-stability, experience, liquidity-preference, confidence, loss-recovery, investment-readiness, and long-term-orientation scores.

### Models and Selection

The training command compares Logistic Regression, Random Forest, XGBoost Classifier, Extra Trees, and HistGradientBoosting on an untouched stratified hold-out set. It records accuracy, weighted precision/recall/F1, multiclass one-vs-rest ROC AUC, training time, and prediction time, then automatically exports the best weighted-F1 model (using ROC AUC, accuracy, and prediction latency as deterministic tie-breakers).

SHAP-compatible selected models receive global-summary, global-bar, local-explanation, and ranked-top-feature artifacts.

### Training

Install dependencies and run from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m ml.risk.train_risk_model
```

The training command generates the default data set if it does not already exist. For a smaller estimator/SHAP smoke run:

```powershell
python -m ml.risk.train_risk_model --quick
```

Risk-specific outputs are kept separate from Phase 3:

- `ml/risk/data/behavioral_risk_profiles.csv` and `behavioral_risk_features.csv`
- `ml/models/risk_model.pkl`, `risk_encoder.pkl`, `risk_scaler.pkl`, and `risk_model_metadata.json`
- `ml/risk/reports/classification_report.md`, `training_metrics.json`, `feature_importance.csv`, and `shap_top_features.csv`
- `ml/risk/plots/confusion_matrix.png`, `roc_curve.png`, `shap_summary.png`, `shap_bar.png`, and `shap_local_explanation.png`

### Behavioral Personas and Inference

Each class resolves to a behavioral persona with its own description, investment philosophy, strengths, and potential risks:

- Low: **Secure Saver** or **Conservative Planner**
- Medium: **Balanced Builder** or **Strategic Investor**
- High: **Growth Explorer** or **Aggressive Wealth Seeker**

Use `predict_risk()` for a profile, probabilities, confidence, and persona, or `predict_with_explanation()` for a single questionnaire response plus model-derived positive and negative factors.

```python
from ml.risk.inference import predict_with_explanation

response = predict_with_explanation({
    "age": 34,
    "occupation": "Salaried professional",
    "income_stability": "Stable",
    "monthly_income": 90000,
    "monthly_savings": 25000,
    "monthly_investment_budget": 15000,
    "emergency_fund_months": 8,
    "investment_goal": "Retirement",
    "investment_horizon_years": 12,
    "expected_annual_return_percent": 12,
    "reaction_to_20_percent_loss": "Hold and wait",
    "previous_investment_experience": "Intermediate",
    "mutual_fund_knowledge": "Good",
    "stock_knowledge": "Basic",
    "preferred_liquidity": "Flexible",
    "dependents": 1,
    "financial_confidence": "High",
    "preferred_investment_frequency": "Monthly",
    "preferred_investment_type": "Balanced mutual funds",
})
```

The output is integration-ready and includes `risk_profile`, persona and persona details, `confidence`, `{low, medium, high}` probabilities, top positive/negative factors, and a non-prescriptive recommendation summary.

## Phase 5 — Investment Intelligence Knowledge Base

### Purpose

Phase 5 is a validated, structured repository of investment-product metadata for future TrustVest components. It retrieves and ranks product matches; it does **not** construct portfolios, allocate money, make a suitability determination, or provide personalised investment advice.

### Products and Market Assumptions

The catalog contains 25 investment product categories across cash, debt funds, deposits, government savings, Indian equity, commodities, REITs, and international equity. Each product has validated metadata for risk, personas, goals, return range, volatility, liquidity, lock-in, tax efficiency, horizon, eligibility flags, trust/popularity, potential risks, and plain-language explanation.

Illustrative planning assumptions for inflation, risk-free rate, equity premium, gold, debt, cash, and default Monte Carlo volatility are saved separately for Phase 7. They are not live market data or forecasts.

### Generate the Knowledge Base

```powershell
python -m pip install -r requirements.txt
python -m ml.investments.knowledge_base
```

Generated artifacts:

- `ml/data/investment_products.json` and `investment_products.csv`
- `ml/data/market_assumptions.json`
- `ml/reports/investment_catalog.md`, `investment_statistics.md`, `risk_distribution.json`, and `goal_distribution.json`
- `ml/docs/investment_intelligence_knowledge_base.md`

### Retrieval and Ranking

The repository-backed API supports `get_all_products()`, `get_product_by_id()`, filters for risk, goal, persona, horizon, liquidity, beginner status, government backing, and credit-score eligibility, plus ranked smart search.

```python
from ml.investments.retrieval import search_products

matches = search_products(
    "retirement",
    context={
        "risk_profile": "Medium",
        "persona": "Balanced Builder",
        "investment_horizon_years": 12,
    },
)
```

Search understands common intents such as `safe investment`, `gold`, `retirement`, `tax`, and `monthly income`. Results are ranked from explicit metadata matches—risk, goal, persona, horizon, liquidity, credit eligibility, popularity, and trust—and include the match reason, supported personas/goals, potential risks, illustrative return range, horizon, and a simple explanation.
