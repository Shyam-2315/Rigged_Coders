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
