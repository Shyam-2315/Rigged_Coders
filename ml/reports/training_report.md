# Credit Model Training Report

## Data Split

- Dataset: `processed_credit_dataset.csv`
- Train/test split: `80%` / `20%`
- Random seed: `2026`
- Cross-validation: `5`-fold KFold

## Model Comparison

| Model | RMSE | MAE | R² | MAPE | Train time (s) | Predict time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Extra Trees Regressor | 1.0168 | 0.7819 | 0.9922 | 1.53% | 1.19 | 0.0315 |
| Random Forest Regressor | 1.0739 | 0.8276 | 0.9914 | 1.61% | 2.09 | 0.0220 |
| XGBoost Regressor | 1.1747 | 0.9051 | 0.9897 | 1.78% | 0.40 | 0.0070 |
| HistGradientBoostingRegressor | 4.8350 | 3.7961 | 0.8247 | 7.79% | 1.58 | 0.0082 |
| Gradient Boosting Regressor | 5.4458 | 4.2580 | 0.7776 | 8.72% | 7.05 | 0.0110 |

XGBoost is the final model because it is a high-capacity, regularised gradient-boosted tree model that supports exact tree contributions for local explanations. The comparison table is retained as empirical evidence; final selection also prioritises the requested explainable XGBoost deployment path.

## Hyperparameter Optimisation

RandomizedSearchCV used negative RMSE across 5 folds. Best cross-validation RMSE: **0.5982**.

| Parameter | Selected value |
| --- | --- |
| `subsample` | `0.85` |
| `n_estimators` | `300` |
| `min_child_weight` | `1` |
| `max_depth` | `8` |
| `learning_rate` | `0.075` |
| `gamma` | `0.05` |
| `colsample_bytree` | `1.0` |

## Hold-out Metrics

| RMSE | MAE | R² | MAPE |
| ---: | ---: | ---: | ---: |
| 0.5824 | 0.4380 | 0.9975 | 0.87% |

## Outputs

- `ml/models/credit_model.pkl` and `ml/models/best_model.pkl`
- `ml/models/model_metadata.json`
- `ml/reports/model_comparison.csv`, `feature_importance.csv`, and `feature_ranking.json`
- SHAP, diagnostic, comparison, and learning-curve plots in `ml/plots/`
