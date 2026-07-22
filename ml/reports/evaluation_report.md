# Credit Model Evaluation Report

## Hold-out Validation Metrics

| Metric | Value |
| --- | ---: |
| RMSE | 0.5824 |
| MAE | 0.4380 |
| R² | 0.9975 |
| MAPE | 0.87% |

## Residual Diagnostics

| Statistic | Value |
| --- | ---: |
| Mean residual | -0.0030 |
| Median residual | 0.0036 |
| Residual standard deviation | 0.5824 |
| 95th percentile absolute residual | 1.1780 |

Residuals are calculated as actual minus predicted credit likelihood. These
figures are for the synthetic hold-out data only and do not establish
performance for real people or lending decisions.

## Generated Visuals

- `ml/plots/prediction_vs_actual.png`
- `ml/plots/residual_distribution.png`
- `ml/plots/residual_histogram.png`
