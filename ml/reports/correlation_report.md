# Processed Dataset Correlation Report

Correlations are Pearson coefficients calculated after numeric scaling and feature selection.

## Strongest Positive Correlations with `credit_likelihood`

| Rank | Feature | Correlation |
| ---: | --- | ---: |
| 1 | `financial_discipline_index` | 0.8811 |
| 2 | `financial_health_score` | 0.8440 |
| 3 | `utility_bill_payment_rate` | 0.7928 |
| 4 | `repayment_consistency` | 0.7452 |
| 5 | `income_stability_score` | 0.7249 |
| 6 | `bank_balance` | 0.7186 |
| 7 | `emergency_fund_months` | 0.7170 |
| 8 | `age_x_income` | 0.6706 |
| 9 | `monthly_income` | 0.6667 |
| 10 | `banking_maturity_score` | 0.6649 |

## Strongest Negative Correlations with `credit_likelihood`

| Rank | Feature | Correlation |
| ---: | --- | ---: |
| 1 | `risk_indicator` | -0.7530 |
| 2 | `missed_utility_payments` | -0.5618 |
| 3 | `employment_type_Not employed` | -0.5296 |
| 4 | `occupation_Homemaker` | -0.4454 |
| 5 | `existing_small_loans` | -0.3472 |
| 6 | `occupation_Student` | -0.2732 |
| 7 | `wallet_usage_Occasional` | -0.2665 |
| 8 | `cash_transaction_ratio` | -0.2561 |
| 9 | `atm_withdrawals` | -0.2353 |
| 10 | `city_tier_Tier 3` | -0.2327 |

Correlation-pruning threshold: `0.95`.