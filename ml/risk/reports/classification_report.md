# Behavioral Risk Classifier Report

## Selected Model

`Logistic Regression` was selected automatically using weighted F1, then weighted one-vs-rest ROC AUC, accuracy, and lower prediction latency as tie-breakers.

## Hold-out Model Comparison

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) | ROC AUC OvR (weighted) | Train (s) | Predict (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9961 | 0.9961 | 0.9961 | 0.9961 | 0.9999 | 0.493 | 0.0033 |
| XGBoost Classifier | 0.9879 | 0.9879 | 0.9879 | 0.9879 | 0.9995 | 3.271 | 0.0435 |
| HistGradientBoosting | 0.9873 | 0.9873 | 0.9873 | 0.9873 | 0.9995 | 3.403 | 0.0982 |
| Random Forest | 0.9653 | 0.9677 | 0.9653 | 0.9657 | 0.9982 | 0.996 | 0.1426 |
| Extra Trees | 0.9644 | 0.9663 | 0.9644 | 0.9648 | 0.9973 | 0.878 | 0.1783 |

## Per-Class Report

```
              precision    recall  f1-score   support

         Low       0.99      1.00      0.99      1484
      Medium       1.00      1.00      1.00      6936
        High       1.00      0.99      0.99      1580

    accuracy                           1.00     10000
   macro avg       1.00      0.99      0.99     10000
weighted avg       1.00      1.00      1.00     10000

```

## Explainability

Generated global summary, bar plot, local explanation, and top-feature report.

The classifier is trained only on synthetic questionnaire data. It supports product research and is not personalised financial advice or a suitability determination.
