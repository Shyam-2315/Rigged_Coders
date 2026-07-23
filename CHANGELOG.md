# Changelog

## v0.9.0

Added

- [x] FastAPI application factory with CORS, lifespan, request-timing middleware, and global error handlers
- [x] POST /api/v1/analyze — unified orchestrator endpoint (all phases in one call)
- [x] POST /api/v1/credit-score — Phase 3 XGBoost credit scoring with SHAP explanations
- [x] POST /api/v1/risk-profile — Phase 4 behavioral risk classifier with persona and probabilities
- [x] GET /api/v1/products — Phase 5 investment knowledge-base search with filters
- [x] POST /api/v1/recommend — Phase 6 deterministic portfolio recommendation
- [x] POST /api/v1/simulate — Phase 7 Monte Carlo financial projection
- [x] GET /api/v1/health and GET /api/v1/version system endpoints
- [x] API-level integration tests (TestClient) for all 7 endpoint groups
- [x] fastapi, uvicorn, and httpx added to requirements.txt

## v0.8.0

Added

- [x] Unified AI Orchestrator
- [x] Pipeline Management
- [x] Response Builder
- [x] Fault Tolerance
- [x] Audit Logging
- [x] Telemetry
- [x] Unified API Contract

## v0.6.0

Added

- [x] Product retrieval from the Phase 5 investment knowledge base
- [x] Deterministic weighted product scoring and repository-backed recommendation facade
- [x] Strategy-based portfolio optimizer and constrained allocation engine
- [x] Portfolio health metrics, product-level explanations, trade-offs, and actionable insights
- [x] Conservative, recommended, and growth alternative portfolios
- [x] Versioned recommendation rules plus generated portfolio examples and metrics reports

## v0.5.0

Added

- [x] Investment knowledge base with 25 structured products
- [x] Product repository and validation layer
- [x] Smart retrieval and explainable ranking
- [x] Market assumptions for future simulations

## v0.4.0

Added

- [x] Behavioral investor dataset with 50,000 correlated profiles
- [x] ML risk classifier with automatic multi-model selection
- [x] Behavioral personas with strengths and potential risks
- [x] Probability scores and model confidence
- [x] SHAP-compatible global and local explanations

## v0.3.0

Added

- Model Comparison
- Hyperparameter Tuning
- XGBoost Credit Model
- SHAP Explainability
- Model Versioning
- Inference Engine
- Evaluation Reports

## v0.2.0

Added

- ✔ Feature Engineering Pipeline
- ✔ Validation Engine
- ✔ Data Cleaning
- ✔ Feature Scaling
- ✔ OneHot Encoding
- ✔ Feature Selection
- ✔ Feature Importance Report

## v0.1.0

Added

- ✔ Synthetic Population Generator
- ✔ 100,000 realistic users
- ✔ Correlated financial behavior
- ✔ Engineered features
- ✔ Credit likelihood target
