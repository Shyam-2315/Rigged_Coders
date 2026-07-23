"""Phase 4: Behavioral AI risk profiling for TrustVest AI.

The package turns conversational-investor questionnaire answers into an
explainable synthetic risk-profile prediction.  It is designed for product
prototyping and research; it is not personalised investment advice.
"""

from .inference import predict_risk, predict_with_explanation

__all__ = ["predict_risk", "predict_with_explanation"]
