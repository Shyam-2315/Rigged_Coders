"""Phase 8 TrustVest Intelligence Orchestrator (TIO).

The package exposes a single unified interface that coordinates credit scoring,
risk profiling, knowledge-base retrieval, portfolio recommendation, and financial
projection modules for FastAPI and frontend consumption.
"""

from .orchestrator import (
    TrustVestIntelligenceOrchestrator,
    analyze_user,
    generate_phase8_reports,
)
from .schemas import UnifiedAnalysisRequest, UnifiedAnalysisResponse

__all__ = [
    "TrustVestIntelligenceOrchestrator",
    "UnifiedAnalysisRequest",
    "UnifiedAnalysisResponse",
    "analyze_user",
    "generate_phase8_reports",
]
