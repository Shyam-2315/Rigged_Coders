"""Audit record generation for Phase 8 orchestration runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .config import OrchestratorConfig
from .schemas import AuditRecord, PipelineTraceEntry, UnifiedAnalysisRequest, UnifiedAnalysisResponse


def build_audit_record(
    request: UnifiedAnalysisRequest,
    response: UnifiedAnalysisResponse,
    *,
    config: OrchestratorConfig,
    models_used: tuple[str, ...],
    simulation_seed: int | None,
) -> AuditRecord:
    """Create a structured audit log entry for one orchestration run."""
    return AuditRecord(
        timestamp=response.timestamp,
        request_id=response.request_id,
        pipeline_version=config.pipeline_version,
        models_used=models_used,
        simulation_seed=simulation_seed,
        execution_time_ms=response.telemetry.total_pipeline_time_ms,
        warnings=response.warnings,
        stage_statuses={entry.stage.value: entry.status.value for entry in response.pipeline_trace},
        educational_disclaimer=config.educational_disclaimer,
    )


def audit_record_to_dict(record: AuditRecord) -> dict[str, object]:
    """Serialize an audit record for JSON export."""
    return record.model_dump(mode="json")


def append_audit_record(record: AuditRecord, *, path: Path) -> None:
    """Append one audit record as a JSON line for downstream log ingestion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(audit_record_to_dict(record), ensure_ascii=False) + "\n")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
