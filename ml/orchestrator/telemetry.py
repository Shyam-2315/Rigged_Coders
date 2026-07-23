"""Telemetry capture for Phase 8 orchestration runs."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from .schemas import ModuleTelemetry, PipelineTraceEntry, StageSkipped, StageStatus, TelemetrySummary


@dataclass
class TelemetryCollector:
    """Collect per-module latency and aggregate pipeline metrics."""

    module_metrics: list[ModuleTelemetry] = field(default_factory=list)
    pipeline_started_at: float = field(default_factory=time.perf_counter)

    @contextmanager
    def track_module(self, module: str):
        started = time.perf_counter()
        status = StageStatus.SUCCESS
        error: Exception | None = None
        try:
            yield
        except StageSkipped:
            status = StageStatus.SKIPPED
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced in pipeline trace
            status = StageStatus.FAILED
            error = exc
            raise
        finally:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self.module_metrics.append(
                ModuleTelemetry(module=module, status=status, latency_ms=round(latency_ms, 3))
            )
            if error is not None:
                _ = error

    def build_summary(self, pipeline_trace: tuple[PipelineTraceEntry, ...]) -> TelemetrySummary:
        total_ms = (time.perf_counter() - self.pipeline_started_at) * 1000.0
        successful = sum(
            1 for entry in pipeline_trace if entry.status in (StageStatus.SUCCESS, StageStatus.SKIPPED)
        )
        success_rate = (successful / len(pipeline_trace) * 100.0) if pipeline_trace else 0.0
        return TelemetrySummary(
            total_pipeline_time_ms=round(total_ms, 3),
            module_metrics=tuple(self.module_metrics),
            success_rate=round(success_rate, 2),
        )
