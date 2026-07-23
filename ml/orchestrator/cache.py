"""Optional in-memory response cache for deterministic orchestrator requests."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from threading import RLock
from typing import Any

from .schemas import UnifiedAnalysisRequest, UnifiedAnalysisResponse


class OrchestratorCache:
    """Simple LRU cache keyed by a canonical request hash."""

    def __init__(self, *, max_entries: int = 256) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[str, UnifiedAnalysisResponse] = OrderedDict()
        self._lock = RLock()

    def get(self, request: UnifiedAnalysisRequest) -> UnifiedAnalysisResponse | None:
        key = self._cache_key(request)
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def set(self, request: UnifiedAnalysisRequest, response: UnifiedAnalysisResponse) -> None:
        key = self._cache_key(request)
        with self._lock:
            self._entries[key] = response
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    @staticmethod
    def _cache_key(request: UnifiedAnalysisRequest) -> str:
        payload = request.model_dump(mode="json", exclude={"request_id"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class NullCache:
    """No-op cache implementation used when caching is disabled."""

    def get(self, request: UnifiedAnalysisRequest) -> UnifiedAnalysisResponse | None:
        return None

    def set(self, request: UnifiedAnalysisRequest, response: UnifiedAnalysisResponse) -> None:
        return None

    def clear(self) -> None:
        return None
