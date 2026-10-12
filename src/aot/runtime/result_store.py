"""Thread-safe, bounded newest-result storage."""

from __future__ import annotations

from collections.abc import Iterable
from threading import Lock

from aot.runtime.types import ProcessedFrame


class LatestResultStore:
    """Retain at most one monotonically newer result for each known source."""

    def __init__(self, source_ids: Iterable[str]) -> None:
        identifiers = tuple(source_ids)
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("Source IDs must be non-empty and unique")
        self._results: dict[str, ProcessedFrame | None] = {
            source_id: None for source_id in identifiers
        }
        self._lock = Lock()

    def put(self, result: ProcessedFrame) -> bool:
        source_id = result.packet.source_id
        with self._lock:
            if source_id not in self._results:
                raise KeyError(source_id)
            current = self._results[source_id]
            if current is not None and result.packet.sequence <= current.packet.sequence:
                return False
            self._results[source_id] = result
            return True

    def get(self, source_id: str) -> ProcessedFrame | None:
        with self._lock:
            if source_id not in self._results:
                raise KeyError(source_id)
            return self._results[source_id]

    def snapshot(self) -> dict[str, ProcessedFrame | None]:
        with self._lock:
            return dict(self._results)
