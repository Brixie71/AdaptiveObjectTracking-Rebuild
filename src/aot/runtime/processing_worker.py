"""Single-threaded, bounded newest-frame processing for all live sources."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable

from aot.capture.types import FramePacket
from aot.detection.types import Detector
from aot.runtime.result_store import LatestResultStore
from aot.runtime.types import ProcessedFrame


class ProcessingWorker:
    """Poll each source in stable order and serialize detector access."""

    def __init__(
        self,
        source_ids: Iterable[str],
        packet_supplier: Callable[[str], FramePacket | None],
        result_store: LatestResultStore,
        detector: Detector | None = None,
        *,
        poll_seconds: float = 0.01,
        wait_for_stop: Callable[[float], bool] | None = None,
    ) -> None:
        identifiers = tuple(source_ids)
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("Source IDs must be non-empty and unique")
        if poll_seconds < 0:
            raise ValueError("Poll interval cannot be negative")
        self._source_ids = identifiers
        self._packet_supplier = packet_supplier
        self._result_store = result_store
        self._detector = detector
        self._poll_seconds = poll_seconds
        self._stop_event = threading.Event()
        self._wait_for_stop = wait_for_stop or self._stop_event.wait
        self._thread: threading.Thread | None = None
        self._last_sequences = {source_id: -1 for source_id in identifiers}
        self._errors = {source_id: None for source_id in identifiers}
        self._state_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Processing worker has already been started")
        self._thread = threading.Thread(
            target=self._run,
            name="frame-processing",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            for source_id in self._source_ids:
                if self._stop_event.is_set():
                    return
                try:
                    captured = self._packet_supplier(source_id)
                except Exception:
                    continue
                if captured is None or captured.source_id != source_id:
                    continue
                if captured.sequence <= self._last_sequences[source_id]:
                    continue
                self._last_sequences[source_id] = captured.sequence
                self._process(captured)
            if self._wait_for_stop(self._poll_seconds):
                return

    def _process(self, captured: FramePacket) -> None:
        try:
            detections = (
                None if self._detector is None else self._detector.detect(captured)
            )
            result = ProcessedFrame(captured, detections, None)
            error = None
        except Exception:
            error = "Detector inference failed"
            result = ProcessedFrame(captured, None, error)
        with self._state_lock:
            self._errors[captured.source_id] = error
        self._result_store.put(result)

    def request_stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot_error(self, source_id: str) -> str | None:
        with self._state_lock:
            if source_id not in self._errors:
                raise KeyError(source_id)
            return self._errors[source_id]
