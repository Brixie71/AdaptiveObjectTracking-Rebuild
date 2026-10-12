"""Restartable ownership boundary for capture, processing, and detection."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable

from aot.capture.manager import CaptureManager, ManagerShutdownError
from aot.capture.types import SourceState
from aot.capture.worker import WorkerSnapshot
from aot.config.types import AppSettings, CaptureRuntimeConfig
from aot.detection.types import Detector
from aot.runtime.processing_worker import ProcessingWorker
from aot.runtime.result_store import LatestResultStore
from aot.runtime.types import LiveRuntimeSnapshot, SourceRuntimeSnapshot


class LiveRuntimeConfigError(RuntimeError):
    """Raised when the configured live source set cannot form a UI session."""


class LiveRuntimeShutdownError(RuntimeError):
    """Raised after bounded cleanup could not stop every owned resource."""

    def __init__(self, source_ids: Iterable[str]) -> None:
        self.source_ids = tuple(dict.fromkeys(source_ids))
        suffix = ", ".join(self.source_ids)
        super().__init__(f"Live runtime did not stop cleanly: {suffix}")


class LiveRuntime:
    """Compose one restartable live session without exposing mutable state."""

    def __init__(
        self,
        settings: AppSettings,
        capture_manager_factory: Callable[[CaptureRuntimeConfig], CaptureManager],
        detector_factory: Callable[[], Detector | None],
        *,
        processing_worker_factory: Callable[..., ProcessingWorker] | None = None,
    ) -> None:
        self._settings = settings
        self._sources = tuple(source for source in settings.sources if source.enabled)
        self._capture_manager_factory = capture_manager_factory
        self._detector_factory = detector_factory
        self._processing_worker_factory = processing_worker_factory or ProcessingWorker
        self._lock = threading.Lock()
        self._capture: CaptureManager | None = None
        self._detector: Detector | None = None
        self._processing: ProcessingWorker | None = None
        self._results: LatestResultStore | None = None
        self._running = False
        self._transitioning = False

    def _validate(self) -> None:
        count = len(self._sources)
        if count < 1 or count > 3:
            raise LiveRuntimeConfigError("Live view requires one to three enabled sources")
        identifiers = tuple(source.source_id for source in self._sources)
        if len(set(identifiers)) != len(identifiers):
            raise LiveRuntimeConfigError("Enabled source IDs must be unique")

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def requires_cleanup(self) -> bool:
        """Report whether a previous lifecycle operation retained resources."""
        with self._lock:
            return any(
                item is not None
                for item in (self._capture, self._processing, self._detector)
            ) and not self._running

    def start(self) -> None:
        self._validate()
        with self._lock:
            if self._running:
                raise RuntimeError("Live runtime is already running")
            if self._transitioning or any(
                item is not None
                for item in (self._capture, self._processing, self._detector)
            ):
                raise RuntimeError("Live runtime lifecycle operation is in progress")
            self._transitioning = True

        capture: CaptureManager | None = None
        detector: Detector | None = None
        processing: ProcessingWorker | None = None
        try:
            capture = self._capture_manager_factory(self._settings.capture)
            capture.start(self._sources)
            if self._settings.detection.enabled:
                detector = self._detector_factory()
            source_ids = tuple(source.source_id for source in self._sources)
            results = LatestResultStore(source_ids)
            processing = self._processing_worker_factory(
                source_ids, capture.latest, results, detector
            )
            processing.start()
            with self._lock:
                self._capture = capture
                self._detector = detector
                self._processing = processing
                self._results = results
                self._running = True
                self._transitioning = False
        except Exception:
            processing_stopped = True
            if processing is not None:
                processing.request_stop()
                processing_stopped = processing.join(
                    self._settings.capture.shutdown_timeout_seconds
                )
            capture_stopped = capture is None
            if capture is not None:
                try:
                    capture.stop()
                except Exception:
                    capture_stopped = False
                else:
                    capture_stopped = True
            detector_closed = detector is None
            if detector is not None and processing_stopped:
                try:
                    detector.close()
                except Exception:
                    detector_closed = False
                else:
                    detector_closed = True
            with self._lock:
                self._capture = None if capture_stopped else capture
                self._processing = None if processing_stopped else processing
                self._detector = None if detector_closed else detector
                self._results = None
                self._running = False
                self._transitioning = False
            raise RuntimeError("Live runtime failed to start") from None

    def snapshot(self) -> LiveRuntimeSnapshot:
        with self._lock:
            capture = self._capture
            results = self._results
            running = self._running

        capture_snapshots = capture.snapshot() if capture is not None else {}
        result_snapshots = results.snapshot() if results is not None else {}
        source_snapshots: list[SourceRuntimeSnapshot] = []
        for configured in self._sources:
            capture_snapshot = capture_snapshots.get(configured.source_id)
            if capture_snapshot is None:
                capture_snapshot = WorkerSnapshot(
                    configured.source_id,
                    SourceState.CLOSED,
                    0,
                    0,
                    None,
                    False,
                )
            source_snapshots.append(
                SourceRuntimeSnapshot(
                    configured.source_id,
                    configured.name,
                    configured.source_type,
                    capture_snapshot,
                    result_snapshots.get(configured.source_id),
                )
            )
        return LiveRuntimeSnapshot(running, tuple(source_snapshots))

    def stop(self) -> None:
        with self._lock:
            if self._transitioning:
                raise RuntimeError("Live runtime lifecycle operation is in progress")
            capture = self._capture
            processing = self._processing
            detector = self._detector
            if capture is None and processing is None and detector is None:
                self._running = False
                return
            self._transitioning = True
            self._running = False

        affected: list[str] = []
        source_ids = tuple(source.source_id for source in self._sources)
        processing_stopped = True
        if processing is not None:
            processing.request_stop()
            processing_stopped = processing.join(
                self._settings.capture.shutdown_timeout_seconds
            )
            if not processing_stopped:
                affected.extend(source_ids)

        capture_stopped = True
        if capture is not None:
            try:
                capture.stop()
            except ManagerShutdownError as error:
                capture_stopped = False
                affected.extend(error.source_ids)
            except Exception:
                capture_stopped = False
                affected.extend(source_ids)

        detector_closed = detector is None
        if detector is not None and processing_stopped:
            try:
                detector.close()
                detector_closed = True
            except Exception:
                affected.extend(source_ids)

        with self._lock:
            if processing_stopped:
                self._processing = None
            if capture_stopped:
                self._capture = None
                self._results = None
            if detector_closed:
                self._detector = None
            self._transitioning = False

        if affected:
            ordered = [source_id for source_id in source_ids if source_id in affected]
            raise LiveRuntimeShutdownError(ordered)
