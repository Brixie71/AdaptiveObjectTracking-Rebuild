"""Shared lifecycle and packet construction for frame sources."""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Lock

import numpy as np

from aot.capture.backend import BackendFactory, BackendProperty
from aot.capture.normalize import FrameNormalizationError, normalize_frame
from aot.capture.types import (
    CaptureOpenError,
    CaptureProperties,
    CaptureReadError,
    FramePacket,
    SourceState,
)
from aot.config.types import CameraSourceConfig


class BaseFrameSource:
    """Reusable source mechanics with project-owned errors and state."""

    reconnectable = False

    def __init__(
        self,
        config: CameraSourceConfig,
        backend_factory: BackendFactory,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._config = config
        self._backend_factory = backend_factory
        self._clock_ns = clock_ns
        self._backend = None
        self._backend_lock = Lock()
        self._close_requested = False
        self._sequence = 0
        self.source_id = config.source_id
        self.source_type = config.source_type
        self.state = SourceState.CLOSED
        self.properties: CaptureProperties | None = None

    def _open(self, backend_input: int | str) -> CaptureProperties:
        with self._backend_lock:
            if self._backend is not None:
                raise CaptureOpenError(self.source_id, "Source is already open")
            self._close_requested = False
        self.state = SourceState.OPENING

        backend = None
        try:
            backend = self._backend_factory(backend_input)
        except Exception:
            pass
        if backend is None:
            self.state = SourceState.FAILED
            raise CaptureOpenError(self.source_id, "Source could not be opened")

        with self._backend_lock:
            cancelled = self._close_requested
            if not cancelled:
                self._backend = backend
        if cancelled:
            try:
                backend.release()
            except Exception:
                pass
            self.state = SourceState.CLOSED
            raise CaptureOpenError(self.source_id, "Source opening was cancelled")

        open_failed = False
        try:
            if not backend.is_opened():
                open_failed = True
            if not open_failed:
                backend.set(BackendProperty.WIDTH, float(self._config.width))
                backend.set(BackendProperty.HEIGHT, float(self._config.height))
                backend.set(BackendProperty.FPS, float(self._config.fps))
                self.properties = CaptureProperties(
                    requested_width=self._config.width,
                    requested_height=self._config.height,
                    requested_fps=self._config.fps,
                    negotiated_width=int(backend.get(BackendProperty.WIDTH)),
                    negotiated_height=int(backend.get(BackendProperty.HEIGHT)),
                    negotiated_fps=float(backend.get(BackendProperty.FPS)),
                )
        except Exception:
            open_failed = True

        with self._backend_lock:
            cancelled = self._close_requested or self._backend is not backend
        if open_failed or cancelled:
            self._fail_open()
            raise CaptureOpenError(self.source_id, "Source could not be opened")

        self._sequence = 0
        self.state = SourceState.OPEN
        assert self.properties is not None
        return self.properties

    def _fail_open(self) -> None:
        with self._backend_lock:
            backend = self._backend
            self._backend = None
        if backend is not None:
            try:
                backend.release()
            except Exception:
                pass
        self.state = SourceState.FAILED

    def _read_raw(self) -> np.ndarray:
        with self._backend_lock:
            backend = self._backend
        if self.state is not SourceState.OPEN or backend is None:
            raise CaptureReadError(self.source_id, "Source is not open")

        succeeded = False
        frame = None
        try:
            succeeded, frame = backend.read()
        except Exception:
            pass
        if not succeeded or frame is None or frame.size == 0:
            self.state = SourceState.FAILED
            raise CaptureReadError(self.source_id, "Source read failed")
        return frame

    def _packet_from(self, frame: np.ndarray) -> FramePacket:
        normalized = None
        try:
            normalized = normalize_frame(
                frame,
                width=self._config.width,
                height=self._config.height,
                rotation=self._config.rotation,
            )
        except FrameNormalizationError:
            pass
        if normalized is None:
            self.state = SourceState.FAILED
            raise CaptureReadError(self.source_id, "Source frame was invalid")
        packet = FramePacket(
            source_id=self.source_id,
            sequence=self._sequence,
            captured_at_monotonic_ns=self._clock_ns(),
            frame=normalized,
        )
        self._sequence += 1
        return packet

    def close(self) -> None:
        with self._backend_lock:
            self._close_requested = True
            backend = self._backend
            self._backend = None
        try:
            if backend is not None:
                backend.release()
        finally:
            self.state = SourceState.CLOSED
