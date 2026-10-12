"""Shared lifecycle and packet construction for frame sources."""

from __future__ import annotations

import time
from collections.abc import Callable

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
        self._sequence = 0
        self.source_id = config.source_id
        self.source_type = config.source_type
        self.state = SourceState.CLOSED
        self.properties: CaptureProperties | None = None

    def _open(self, backend_input: int | str) -> CaptureProperties:
        if self._backend is not None:
            raise CaptureOpenError(self.source_id, "Source is already open")
        self.state = SourceState.OPENING
        try:
            backend = self._backend_factory(backend_input)
            self._backend = backend
            if not backend.is_opened():
                raise CaptureOpenError(self.source_id, "Source could not be opened")
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
        except CaptureOpenError:
            self._fail_open()
            raise
        except Exception as error:
            self._fail_open()
            raise CaptureOpenError(
                self.source_id, "Source could not be opened"
            ) from error

        self._sequence = 0
        self.state = SourceState.OPEN
        return self.properties

    def _fail_open(self) -> None:
        if self._backend is not None:
            self._backend.release()
            self._backend = None
        self.state = SourceState.FAILED

    def _read_raw(self) -> np.ndarray:
        if self.state is not SourceState.OPEN or self._backend is None:
            raise CaptureReadError(self.source_id, "Source is not open")
        try:
            succeeded, frame = self._backend.read()
        except Exception as error:
            self.state = SourceState.FAILED
            raise CaptureReadError(self.source_id, "Source read failed") from error
        if not succeeded or frame is None or frame.size == 0:
            self.state = SourceState.FAILED
            raise CaptureReadError(self.source_id, "Source read failed")
        return frame

    def _packet_from(self, frame: np.ndarray) -> FramePacket:
        try:
            normalized = normalize_frame(
                frame,
                width=self._config.width,
                height=self._config.height,
                rotation=self._config.rotation,
            )
        except FrameNormalizationError as error:
            self.state = SourceState.FAILED
            raise CaptureReadError(self.source_id, "Source frame was invalid") from error
        packet = FramePacket(
            source_id=self.source_id,
            sequence=self._sequence,
            captured_at_monotonic_ns=self._clock_ns(),
            frame=normalized,
        )
        self._sequence += 1
        return packet

    def close(self) -> None:
        if self._backend is not None:
            self._backend.release()
            self._backend = None
        self.state = SourceState.CLOSED
