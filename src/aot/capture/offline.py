"""Finite offline-video frame source."""

from __future__ import annotations

import time
from collections.abc import Callable

from aot.capture.backend import BackendFactory, open_cv_backend
from aot.capture.base import BaseFrameSource
from aot.capture.types import (
    CaptureOpenError,
    CaptureProperties,
    CaptureReadError,
    EndOfStream,
    FramePacket,
    SourceState,
)
from aot.config.types import CameraSourceConfig, SourceType


class OfflineFrameSource(BaseFrameSource):
    reconnectable = False

    def __init__(
        self,
        config: CameraSourceConfig,
        backend_factory: BackendFactory = open_cv_backend,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if config.source_type is not SourceType.OFFLINE:
            raise ValueError("Offline source requires offline configuration")
        if config.offline_path is None:
            raise ValueError("Offline source requires a file path")
        if not config.offline_path.is_absolute():
            raise ValueError("Offline source path must be absolute")
        super().__init__(config, backend_factory, clock_ns)

    def open(self) -> CaptureProperties:
        assert self._config.offline_path is not None
        path = self._config.offline_path.resolve()
        if not path.is_file():
            self.state = SourceState.FAILED
            raise CaptureOpenError(self.source_id, "Offline source file is unavailable")
        return self._open(str(path))

    def read(self) -> FramePacket:
        try:
            frame = self._read_raw()
        except CaptureReadError as error:
            self.state = SourceState.EOF
            raise EndOfStream(self.source_id, "Offline source reached end of stream") from error
        return self._packet_from(frame)
