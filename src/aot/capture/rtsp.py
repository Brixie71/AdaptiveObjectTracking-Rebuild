"""Secret-safe RTSP frame source."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

from aot.capture.backend import BackendFactory, open_cv_backend
from aot.capture.base import BaseFrameSource
from aot.capture.types import (
    CaptureOpenError,
    CaptureProperties,
    CaptureReadError,
    FramePacket,
)
from aot.config.loader import resolve_rtsp_url
from aot.config.types import CameraSourceConfig, SourceType


class RtspFrameSource(BaseFrameSource):
    reconnectable = True

    def __init__(
        self,
        config: CameraSourceConfig,
        environment: Mapping[str, str] | None = None,
        backend_factory: BackendFactory = open_cv_backend,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if config.source_type is not SourceType.RTSP:
            raise ValueError("RTSP source requires RTSP configuration")
        if not config.rtsp_env_key:
            raise ValueError("RTSP source requires an environment key")
        super().__init__(config, backend_factory, clock_ns)
        self.rtsp_env_key = config.rtsp_env_key
        self._url_resolver = lambda: resolve_rtsp_url(config, environment)

    def open(self) -> CaptureProperties:
        try:
            return self._open(self._url_resolver())
        except Exception as error:
            raise CaptureOpenError(
                self.source_id, "RTSP source could not be opened"
            ) from error

    def read(self) -> FramePacket:
        try:
            return self._packet_from(self._read_raw())
        except CaptureReadError:
            if self._backend is not None:
                self._backend.release()
                self._backend = None
            raise
