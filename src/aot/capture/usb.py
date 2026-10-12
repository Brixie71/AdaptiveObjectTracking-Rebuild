"""USB camera adapter."""

from __future__ import annotations

import time
from collections.abc import Callable

from aot.capture.backend import BackendFactory, open_cv_backend
from aot.capture.base import BaseFrameSource
from aot.capture.types import CaptureProperties, FramePacket
from aot.config.types import CameraSourceConfig, SourceType


class UsbFrameSource(BaseFrameSource):
    reconnectable = True

    def __init__(
        self,
        config: CameraSourceConfig,
        backend_factory: BackendFactory = open_cv_backend,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if config.source_type is not SourceType.USB:
            raise ValueError("USB source requires USB configuration")
        if config.device_index is None:
            raise ValueError("USB source requires a device index")
        super().__init__(config, backend_factory, clock_ns)

    def open(self) -> CaptureProperties:
        assert self._config.device_index is not None
        return self._open(self._config.device_index)

    def read(self) -> FramePacket:
        return self._packet_from(self._read_raw())
