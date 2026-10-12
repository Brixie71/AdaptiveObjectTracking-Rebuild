"""Secret-safe RTSP frame source."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from threading import Lock
from weakref import WeakKeyDictionary

from aot.capture.backend import BackendFactory, open_cv_backend
from aot.capture.base import BaseFrameSource
from aot.capture.types import (
    CaptureOpenError,
    CaptureProperties,
    CaptureReadError,
    FramePacket,
    SourceState,
)
from aot.config.types import CameraSourceConfig, SourceType


_INJECTED_ENVIRONMENTS: WeakKeyDictionary[object, Mapping[str, str]] = (
    WeakKeyDictionary()
)
_INJECTED_ENVIRONMENTS_LOCK = Lock()


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
        if environment is not None:
            with _INJECTED_ENVIRONMENTS_LOCK:
                _INJECTED_ENVIRONMENTS[self] = environment

    def _resolve_url(self) -> str | None:
        with _INJECTED_ENVIRONMENTS_LOCK:
            environment = _INJECTED_ENVIRONMENTS.get(self)
        values = os.environ if environment is None else environment
        return values.get(self.rtsp_env_key)

    def open(self) -> CaptureProperties:
        properties = None
        try:
            resolved = self._resolve_url()
            if resolved and resolved.strip():
                properties = self._open(resolved)
        except Exception:
            pass
        if properties is None:
            self.state = SourceState.FAILED
            raise CaptureOpenError(self.source_id, "RTSP source could not be opened")
        return properties

    def read(self) -> FramePacket:
        packet = None
        try:
            packet = self._packet_from(self._read_raw())
        except CaptureReadError:
            pass
        if packet is None:
            with self._backend_lock:
                backend = self._backend
                self._backend = None
            if backend is not None:
                try:
                    backend.release()
                except Exception:
                    pass
            raise CaptureReadError(self.source_id, "RTSP source read failed")
        return packet
