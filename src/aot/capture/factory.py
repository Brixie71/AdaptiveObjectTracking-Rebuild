"""Central frame-source selection and dependency injection."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

from aot.capture.backend import BackendFactory, open_cv_backend
from aot.capture.offline import OfflineFrameSource
from aot.capture.rtsp import RtspFrameSource
from aot.capture.types import CaptureError, FrameSource
from aot.capture.usb import UsbFrameSource
from aot.config.types import CameraSourceConfig, SourceType


def create_frame_source(
    config: CameraSourceConfig,
    *,
    environment: Mapping[str, str] | None = None,
    backend_factory: BackendFactory = open_cv_backend,
    clock_ns: Callable[[], int] = time.monotonic_ns,
) -> FrameSource:
    """Build the sole adapter appropriate for a validated source record."""
    if config.source_type not in {
        SourceType.USB,
        SourceType.RTSP,
        SourceType.OFFLINE,
    }:
        raise CaptureError(config.source_id, "Unsupported source type")

    source_fields = (
        config.device_index,
        config.rtsp_env_key,
        config.offline_path,
    )
    populated = sum(value is not None for value in source_fields)
    if populated != 1:
        raise CaptureError(config.source_id, "Invalid source configuration")

    try:
        if config.source_type is SourceType.USB:
            if config.device_index is None:
                raise ValueError
            return UsbFrameSource(config, backend_factory, clock_ns)
        if config.source_type is SourceType.RTSP:
            if not config.rtsp_env_key:
                raise ValueError
            return RtspFrameSource(config, environment, backend_factory, clock_ns)
        if config.source_type is SourceType.OFFLINE:
            if config.offline_path is None:
                raise ValueError
            return OfflineFrameSource(config, backend_factory, clock_ns)
    except ValueError as error:
        raise CaptureError(config.source_id, "Invalid source configuration") from error
    raise CaptureError(config.source_id, "Unsupported source type")
