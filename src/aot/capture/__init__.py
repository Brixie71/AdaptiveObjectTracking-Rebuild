"""Camera and offline-video capture boundaries."""

from .factory import create_frame_source
from .types import (
    CaptureError,
    CaptureOpenError,
    CaptureProperties,
    CaptureReadError,
    EndOfStream,
    FramePacket,
    FrameSource,
    SourceState,
)
from .offline import OfflineFrameSource
from .rtsp import RtspFrameSource
from .usb import UsbFrameSource

__all__ = [
    "CaptureError",
    "CaptureOpenError",
    "CaptureProperties",
    "CaptureReadError",
    "create_frame_source",
    "EndOfStream",
    "FramePacket",
    "FrameSource",
    "OfflineFrameSource",
    "RtspFrameSource",
    "SourceState",
    "UsbFrameSource",
]
