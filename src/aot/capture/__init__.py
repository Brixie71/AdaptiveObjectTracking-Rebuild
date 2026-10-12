"""Camera and offline-video capture boundaries."""

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
from .usb import UsbFrameSource

__all__ = [
    "CaptureError",
    "CaptureOpenError",
    "CaptureProperties",
    "CaptureReadError",
    "EndOfStream",
    "FramePacket",
    "FrameSource",
    "OfflineFrameSource",
    "SourceState",
    "UsbFrameSource",
]
