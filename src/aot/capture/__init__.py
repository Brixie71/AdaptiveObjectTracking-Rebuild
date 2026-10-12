"""Camera and offline-video capture boundaries."""

from .factory import create_frame_source
from .buffer import NewestFrameBuffer
from .manager import CaptureManager, ManagerShutdownError
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
from .worker import CaptureWorker, ReconnectPolicy, WorkerSnapshot

__all__ = [
    "CaptureError",
    "CaptureManager",
    "CaptureOpenError",
    "CaptureProperties",
    "CaptureReadError",
    "create_frame_source",
    "EndOfStream",
    "FramePacket",
    "FrameSource",
    "ManagerShutdownError",
    "NewestFrameBuffer",
    "OfflineFrameSource",
    "RtspFrameSource",
    "SourceState",
    "CaptureWorker",
    "ReconnectPolicy",
    "UsbFrameSource",
    "WorkerSnapshot",
]
