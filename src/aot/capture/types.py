"""Project-owned capture records, lifecycle states, and source contract."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np

from aot.config.types import SourceType


class SourceState(str, Enum):
    """Lifecycle states shared by every source adapter."""

    CLOSED = "closed"
    OPENING = "opening"
    OPEN = "open"
    RECONNECTING = "reconnecting"
    EOF = "eof"
    FAILED = "failed"


@dataclass(frozen=True)
class CaptureProperties:
    """Requested and backend-negotiated source properties."""

    requested_width: int
    requested_height: int
    requested_fps: float
    negotiated_width: int
    negotiated_height: int
    negotiated_fps: float


@dataclass(frozen=True)
class FramePacket:
    """One normalized frame and its source-local ordering metadata."""

    source_id: str
    sequence: int
    captured_at_monotonic_ns: int
    frame: np.ndarray


class CaptureError(RuntimeError):
    """Base error containing only a source identifier and safe summary."""

    def __init__(self, source_id: str, summary: str) -> None:
        self.source_id = source_id
        self.summary = summary
        super().__init__(summary)


class CaptureOpenError(CaptureError):
    """Raised when a source cannot acquire a usable capture handle."""


class CaptureReadError(CaptureError):
    """Raised when an opened live source cannot provide its next frame."""


class EndOfStream(CaptureError):
    """Raised when an offline source reaches its normal terminal state."""


@runtime_checkable
class FrameSource(Protocol):
    """Common lifecycle and frame contract for all capture adapters."""

    source_id: str
    source_type: SourceType
    reconnectable: bool
    state: SourceState
    properties: CaptureProperties | None

    def open(self) -> CaptureProperties:
        """Acquire the source and report negotiated properties."""

    def read(self) -> FramePacket:
        """Read one normalized frame or raise a project-owned error."""

    def close(self) -> None:
        """Release all resources; repeated calls are safe."""
