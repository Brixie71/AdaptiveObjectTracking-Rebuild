"""Immutable detection records and the application detector contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from aot.capture.types import FramePacket


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: BoundingBox


@dataclass(frozen=True)
class DetectionBatch:
    source_id: str
    frame_sequence: int
    captured_at_monotonic_ns: int
    frame_width: int
    frame_height: int
    detections: tuple[Detection, ...]
    rejected_count: int


class DetectionError(RuntimeError):
    """Base error containing only a safe application-owned summary."""

    def __init__(self, summary: str) -> None:
        self.summary = summary
        super().__init__(summary)


class DetectorLoadError(DetectionError):
    """Raised when a detector cannot be initialized safely."""


class DetectorInferenceError(DetectionError):
    """Raised when a loaded detector cannot process a frame."""


@runtime_checkable
class Detector(Protocol):
    model_id: str
    class_names: tuple[str, ...]

    def detect(self, packet: FramePacket) -> DetectionBatch:
        """Return filtered project-owned detections for one captured frame."""

    def close(self) -> None:
        """Release detector resources; repeated calls must be safe."""
