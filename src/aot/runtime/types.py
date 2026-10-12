"""Immutable records crossing the capture, processing, and UI boundaries."""

from __future__ import annotations

from dataclasses import dataclass

from aot.capture.types import FramePacket
from aot.capture.worker import WorkerSnapshot
from aot.config.types import SourceType
from aot.detection.types import DetectionBatch


@dataclass(frozen=True)
class ProcessedFrame:
    """A captured frame paired only with detections from that exact frame."""

    packet: FramePacket
    detections: DetectionBatch | None
    processing_error: str | None

    def __post_init__(self) -> None:
        detections = self.detections
        if detections is None:
            return
        if (
            detections.source_id != self.packet.source_id
            or detections.frame_sequence != self.packet.sequence
        ):
            raise ValueError("Detection provenance does not match the frame packet")


@dataclass(frozen=True)
class SourceRuntimeSnapshot:
    """Display-ready state for one configured source."""

    source_id: str
    display_name: str
    source_type: SourceType
    capture: WorkerSnapshot
    result: ProcessedFrame | None

    def __post_init__(self) -> None:
        if self.capture.source_id != self.source_id:
            raise ValueError("Capture snapshot source does not match source ID")
        if self.result is not None and self.result.packet.source_id != self.source_id:
            raise ValueError("Processed result source does not match source ID")


@dataclass(frozen=True)
class LiveRuntimeSnapshot:
    """One immutable presentation snapshot for the complete live runtime."""

    running: bool
    sources: tuple[SourceRuntimeSnapshot, ...]
