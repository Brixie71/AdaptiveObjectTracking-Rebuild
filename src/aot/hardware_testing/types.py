"""Sanitized result records for Settings hardware testing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from aot.capture.types import CaptureProperties
from aot.config.types import SourceType


class HardwareTestStatus(str, Enum):
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    DEGRADED = "degraded"
    SUCCESS = "success"


@dataclass(frozen=True)
class SourceTestResult:
    """Probe metrics; p95 latency uses the nearest-rank method."""

    source_id: str
    source_type: SourceType | str
    status: HardwareTestStatus
    properties: CaptureProperties | None
    frames_requested: int
    frames_received: int
    mean_capture_latency_ms: float | None
    p95_capture_latency_ms: float | None
    dropped_frames: int
    started_at_utc: datetime
    ended_at_utc: datetime
    summary: str

    def to_dict(self) -> dict[str, object]:
        source_type = (
            self.source_type.value
            if isinstance(self.source_type, SourceType)
            else str(self.source_type)
        )
        requested = None
        negotiated = None
        if self.properties is not None:
            requested = {
                "width": self.properties.requested_width,
                "height": self.properties.requested_height,
                "fps": self.properties.requested_fps,
            }
            negotiated = {
                "width": self.properties.negotiated_width,
                "height": self.properties.negotiated_height,
                "fps": self.properties.negotiated_fps,
            }
        return {
            "source_id": self.source_id,
            "source_type": source_type,
            "status": self.status.value,
            "requested_properties": requested,
            "negotiated_properties": negotiated,
            "frames_requested": self.frames_requested,
            "frames_received": self.frames_received,
            "mean_capture_latency_ms": self.mean_capture_latency_ms,
            "p95_capture_latency_ms": self.p95_capture_latency_ms,
            "dropped_frames": self.dropped_frames,
            "started_at_utc": self.started_at_utc.isoformat(),
            "ended_at_utc": self.ended_at_utc.isoformat(),
            "summary": self.summary,
        }


@dataclass(frozen=True)
class DetectorTestResult:
    """Sanitized detector-loading and inference benchmark result."""

    model_id: str
    status: HardwareTestStatus
    device: str
    warmup_runs: int
    measured_runs: int
    successful_runs: int
    mean_inference_ms: float | None
    p95_inference_ms: float | None
    detection_count: int
    started_at_utc: datetime
    ended_at_utc: datetime
    summary: str

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "status": self.status.value,
            "device": self.device,
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
            "successful_runs": self.successful_runs,
            "mean_inference_ms": self.mean_inference_ms,
            "p95_inference_ms": self.p95_inference_ms,
            "detection_count": self.detection_count,
            "started_at_utc": self.started_at_utc.isoformat(),
            "ended_at_utc": self.ended_at_utc.isoformat(),
            "summary": self.summary,
        }


@dataclass(frozen=True)
class UsbDeviceCandidate:
    device_index: int
    width: int
    height: int
    fps: float

    def to_dict(self) -> dict[str, object]:
        return {
            "device_index": self.device_index,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }
