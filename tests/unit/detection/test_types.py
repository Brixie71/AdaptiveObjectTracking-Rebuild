from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from aot.capture.types import FramePacket
from aot.detection.types import (
    BoundingBox,
    Detection,
    DetectionBatch,
    DetectionError,
    Detector,
)


class MinimalDetector:
    model_id = "fake-detector"
    class_names = ("person",)

    def detect(self, packet: FramePacket) -> DetectionBatch:
        height, width = packet.frame.shape[:2]
        return DetectionBatch(
            source_id=packet.source_id,
            frame_sequence=packet.sequence,
            captured_at_monotonic_ns=packet.captured_at_monotonic_ns,
            frame_width=width,
            frame_height=height,
            detections=(),
            rejected_count=0,
        )

    def close(self) -> None:
        return None


def test_detection_records_are_immutable_and_preserve_capture_provenance() -> None:
    packet = FramePacket(
        source_id="camera-1",
        sequence=7,
        captured_at_monotonic_ns=123456,
        frame=np.zeros((24, 32, 3), dtype=np.uint8),
    )
    detector = MinimalDetector()

    batch = detector.detect(packet)
    detection = Detection(0, "person", 0.9, BoundingBox(1.0, 2.0, 10.0, 20.0))

    assert isinstance(detector, Detector)
    assert batch.source_id == "camera-1"
    assert batch.frame_sequence == 7
    assert batch.captured_at_monotonic_ns == 123456
    assert (batch.frame_width, batch.frame_height) == (32, 24)
    with pytest.raises(FrozenInstanceError):
        detection.confidence = 0.1
    with pytest.raises(FrozenInstanceError):
        batch.rejected_count = 1


def test_detection_error_retains_only_the_supplied_safe_summary() -> None:
    error = DetectionError("Inference failed")

    assert str(error) == "Inference failed"
    assert error.summary == "Inference failed"
    assert error.args == ("Inference failed",)
