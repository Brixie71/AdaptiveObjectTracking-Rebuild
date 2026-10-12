from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from aot.capture.types import FramePacket, SourceState
from aot.capture.worker import WorkerSnapshot
from aot.config.types import SourceType
from aot.detection.types import DetectionBatch
from aot.runtime.types import LiveRuntimeSnapshot, ProcessedFrame, SourceRuntimeSnapshot


def packet(source_id: str = "camera-1", sequence: int = 7) -> FramePacket:
    return FramePacket(source_id, sequence, 123, np.zeros((2, 3, 3), dtype=np.uint8))


def batch(source_id: str = "camera-1", sequence: int = 7) -> DetectionBatch:
    return DetectionBatch(source_id, sequence, 123, 3, 2, (), 0)


def test_runtime_snapshots_are_frozen_and_preserve_provenance() -> None:
    result = ProcessedFrame(packet(), batch(), None)
    capture = WorkerSnapshot("camera-1", SourceState.OPEN, 1, 0, None, True)
    source = SourceRuntimeSnapshot(
        "camera-1", "Front", SourceType.USB, capture, result
    )
    live = LiveRuntimeSnapshot(True, (source,))

    assert result.packet.source_id == result.detections.source_id
    assert result.packet.sequence == result.detections.frame_sequence
    assert live.sources == (source,)
    with pytest.raises(FrozenInstanceError):
        live.running = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("detection_source", "detection_sequence"),
    [("camera-2", 7), ("camera-1", 8)],
)
def test_processed_frame_rejects_mismatched_detection_provenance(
    detection_source: str,
    detection_sequence: int,
) -> None:
    with pytest.raises(ValueError, match="provenance"):
        ProcessedFrame(packet(), batch(detection_source, detection_sequence), None)


def test_source_snapshot_rejects_mismatched_capture_or_result() -> None:
    capture = WorkerSnapshot("camera-2", SourceState.OPEN, 0, 0, None, True)
    with pytest.raises(ValueError, match="source"):
        SourceRuntimeSnapshot(
            "camera-1", "Front", SourceType.USB, capture, ProcessedFrame(packet(), None, None)
        )
