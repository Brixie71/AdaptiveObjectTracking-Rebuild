from __future__ import annotations

from pathlib import Path

from aot.capture.offline import OfflineFrameSource
from aot.capture.types import EndOfStream
from aot.config.types import CameraSourceConfig, DetectionConfig, SourceType
from aot.detection.yolo import YoloDetector


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "media" / "gate2-three-frame.avi"


class DeterministicBackend:
    class_names = {0: "person", 1: "backpack", 2: "car"}

    def predict(self, _frame, **_policy):
        return (
            (1, 2, 30, 40, 0.95, 0),
            (2, 3, 25, 35, 0.90, 1),
            (3, 4, 20, 30, 0.85, 2),
            (1, 2, 3),
        )

    def close(self) -> None:
        return None


def test_offline_capture_provenance_survives_allowlist_filtered_detection() -> None:
    source = OfflineFrameSource(
        CameraSourceConfig(
            source_id="offline-detection",
            name="Offline detection",
            source_type=SourceType.OFFLINE,
            enabled=True,
            offline_path=FIXTURE,
            width=320,
            height=240,
            fps=5.0,
        )
    )
    detector = YoloDetector(
        "detector-test",
        DetectionConfig(
            enabled=True,
            model_id="detector-test",
            allowed_classes=("person", "backpack"),
        ),
        DeterministicBackend(),
        "cpu",
    )

    source.open()
    batches = []
    try:
        while True:
            packet = source.read()
            batch = detector.detect(packet)
            assert batch.source_id == packet.source_id
            assert batch.frame_sequence == packet.sequence
            assert batch.captured_at_monotonic_ns == packet.captured_at_monotonic_ns
            assert (batch.frame_width, batch.frame_height) == (320, 240)
            batches.append(batch)
    except EndOfStream:
        pass
    finally:
        detector.close()
        source.close()

    assert len(batches) == 3
    assert [batch.frame_sequence for batch in batches] == [0, 1, 2]
    assert all(
        [item.class_name for item in batch.detections] == ["person", "backpack"]
        for batch in batches
    )
    assert all(batch.rejected_count == 2 for batch in batches)
