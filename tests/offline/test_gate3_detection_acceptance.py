from __future__ import annotations

from pathlib import Path

from aot.capture.offline import OfflineFrameSource
from aot.config.types import CameraSourceConfig, DetectionConfig, SourceType
from aot.detection.types import DetectorInferenceError
from aot.detection.yolo import YoloDetector


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "media" / "gate2-three-frame.avi"


class SourceBackend:
    class_names = {0: "person", 1: "backpack", 2: "car"}

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def predict(self, _frame, **_policy):
        if self.fail:
            raise RuntimeError("isolated detector failure")
        return (
            (1, 2, 30, 40, 0.95, 0),
            (2, 3, 25, 35, 0.90, 1),
            (3, 4, 20, 30, 0.85, 2),
        )

    def close(self) -> None:
        return None


def _source(source_id: str) -> OfflineFrameSource:
    return OfflineFrameSource(
        CameraSourceConfig(
            source_id=source_id,
            name=source_id,
            source_type=SourceType.OFFLINE,
            enabled=True,
            offline_path=FIXTURE,
            width=320,
            height=240,
            fps=5.0,
        )
    )


def _detector(*, fail: bool = False) -> YoloDetector:
    return YoloDetector(
        "detector-test",
        DetectionConfig(
            enabled=True,
            model_id="detector-test",
            allowed_classes=("person", "backpack"),
        ),
        SourceBackend(fail=fail),
        "cpu",
    )


def test_three_source_detection_is_allowlisted_and_failure_isolated() -> None:
    sources = [_source(f"offline-{index}") for index in range(1, 4)]
    detectors = [_detector(), _detector(fail=True), _detector()]
    batches = {}
    failures = {}

    try:
        for source in sources:
            source.open()
        packets = [source.read() for source in sources]
        for packet, detector in zip(packets, detectors, strict=True):
            try:
                batches[packet.source_id] = detector.detect(packet)
            except DetectorInferenceError as error:
                failures[packet.source_id] = str(error)
    finally:
        for detector in detectors:
            detector.close()
        for source in sources:
            source.close()

    assert set(batches) == {"offline-1", "offline-3"}
    assert set(failures) == {"offline-2"}
    for source_id, batch in batches.items():
        assert batch.source_id == source_id
        assert [item.class_name for item in batch.detections] == ["person", "backpack"]
        assert batch.rejected_count == 1
