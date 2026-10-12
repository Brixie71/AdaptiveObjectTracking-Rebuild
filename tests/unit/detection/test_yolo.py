from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from aot.capture.types import FramePacket
from aot.config.types import DetectionConfig
from aot.detection.backend import UltralyticsYoloBackend
from aot.detection.types import DetectionError, DetectorInferenceError, DetectorLoadError
from aot.detection.yolo import YoloDetector

from .fakes import FakeYoloBackend


def _packet() -> FramePacket:
    return FramePacket(
        source_id="camera-1",
        sequence=3,
        captured_at_monotonic_ns=123,
        frame=np.zeros((80, 120, 3), dtype=np.uint8),
    )


def _config(*classes: str) -> DetectionConfig:
    return DetectionConfig(
        enabled=True,
        model_id="yolov8n-coco",
        confidence_threshold=0.4,
        iou_threshold=0.6,
        max_detections=25,
        allowed_classes=classes or ("person", "backpack"),
    )


def test_yolo_detector_propagates_policy_and_returns_owned_results() -> None:
    backend = FakeYoloBackend(
        rows=(
            (1, 2, 30, 40, 0.9, 0),
            (2, 3, 20, 30, 0.8, 2),
            (3, 4, 25, 35, 0.7, 1),
        )
    )
    detector = YoloDetector("yolov8n-coco", _config(), backend, "cuda:1")
    packet = _packet()

    batch = detector.detect(packet)

    assert [item.class_name for item in batch.detections] == ["person", "backpack"]
    assert batch.rejected_count == 1
    assert detector.class_names == ("person", "backpack", "car")
    assert backend.calls == [
        {
            "frame": packet.frame,
            "confidence_threshold": 0.4,
            "iou_threshold": 0.6,
            "max_detections": 25,
            "device": "cuda:1",
        }
    ]


def test_yolo_detector_rejects_unknown_allowlisted_class_at_load() -> None:
    with pytest.raises(DetectorLoadError, match="allowlist"):
        YoloDetector("yolov8n-coco", _config("unknown"), FakeYoloBackend(), "cpu")


def test_yolo_detector_translates_sensitive_inference_and_cleanup_errors() -> None:
    secret = "rtsp://admin:camera-pass@example.invalid/live"
    detector = YoloDetector(
        "yolov8n-coco",
        _config(),
        FakeYoloBackend(predict_error=RuntimeError(secret)),
        "cpu",
    )

    with pytest.raises(DetectorInferenceError) as inference_error:
        detector.detect(_packet())

    assert secret not in str(inference_error.value)

    cleanup_backend = FakeYoloBackend(close_error=RuntimeError(secret))
    cleanup_detector = YoloDetector("yolov8n-coco", _config(), cleanup_backend, "cpu")
    with pytest.raises(DetectionError) as cleanup_error:
        cleanup_detector.close()
    cleanup_detector.close()

    assert secret not in str(cleanup_error.value)
    assert cleanup_backend.close_count == 1


def test_yolo_detector_rejects_detection_after_close() -> None:
    detector = YoloDetector("yolov8n-coco", _config(), FakeYoloBackend(), "cpu")
    detector.close()

    with pytest.raises(DetectorInferenceError, match="closed"):
        detector.detect(_packet())


class _Tensor:
    def __init__(self, values) -> None:
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.values)


class _Model:
    names = {0: "person", 1: "backpack"}

    def __init__(self, results) -> None:
        self.results = results
        self.arguments = None

    def predict(self, **arguments):
        self.arguments = arguments
        return self.results


def test_ultralytics_backend_converts_one_result_to_plain_rows() -> None:
    boxes = SimpleNamespace(
        xyxy=_Tensor([[1.0, 2.0, 30.0, 40.0]]),
        conf=_Tensor([0.9]),
        cls=_Tensor([0.0]),
    )
    model = _Model([SimpleNamespace(boxes=boxes)])
    backend = UltralyticsYoloBackend(model)
    frame = _packet().frame

    rows = backend.predict(
        frame,
        confidence_threshold=0.4,
        iou_threshold=0.6,
        max_detections=25,
        device="cpu",
    )

    assert rows == ((1.0, 2.0, 30.0, 40.0, 0.9, 0.0),)
    assert model.arguments == {
        "source": frame,
        "conf": 0.4,
        "iou": 0.6,
        "max_det": 25,
        "device": "cpu",
        "verbose": False,
    }


@pytest.mark.parametrize("results", [[], [SimpleNamespace(boxes=None), SimpleNamespace(boxes=None)]])
def test_ultralytics_backend_requires_exactly_one_result(results) -> None:
    backend = UltralyticsYoloBackend(_Model(results))

    with pytest.raises(ValueError, match="one result"):
        backend.predict(
            _packet().frame,
            confidence_threshold=0.25,
            iou_threshold=0.7,
            max_detections=300,
            device="cpu",
        )
