"""Project-owned YOLO detector adapter."""

from __future__ import annotations

from aot.capture.types import FramePacket
from aot.config.types import DetectionConfig

from .backend import YoloBackend, isolated_backend_output
from .normalize import normalize_detections, resolve_allowed_class_ids
from .types import (
    DetectionBatch,
    DetectionError,
    DetectorInferenceError,
    DetectorLoadError,
)


class YoloDetector:
    """Apply one configured allowlist to one verified YOLO backend."""

    def __init__(
        self,
        model_id: str,
        config: DetectionConfig,
        backend: YoloBackend,
        device: str,
    ) -> None:
        try:
            allowed_class_ids = resolve_allowed_class_ids(
                backend.class_names,
                config.allowed_classes,
            )
        except Exception:
            raise DetectorLoadError(
                "Detector allowlist is incompatible with the model"
            ) from None
        self.model_id = model_id
        self.class_names = tuple(
            name for _, name in sorted(backend.class_names.items(), key=lambda item: item[0])
        )
        self.device = device
        self._config = config
        self._backend = backend
        self._allowed_class_ids = allowed_class_ids
        self._closed = False

    def detect(self, packet: FramePacket) -> DetectionBatch:
        if self._closed:
            raise DetectorInferenceError("Detector is closed")
        try:
            with isolated_backend_output():
                rows = self._backend.predict(
                    packet.frame,
                    confidence_threshold=self._config.confidence_threshold,
                    iou_threshold=self._config.iou_threshold,
                    max_detections=self._config.max_detections,
                    device=self.device,
                )
                return normalize_detections(
                    packet,
                    rows,
                    class_names=self._backend.class_names,
                    allowed_class_ids=self._allowed_class_ids,
                    max_detections=self._config.max_detections,
                )
        except Exception:
            raise DetectorInferenceError("Detector inference failed") from None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            with isolated_backend_output():
                self._backend.close()
        except Exception:
            raise DetectionError("Detector cleanup failed") from None
