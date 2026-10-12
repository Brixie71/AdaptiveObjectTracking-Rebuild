"""Ultralytics isolation layer for the project-owned detector adapter."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class YoloBackend(Protocol):
    class_names: Mapping[int, str]

    def predict(
        self,
        frame: np.ndarray,
        *,
        confidence_threshold: float,
        iou_threshold: float,
        max_detections: int,
        device: str,
    ) -> Iterable[Sequence[object]]:
        """Return plain xyxy/confidence/class rows for one frame."""

    def close(self) -> None:
        """Release backend resources."""


def _plain_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class UltralyticsYoloBackend:
    """Keep all Ultralytics result objects behind one narrow boundary."""

    def __init__(self, model: Any) -> None:
        names = getattr(model, "names", None)
        if not isinstance(names, Mapping):
            raise ValueError("Detector model has invalid class metadata")
        self.class_names = dict(names)
        self._model = model

    def predict(
        self,
        frame: np.ndarray,
        *,
        confidence_threshold: float,
        iou_threshold: float,
        max_detections: int,
        device: str,
    ) -> tuple[tuple[object, ...], ...]:
        if self._model is None:
            raise ValueError("Detector backend is closed")
        results = self._model.predict(
            source=frame,
            conf=confidence_threshold,
            iou=iou_threshold,
            max_det=max_detections,
            device=device,
            verbose=False,
        )
        if not isinstance(results, Sequence) or len(results) != 1:
            raise ValueError("Detector backend must return exactly one result")
        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return ()
        xyxy = _plain_array(getattr(boxes, "xyxy", None))
        confidence = _plain_array(getattr(boxes, "conf", None))
        classes = _plain_array(getattr(boxes, "cls", None))
        if xyxy.ndim != 2 or xyxy.shape[1:] != (4,):
            raise ValueError("Detector backend returned malformed boxes")
        if confidence.ndim != 1 or classes.ndim != 1:
            raise ValueError("Detector backend returned malformed scores")
        if not (len(xyxy) == len(confidence) == len(classes)):
            raise ValueError("Detector backend returned inconsistent row counts")
        return tuple(
            (
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
                float(score),
                float(class_id),
            )
            for box, score, class_id in zip(xyxy, confidence, classes, strict=True)
        )

    def close(self) -> None:
        self._model = None


def open_ultralytics_backend(path: Path) -> YoloBackend:
    """Load one verified artifact through a lazy Ultralytics import."""

    from ultralytics import YOLO

    return UltralyticsYoloBackend(YOLO(str(path)))
