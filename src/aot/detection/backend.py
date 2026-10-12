"""Ultralytics isolation layer for the project-owned detector adapter."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import io
import logging
import threading
from pathlib import Path
from typing import Any, Protocol

import numpy as np


_OUTPUT_ISOLATION_LOCK = threading.RLock()


@contextmanager
def isolated_backend_output():
    """Contain synchronous third-party console and Ultralytics logger output."""

    with _OUTPUT_ISOLATION_LOCK:
        logger = logging.getLogger("ultralytics")
        previously_disabled = logger.disabled
        logger.disabled = True
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                yield
        finally:
            logger.disabled = previously_disabled


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


def _numeric_array(value: Any, field: str) -> np.ndarray:
    result = _plain_array(value)
    if not np.issubdtype(result.dtype, np.number) or np.issubdtype(
        result.dtype,
        np.bool_,
    ):
        raise ValueError(f"Detector backend returned non-numeric {field}")
    return result


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
        if not hasattr(results[0], "boxes") or results[0].boxes is None:
            raise ValueError("Detector backend result has no boxes")
        boxes = results[0].boxes
        xyxy = _numeric_array(getattr(boxes, "xyxy", None), "boxes")
        confidence = _numeric_array(getattr(boxes, "conf", None), "confidence")
        classes = _numeric_array(getattr(boxes, "cls", None), "class IDs")
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

    with isolated_backend_output():
        from ultralytics import YOLO

        return UltralyticsYoloBackend(YOLO(str(path)))
