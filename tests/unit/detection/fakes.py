from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np


class FakeYoloBackend:
    def __init__(
        self,
        *,
        class_names: Mapping[int, str] | None = None,
        rows: Iterable[Sequence[object]] = (),
        predict_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.class_names = dict(class_names or {0: "person", 1: "backpack", 2: "car"})
        self.rows = tuple(tuple(row) for row in rows)
        self.predict_error = predict_error
        self.close_error = close_error
        self.calls: list[dict[str, object]] = []
        self.close_count = 0

    def predict(
        self,
        frame: np.ndarray,
        *,
        confidence_threshold: float,
        iou_threshold: float,
        max_detections: int,
        device: str,
    ) -> tuple[tuple[object, ...], ...]:
        self.calls.append(
            {
                "frame": frame,
                "confidence_threshold": confidence_threshold,
                "iou_threshold": iou_threshold,
                "max_detections": max_detections,
                "device": device,
            }
        )
        if self.predict_error is not None:
            raise self.predict_error
        return self.rows

    def close(self) -> None:
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error
