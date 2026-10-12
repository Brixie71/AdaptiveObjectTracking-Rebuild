"""Deterministic target selection and in-memory current-target ownership."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import math
from threading import Lock
import uuid

import numpy as np

from aot.runtime.types import ProcessedFrame
from aot.selection.geometry import clip_box
from aot.selection.types import PixelBox, SelectedTarget


class TargetSelectionError(RuntimeError):
    """Fixed safe error raised when displayed data cannot be selected."""


class TargetSelector:
    """Create and own one current in-memory target reference."""

    def __init__(
        self,
        *,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._id_factory = id_factory or (lambda: str(uuid.uuid4()))
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = Lock()
        self._current: SelectedTarget | None = None

    @property
    def current(self) -> SelectedTarget | None:
        with self._lock:
            return self._current

    def select(
        self,
        result: ProcessedFrame,
        frame_x: float,
        frame_y: float,
    ) -> SelectedTarget | None:
        try:
            target = self._build_target(result, frame_x, frame_y)
        except Exception as error:
            if isinstance(error, TargetSelectionError):
                raise
            raise TargetSelectionError("Target selection unavailable") from None
        if target is None:
            return None
        with self._lock:
            self._current = target
        return target

    def disengage(self) -> SelectedTarget | None:
        with self._lock:
            previous = self._current
            self._current = None
            return previous

    def _build_target(
        self,
        result: ProcessedFrame,
        frame_x: float,
        frame_y: float,
    ) -> SelectedTarget | None:
        if (
            isinstance(frame_x, bool)
            or isinstance(frame_y, bool)
            or not isinstance(frame_x, (int, float))
            or not isinstance(frame_y, (int, float))
            or not math.isfinite(float(frame_x))
            or not math.isfinite(float(frame_y))
        ):
            raise ValueError("Invalid selection point")
        if not isinstance(result, ProcessedFrame):
            raise TypeError("Invalid processed frame")
        batch = result.detections
        if batch is None:
            return None
        packet = result.packet
        frame = packet.frame
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.shape[0] < 1
            or frame.shape[1] < 1
        ):
            raise ValueError("Invalid frame")
        frame_height, frame_width = frame.shape[:2]
        if (
            batch.source_id != packet.source_id
            or batch.frame_sequence != packet.sequence
            or batch.captured_at_monotonic_ns != packet.captured_at_monotonic_ns
            or batch.frame_width != frame_width
            or batch.frame_height != frame_height
        ):
            raise ValueError("Detection provenance mismatch")

        candidates: list[tuple[float, int, int, PixelBox, object]] = []
        for index, detection in enumerate(batch.detections):
            confidence = float(detection.confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("Invalid detection confidence")
            box = clip_box(detection.box, frame_width, frame_height)
            if box is None:
                continue
            if (
                box.left <= frame_x < box.right
                and box.top <= frame_y < box.bottom
            ):
                candidates.append((-confidence, box.area, index, box, detection))
        if not candidates:
            return None
        _confidence, _area, _index, box, detection = min(candidates)
        crop = frame[box.top : box.bottom, box.left : box.right]
        return SelectedTarget(
            target_id=self._id_factory(),
            source_id=packet.source_id,
            frame_sequence=packet.sequence,
            captured_at_monotonic_ns=packet.captured_at_monotonic_ns,
            selected_at_utc=self._clock(),
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            initial_box=box,
            reference_crop=crop,
        )
