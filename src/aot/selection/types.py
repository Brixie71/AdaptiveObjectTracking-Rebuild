"""Immutable records produced by target selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math

import numpy as np


@dataclass(frozen=True)
class PixelBox:
    """Validated half-open pixel coordinates."""

    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if (
            any(not isinstance(value, int) or isinstance(value, bool) for value in values)
            or self.left < 0
            or self.top < 0
            or self.right <= self.left
            or self.bottom <= self.top
        ):
            raise ValueError("Invalid pixel box")

    @property
    def area(self) -> int:
        return (self.right - self.left) * (self.bottom - self.top)


@dataclass(frozen=True, eq=False)
class SelectedTarget:
    """One in-memory target reference copied from an exact processed frame."""

    target_id: str
    source_id: str
    frame_sequence: int
    captured_at_monotonic_ns: int
    selected_at_utc: datetime
    class_id: int
    class_name: str
    confidence: float
    initial_box: PixelBox
    reference_crop: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.target_id, str) or not self.target_id.strip():
            raise ValueError("Invalid target ID")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("Invalid source ID")
        if (
            not isinstance(self.frame_sequence, int)
            or isinstance(self.frame_sequence, bool)
            or self.frame_sequence < 0
        ):
            raise ValueError("Invalid frame sequence")
        if (
            not isinstance(self.captured_at_monotonic_ns, int)
            or isinstance(self.captured_at_monotonic_ns, bool)
            or self.captured_at_monotonic_ns < 0
        ):
            raise ValueError("Invalid capture timestamp")
        selected_at = self.selected_at_utc
        if (
            not isinstance(selected_at, datetime)
            or selected_at.tzinfo is None
            or selected_at.utcoffset() != timedelta(0)
        ):
            raise ValueError("Selection time must be UTC")
        if not isinstance(self.class_id, int) or isinstance(self.class_id, bool):
            raise ValueError("Invalid class ID")
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("Invalid class name")
        if (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("Invalid confidence")
        if not isinstance(self.initial_box, PixelBox):
            raise ValueError("Invalid pixel box")

        crop = self.reference_crop
        if (
            not isinstance(crop, np.ndarray)
            or crop.dtype != np.uint8
            or crop.ndim != 3
            or crop.shape[2] != 3
            or crop.shape[0] < 1
            or crop.shape[1] < 1
        ):
            raise ValueError("Invalid target crop")
        if crop.shape[:2] != (
            self.initial_box.bottom - self.initial_box.top,
            self.initial_box.right - self.initial_box.left,
        ):
            raise ValueError("Target crop dimensions do not match pixel box")
        owned = np.array(crop, dtype=np.uint8, order="C", copy=True)
        owned.setflags(write=False)
        object.__setattr__(self, "reference_crop", owned)
