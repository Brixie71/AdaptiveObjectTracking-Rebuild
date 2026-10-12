"""Canonical frame normalization for all capture adapters."""

from __future__ import annotations

import cv2
import numpy as np


class FrameNormalizationError(ValueError):
    """A frame cannot be converted to the canonical BGR representation."""


_ROTATION_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def normalize_frame(
    frame: np.ndarray,
    *,
    width: int,
    height: int,
    rotation: int,
) -> np.ndarray:
    """Return a contiguous uint8 BGR frame at the requested dimensions."""
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise FrameNormalizationError("Frame dimensions must be positive integers")
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise FrameNormalizationError("Frame dimensions must be positive integers")
    if rotation not in {0, 90, 180, 270}:
        raise FrameNormalizationError("Frame rotation must be 0, 90, 180, or 270")
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise FrameNormalizationError("Frame must be a non-empty numpy array")
    if frame.dtype != np.uint8:
        raise FrameNormalizationError("Frame dtype must be uint8")

    try:
        if frame.ndim == 2:
            result = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            result = frame
        elif frame.ndim == 3 and frame.shape[2] == 4:
            result = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            raise FrameNormalizationError(
                "Frame must be grayscale, BGR, or BGRA"
            )

        if rotation:
            result = cv2.rotate(result, _ROTATION_CODES[rotation])
        if result.shape[:2] != (height, width):
            result = cv2.resize(result, (width, height))
    except cv2.error as error:
        raise FrameNormalizationError("OpenCV could not normalize the frame") from error

    return np.ascontiguousarray(result)
