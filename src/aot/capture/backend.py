"""Narrow backend boundary around OpenCV video capture."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Protocol

import cv2
import numpy as np


class BackendProperty(str, Enum):
    WIDTH = "width"
    HEIGHT = "height"
    FPS = "fps"


class CaptureBackend(Protocol):
    def is_opened(self) -> bool: ...

    def set(self, property_name: BackendProperty, value: float) -> bool: ...

    def get(self, property_name: BackendProperty) -> float: ...

    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


_OPENCV_PROPERTIES = {
    BackendProperty.WIDTH: cv2.CAP_PROP_FRAME_WIDTH,
    BackendProperty.HEIGHT: cv2.CAP_PROP_FRAME_HEIGHT,
    BackendProperty.FPS: cv2.CAP_PROP_FPS,
}


class OpenCvCaptureBackend:
    """Project-owned wrapper that contains all VideoCapture constants."""

    def __init__(self, source: int | str) -> None:
        self._capture = cv2.VideoCapture(source)

    def is_opened(self) -> bool:
        return self._capture.isOpened()

    def set(self, property_name: BackendProperty, value: float) -> bool:
        return self._capture.set(_OPENCV_PROPERTIES[property_name], value)

    def get(self, property_name: BackendProperty) -> float:
        return self._capture.get(_OPENCV_PROPERTIES[property_name])

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self._capture.read()

    def release(self) -> None:
        self._capture.release()


def open_cv_backend(source: int | str) -> CaptureBackend:
    return OpenCvCaptureBackend(source)


BackendFactory = Callable[[int | str], CaptureBackend]
