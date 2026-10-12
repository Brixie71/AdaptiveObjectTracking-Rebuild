"""Project-owned object-detection contracts and adapters."""

from .types import (
    BoundingBox,
    Detection,
    DetectionBatch,
    DetectionError,
    Detector,
    DetectorInferenceError,
    DetectorLoadError,
)
from .factory import create_detector

__all__ = [
    "BoundingBox",
    "Detection",
    "DetectionBatch",
    "DetectionError",
    "Detector",
    "DetectorInferenceError",
    "DetectorLoadError",
    "create_detector",
]
