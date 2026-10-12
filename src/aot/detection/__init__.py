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

__all__ = [
    "BoundingBox",
    "Detection",
    "DetectionBatch",
    "DetectionError",
    "Detector",
    "DetectorInferenceError",
    "DetectorLoadError",
]
