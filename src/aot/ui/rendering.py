"""Pure display conversion and detection-overlay helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen

from aot.runtime.types import ProcessedFrame
from aot.selection.geometry import clip_box


class FrameRenderingError(ValueError):
    """Raised when application-owned frame data cannot be displayed safely."""


@dataclass(frozen=True)
class OverlayDescription:
    box: tuple[int, int, int, int]
    label: str


def frame_to_qimage(frame: np.ndarray) -> QImage:
    """Convert one uint8 BGR frame into an independently owned RGB image."""
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.shape[0] < 1
        or frame.shape[1] < 1
    ):
        raise FrameRenderingError("Frame is not displayable")
    rgb = np.ascontiguousarray(frame[:, :, ::-1])
    height, width, _channels = rgb.shape
    return QImage(
        rgb.data,
        width,
        height,
        rgb.strides[0],
        QImage.Format_RGB888,
    ).copy()


def _validate_provenance(result: ProcessedFrame) -> None:
    batch = result.detections
    if batch is None:
        return
    if (
        batch.source_id != result.packet.source_id
        or batch.frame_sequence != result.packet.sequence
    ):
        raise FrameRenderingError("Detection provenance does not match displayed frame")


def describe_overlays(result: ProcessedFrame) -> tuple[OverlayDescription, ...]:
    """Return clipped, deterministic overlay geometry for tests and painting."""
    _validate_provenance(result)
    batch = result.detections
    if batch is None:
        return ()
    frame_height, frame_width = result.packet.frame.shape[:2]
    descriptions: list[OverlayDescription] = []
    for detection in batch.detections:
        box = clip_box(detection.box, frame_width, frame_height)
        if box is None:
            continue
        descriptions.append(
            OverlayDescription(
                (box.left, box.top, box.right, box.bottom),
                f"{detection.class_name} {detection.confidence:.0%}",
            )
        )
    return tuple(descriptions)


def render_processed_frame(result: ProcessedFrame) -> QImage:
    """Render project-owned overlays onto a UI-owned image copy."""
    image = frame_to_qimage(result.packet.frame)
    overlays = describe_overlays(result)
    if not overlays:
        return image
    painter = QPainter(image)
    painter.setRenderHint(QPainter.Antialiasing, False)
    painter.setPen(QPen(QColor(0, 220, 120), 2))
    painter.setFont(QFont("Segoe UI", 9))
    for overlay in overlays:
        x1, y1, x2, y2 = overlay.box
        painter.drawRect(QRect(x1, y1, x2 - x1, y2 - y1))
        painter.drawText(x1 + 2, max(10, y1 - 3), overlay.label)
    painter.end()
    return image
