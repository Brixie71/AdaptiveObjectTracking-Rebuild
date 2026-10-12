"""Native PyQt5 user-interface components."""

from aot.ui.camera_panel import CameraPanel
from aot.ui.rendering import (
    FrameRenderingError,
    OverlayDescription,
    describe_overlays,
    frame_to_qimage,
    render_processed_frame,
)

__all__ = [
    "CameraPanel",
    "FrameRenderingError",
    "OverlayDescription",
    "describe_overlays",
    "frame_to_qimage",
    "render_processed_frame",
]
