"""Native PyQt5 user-interface components."""

from aot.ui.camera_panel import CameraPanel
from aot.ui.background_task import FunctionTask
from aot.ui.controller import RuntimeController
from aot.ui.main_window import MainWindow
from aot.ui.rendering import (
    FrameRenderingError,
    OverlayDescription,
    describe_overlays,
    frame_to_qimage,
    render_processed_frame,
)

__all__ = [
    "CameraPanel",
    "FunctionTask",
    "FrameRenderingError",
    "MainWindow",
    "OverlayDescription",
    "RuntimeController",
    "describe_overlays",
    "frame_to_qimage",
    "render_processed_frame",
]
