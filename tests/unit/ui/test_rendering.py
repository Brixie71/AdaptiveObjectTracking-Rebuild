from __future__ import annotations

import gc

import numpy as np
import pytest

from aot.capture.types import FramePacket
from aot.detection.types import BoundingBox, Detection, DetectionBatch
from aot.runtime.types import ProcessedFrame
from aot.ui.rendering import (
    FrameRenderingError,
    describe_overlays,
    frame_to_qimage,
    render_processed_frame,
)


def result_with_detection() -> ProcessedFrame:
    packet = FramePacket("camera-1", 4, 10, np.zeros((10, 20, 3), dtype=np.uint8))
    detection = Detection(0, "person", 0.876, BoundingBox(-4, 2, 25, 15))
    batch = DetectionBatch("camera-1", 4, 10, 20, 10, (detection,), 0)
    return ProcessedFrame(packet, batch, None)


def test_bgr_pixels_convert_to_rgb_and_own_their_memory() -> None:
    frame = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    image = frame_to_qimage(frame)
    frame[:] = 0
    del frame
    gc.collect()

    assert image.pixelColor(0, 0).getRgb()[:3] == (0, 0, 255)
    assert image.pixelColor(1, 0).getRgb()[:3] == (0, 255, 0)
    assert image.pixelColor(2, 0).getRgb()[:3] == (255, 0, 0)


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
        np.zeros((0, 2, 3), dtype=np.uint8),
    ],
)
def test_malformed_frames_raise_ui_owned_error(frame: np.ndarray) -> None:
    with pytest.raises(FrameRenderingError, match="displayable"):
        frame_to_qimage(frame)


def test_overlay_description_clips_boxes_and_formats_label(qapp) -> None:
    descriptions = describe_overlays(result_with_detection())

    assert len(descriptions) == 1
    assert descriptions[0].box == (0, 2, 19, 9)
    assert descriptions[0].label == "person 88%"
    assert not render_processed_frame(result_with_detection()).isNull()


def test_render_rejects_detection_provenance_changed_after_construction() -> None:
    result = result_with_detection()
    object.__setattr__(result.detections, "frame_sequence", 99)

    with pytest.raises(FrameRenderingError, match="provenance"):
        render_processed_frame(result)
