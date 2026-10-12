from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from aot.selection.types import PixelBox, SelectedTarget


def test_pixel_box_is_half_open_and_reports_area() -> None:
    box = PixelBox(2, 3, 7, 11)

    assert box.area == 40
    assert (box.right - box.left, box.bottom - box.top) == (5, 8)


@pytest.mark.parametrize(
    "coordinates",
    [(-1, 0, 1, 1), (0, -1, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1)],
)
def test_pixel_box_rejects_negative_or_empty_geometry(coordinates) -> None:
    with pytest.raises(ValueError, match="pixel box"):
        PixelBox(*coordinates)


def test_selected_target_copies_crop_and_preserves_traceability() -> None:
    source = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    crop = source[1:3, 1:4]
    selected_at = datetime(2026, 10, 12, 9, 30, tzinfo=timezone.utc)
    target = SelectedTarget(
        "target-1",
        "camera-a",
        17,
        123456,
        selected_at,
        0,
        "person",
        0.91,
        PixelBox(1, 1, 4, 3),
        crop,
    )
    expected = crop.copy()
    source[:] = 0

    assert target.target_id == "target-1"
    assert target.source_id == "camera-a"
    assert target.frame_sequence == 17
    assert target.captured_at_monotonic_ns == 123456
    assert target.selected_at_utc == selected_at
    assert target.class_id == 0
    assert target.class_name == "person"
    assert target.confidence == pytest.approx(0.91)
    assert np.array_equal(target.reference_crop, expected)
    assert target.reference_crop.flags.c_contiguous
    assert target.reference_crop.flags.owndata
    assert not target.reference_crop.flags.writeable
    with pytest.raises(ValueError):
        target.reference_crop[0, 0] = 1


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"target_id": ""}, "target ID"),
        ({"source_id": ""}, "source ID"),
        ({"frame_sequence": -1}, "frame sequence"),
        ({"captured_at_monotonic_ns": -1}, "capture timestamp"),
        ({"selected_at_utc": datetime(2026, 1, 1)}, "UTC"),
        ({"class_name": ""}, "class name"),
        ({"confidence": float("nan")}, "confidence"),
        ({"reference_crop": np.zeros((2, 3, 3), dtype=np.float32)}, "crop"),
        ({"reference_crop": np.zeros((1, 3, 3), dtype=np.uint8)}, "dimensions"),
    ],
)
def test_selected_target_rejects_invalid_metadata(updates, message) -> None:
    values = {
        "target_id": "target-1",
        "source_id": "camera-a",
        "frame_sequence": 1,
        "captured_at_monotonic_ns": 2,
        "selected_at_utc": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "class_id": 0,
        "class_name": "person",
        "confidence": 0.5,
        "initial_box": PixelBox(0, 0, 3, 2),
        "reference_crop": np.zeros((2, 3, 3), dtype=np.uint8),
    }
    values.update(updates)

    with pytest.raises(ValueError, match=message):
        SelectedTarget(**values)
