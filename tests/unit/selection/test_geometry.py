from __future__ import annotations

import math

import pytest

from aot.detection.types import BoundingBox
from aot.selection.geometry import clip_box, map_preview_point_to_frame
from aot.selection.types import PixelBox


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        (BoundingBox(-1.2, 2.2, 12.1, 8.01), PixelBox(0, 2, 10, 9)),
        (BoundingBox(1.9, 0.1, 4.01, 3.2), PixelBox(1, 0, 5, 4)),
        (BoundingBox(9.2, 8.2, 14.0, 15.0), PixelBox(9, 8, 10, 10)),
    ],
)
def test_clip_box_uses_half_open_floor_ceil_geometry(box, expected) -> None:
    assert clip_box(box, 10, 10) == expected


@pytest.mark.parametrize(
    "box",
    [
        BoundingBox(2, 2, 2, 4),
        BoundingBox(4, 2, 2, 4),
        BoundingBox(20, 20, 30, 30),
        BoundingBox(math.nan, 0, 1, 1),
        BoundingBox(0, math.inf, 1, 2),
    ],
)
def test_clip_box_rejects_empty_inverted_outside_or_non_finite_boxes(box) -> None:
    assert clip_box(box, 10, 10) is None


@pytest.mark.parametrize("dimensions", [(0, 10), (10, 0), (-1, 10)])
def test_clip_box_rejects_invalid_frame_dimensions(dimensions) -> None:
    with pytest.raises(ValueError, match="frame dimensions"):
        clip_box(BoundingBox(0, 0, 1, 1), *dimensions)


def test_preview_mapping_handles_exact_fit_and_exclusive_edges() -> None:
    assert map_preview_point_to_frame(50, 25, 100, 50, 200, 100) == (100.0, 50.0)
    assert map_preview_point_to_frame(0, 0, 100, 50, 200, 100) == (0.0, 0.0)
    assert map_preview_point_to_frame(100, 25, 100, 50, 200, 100) is None
    assert map_preview_point_to_frame(50, 50, 100, 50, 200, 100) is None


def test_preview_mapping_rejects_letterbox_and_pillarbox_margins() -> None:
    # 2:1 frame inside a square preview: 25 px top/bottom letterbox.
    assert map_preview_point_to_frame(50, 24.9, 100, 100, 200, 100) is None
    assert map_preview_point_to_frame(50, 25, 100, 100, 200, 100) == (100.0, 0.0)
    assert map_preview_point_to_frame(50, 75, 100, 100, 200, 100) is None
    # 1:2 frame inside a square preview: 25 px left/right pillarbox.
    assert map_preview_point_to_frame(24.9, 50, 100, 100, 100, 200) is None
    assert map_preview_point_to_frame(25, 50, 100, 100, 100, 200) == (0.0, 100.0)
    assert map_preview_point_to_frame(75, 50, 100, 100, 100, 200) is None


@pytest.mark.parametrize(
    "values",
    [
        (math.nan, 0, 10, 10, 10, 10),
        (0, math.inf, 10, 10, 10, 10),
        (0, 0, 0, 10, 10, 10),
        (0, 0, 10, 10, 0, 10),
    ],
)
def test_preview_mapping_rejects_invalid_values(values) -> None:
    with pytest.raises(ValueError, match="preview mapping"):
        map_preview_point_to_frame(*values)
