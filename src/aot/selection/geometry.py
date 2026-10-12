"""Pure geometry shared by selection and native preview rendering."""

from __future__ import annotations

import math

from aot.detection.types import BoundingBox
from aot.selection.types import PixelBox


def _positive_finite(value: int | float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def clip_box(
    box: BoundingBox,
    frame_width: int,
    frame_height: int,
) -> PixelBox | None:
    """Clip a detection to half-open integer frame coordinates."""
    if not _positive_finite(frame_width) or not _positive_finite(frame_height):
        raise ValueError("Invalid frame dimensions")
    coordinates = (box.x1, box.y1, box.x2, box.y2)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in coordinates
    ):
        return None
    if box.x2 <= box.x1 or box.y2 <= box.y1:
        return None
    left = max(0, min(int(frame_width), math.floor(box.x1)))
    top = max(0, min(int(frame_height), math.floor(box.y1)))
    right = max(0, min(int(frame_width), math.ceil(box.x2)))
    bottom = max(0, min(int(frame_height), math.ceil(box.y2)))
    if right <= left or bottom <= top:
        return None
    return PixelBox(left, top, right, bottom)


def map_preview_point_to_frame(
    click_x: float,
    click_y: float,
    preview_width: float,
    preview_height: float,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float] | None:
    """Map a label-local point through a centered aspect-fit preview."""
    values = (
        click_x,
        click_y,
        preview_width,
        preview_height,
        frame_width,
        frame_height,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in values
    ) or not all(
        _positive_finite(value)
        for value in (preview_width, preview_height, frame_width, frame_height)
    ):
        raise ValueError("Invalid preview mapping values")

    if preview_width / frame_width <= preview_height / frame_height:
        rendered_width = int(preview_width)
        rendered_height = int(frame_height * preview_width / frame_width)
    else:
        rendered_height = int(preview_height)
        rendered_width = int(frame_width * preview_height / frame_height)
    offset_x = math.floor((preview_width - rendered_width) / 2.0)
    offset_y = math.floor((preview_height - rendered_height) / 2.0)
    local_x = click_x - offset_x
    local_y = click_y - offset_y
    if (
        local_x < 0
        or local_y < 0
        or local_x >= rendered_width
        or local_y >= rendered_height
    ):
        return None
    return (
        local_x * frame_width / rendered_width,
        local_y * frame_height / rendered_height,
    )
