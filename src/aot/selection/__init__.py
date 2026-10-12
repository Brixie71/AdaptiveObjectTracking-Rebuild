"""Project-owned target-selection contracts."""

from aot.selection.geometry import clip_box, map_preview_point_to_frame
from aot.selection.target_selector import TargetSelectionError, TargetSelector
from aot.selection.types import PixelBox, SelectedTarget

__all__ = (
    "PixelBox",
    "SelectedTarget",
    "TargetSelectionError",
    "TargetSelector",
    "clip_box",
    "map_preview_point_to_frame",
)
