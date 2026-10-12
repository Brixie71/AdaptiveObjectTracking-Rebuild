"""Translate backend rows into validated project-owned detections."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from typing import AbstractSet

import numpy as np

from aot.capture.types import FramePacket

from .types import BoundingBox, Detection, DetectionBatch, DetectionError


def resolve_allowed_class_ids(
    class_names: Mapping[int, str],
    allowed_classes: tuple[str, ...],
) -> frozenset[int]:
    """Resolve human-readable allowed labels against one detector model."""

    normalized: dict[str, int] = {}
    for class_id, class_name in class_names.items():
        if isinstance(class_id, bool) or not isinstance(class_id, int):
            raise DetectionError("Detector class metadata is invalid")
        if not isinstance(class_name, str) or not class_name.strip():
            raise DetectionError("Detector class metadata is invalid")
        key = class_name.strip().casefold()
        if key in normalized:
            raise DetectionError("Detector class metadata is ambiguous")
        normalized[key] = class_id

    resolved: set[int] = set()
    for allowed in allowed_classes:
        key = allowed.strip().casefold()
        if key not in normalized:
            raise DetectionError(f"Unknown allowed class: {allowed.strip()}")
        resolved.add(normalized[key])
    return frozenset(resolved)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _class_id(value: object) -> int | None:
    number = _finite_number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _valid_frame(packet: FramePacket) -> bool:
    frame = packet.frame
    return (
        isinstance(frame, np.ndarray)
        and frame.dtype == np.uint8
        and frame.ndim == 3
        and frame.shape[0] > 0
        and frame.shape[1] > 0
        and frame.shape[2] == 3
        and frame.flags.c_contiguous
    )


def normalize_detections(
    packet: FramePacket,
    rows: Iterable[Sequence[object]],
    *,
    class_names: Mapping[int, str],
    allowed_class_ids: AbstractSet[int],
    max_detections: int,
) -> DetectionBatch:
    """Filter, clip, and validate plain backend rows for one frame."""

    if not _valid_frame(packet):
        raise DetectionError("Invalid frame packet")
    if (
        isinstance(max_detections, bool)
        or not isinstance(max_detections, int)
        or max_detections <= 0
    ):
        raise DetectionError("Invalid maximum detection count")

    height, width = packet.frame.shape[:2]
    detections: list[Detection] = []
    rejected_count = 0
    for row in rows:
        if len(detections) >= max_detections:
            break
        try:
            if len(row) != 6:
                raise ValueError
            x1, y1, x2, y2, confidence, raw_class_id = row
        except (TypeError, ValueError):
            rejected_count += 1
            continue

        coordinates = tuple(_finite_number(value) for value in (x1, y1, x2, y2))
        score = _finite_number(confidence)
        class_id = _class_id(raw_class_id)
        if (
            any(value is None for value in coordinates)
            or score is None
            or not 0.0 <= score <= 1.0
            or class_id is None
            or class_id not in class_names
        ):
            rejected_count += 1
            continue
        if class_id not in allowed_class_ids:
            rejected_count += 1
            continue

        raw_x1, raw_y1, raw_x2, raw_y2 = coordinates
        clipped_x1 = min(max(float(raw_x1), 0.0), float(width))
        clipped_y1 = min(max(float(raw_y1), 0.0), float(height))
        clipped_x2 = min(max(float(raw_x2), 0.0), float(width))
        clipped_y2 = min(max(float(raw_y2), 0.0), float(height))
        if clipped_x1 >= clipped_x2 or clipped_y1 >= clipped_y2:
            rejected_count += 1
            continue

        detections.append(
            Detection(
                class_id=class_id,
                class_name=class_names[class_id],
                confidence=score,
                box=BoundingBox(clipped_x1, clipped_y1, clipped_x2, clipped_y2),
            )
        )

    return DetectionBatch(
        source_id=packet.source_id,
        frame_sequence=packet.sequence,
        captured_at_monotonic_ns=packet.captured_at_monotonic_ns,
        frame_width=width,
        frame_height=height,
        detections=tuple(detections),
        rejected_count=rejected_count,
    )
