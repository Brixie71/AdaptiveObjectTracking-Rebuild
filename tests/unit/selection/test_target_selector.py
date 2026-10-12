from __future__ import annotations

from datetime import datetime, timezone
import math

import numpy as np
import pytest

from aot.capture.types import FramePacket
from aot.detection.types import BoundingBox, Detection, DetectionBatch
from aot.runtime.types import ProcessedFrame
from aot.selection.target_selector import TargetSelectionError, TargetSelector
from aot.selection.types import PixelBox


SELECTED_AT = datetime(2026, 10, 12, 10, 0, tzinfo=timezone.utc)


def detection(
    name: str = "person",
    confidence: float = 0.9,
    box: BoundingBox = BoundingBox(1, 1, 5, 4),
) -> Detection:
    return Detection(0, name, confidence, box)


def processed(
    detections=(detection(),),
    *,
    source_id: str = "camera-a",
    sequence: int = 7,
    captured_at: int = 123,
    frame: np.ndarray | None = None,
) -> ProcessedFrame:
    if frame is None:
        frame = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    packet = FramePacket(source_id, sequence, captured_at, frame)
    batch = DetectionBatch(
        source_id,
        sequence,
        captured_at,
        frame.shape[1],
        frame.shape[0],
        tuple(detections),
        0,
    )
    return ProcessedFrame(packet, batch, None)


def selector(*ids: str) -> TargetSelector:
    identifiers = iter(ids or ("target-1",))
    return TargetSelector(id_factory=identifiers.__next__, clock=lambda: SELECTED_AT)


def test_select_creates_traceable_target_and_owned_crop() -> None:
    result = processed()
    expected = result.packet.frame[1:4, 1:5].copy()
    target_selector = selector()

    target = target_selector.select(result, 2, 2)
    result.packet.frame[:] = 0

    assert target is target_selector.current
    assert target.target_id == "target-1"
    assert target.source_id == "camera-a"
    assert target.frame_sequence == 7
    assert target.captured_at_monotonic_ns == 123
    assert target.selected_at_utc == SELECTED_AT
    assert target.class_name == "person"
    assert target.confidence == pytest.approx(0.9)
    assert target.initial_box == PixelBox(1, 1, 5, 4)
    assert np.array_equal(target.reference_crop, expected)


@pytest.mark.parametrize(
    ("point", "selected"),
    [((1, 1), True), ((4.999, 3.999), True), ((5, 2), False), ((2, 4), False)],
)
def test_hit_testing_uses_half_open_boundaries(point, selected) -> None:
    target_selector = selector()

    result = target_selector.select(processed(), *point)

    assert (result is not None) is selected


@pytest.mark.parametrize(
    ("detections", "expected_name"),
    [
        (
            (
                detection("lower", 0.7, BoundingBox(0, 0, 6, 6)),
                detection("higher", 0.9, BoundingBox(0, 0, 6, 6)),
            ),
            "higher",
        ),
        (
            (
                detection("large", 0.9, BoundingBox(0, 0, 7, 6)),
                detection("small", 0.9, BoundingBox(1, 1, 4, 4)),
            ),
            "small",
        ),
        (
            (
                detection("first", 0.9, BoundingBox(0, 0, 4, 4)),
                detection("second", 0.9, BoundingBox(0, 0, 4, 4)),
            ),
            "first",
        ),
    ],
)
def test_overlap_precedence_is_confidence_area_then_source_order(
    detections, expected_name
) -> None:
    target = selector().select(processed(detections), 2, 2)

    assert target.class_name == expected_name


def test_valid_selection_replaces_state_but_empty_click_preserves_it() -> None:
    target_selector = selector("target-1", "target-2")
    first = target_selector.select(processed(), 2, 2)
    assert target_selector.select(processed(), 7.5, 5.5) is None
    assert target_selector.current is first

    second = target_selector.select(
        processed((detection("backpack", 0.8, BoundingBox(0, 0, 3, 3)),)),
        1,
        1,
    )

    assert second.target_id == "target-2"
    assert second.class_name == "backpack"
    assert target_selector.current is second


def test_disengage_returns_previous_target_and_is_idempotent() -> None:
    target_selector = selector()
    target = target_selector.select(processed(), 2, 2)

    assert target_selector.disengage() is target
    assert target_selector.current is None
    assert target_selector.disengage() is None


def test_missing_detections_returns_none() -> None:
    packet = FramePacket("camera-a", 1, 2, np.zeros((2, 2, 3), dtype=np.uint8))
    assert selector().select(ProcessedFrame(packet, None, None), 1, 1) is None


@pytest.mark.parametrize("point", [(math.nan, 1), (1, math.inf)])
def test_non_finite_click_is_sanitized_and_preserves_state(point) -> None:
    target_selector = selector("target-1", "target-2")
    current = target_selector.select(processed(), 2, 2)

    with pytest.raises(TargetSelectionError, match="^Target selection unavailable$"):
        target_selector.select(processed(), *point)

    assert target_selector.current is current


def test_malformed_frame_is_sanitized_and_preserves_state() -> None:
    target_selector = selector("target-1", "target-2")
    current = target_selector.select(processed(), 2, 2)
    malformed = processed(frame=np.zeros((6, 8, 3), dtype=np.float32))

    with pytest.raises(TargetSelectionError, match="^Target selection unavailable$"):
        target_selector.select(malformed, 2, 2)

    assert target_selector.current is current


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_id", "camera-b"),
        ("frame_sequence", 999),
        ("captured_at_monotonic_ns", 999),
        ("frame_width", 999),
        ("frame_height", 999),
    ],
)
def test_mismatched_detection_provenance_is_sanitized_and_preserves_state(
    field, value
) -> None:
    target_selector = selector("target-1", "target-2")
    current = target_selector.select(processed(), 2, 2)
    mismatched = processed()
    object.__setattr__(mismatched.detections, field, value)

    with pytest.raises(TargetSelectionError, match="^Target selection unavailable$"):
        target_selector.select(mismatched, 2, 2)

    assert target_selector.current is current
