from __future__ import annotations

import math

import numpy as np
import pytest

from aot.capture.types import FramePacket
from aot.detection.normalize import normalize_detections, resolve_allowed_class_ids
from aot.detection.types import DetectionError


def _packet(frame: np.ndarray | None = None) -> FramePacket:
    return FramePacket(
        source_id="camera-1",
        sequence=4,
        captured_at_monotonic_ns=999,
        frame=np.zeros((100, 200, 3), dtype=np.uint8) if frame is None else frame,
    )


def test_class_allowlist_resolves_case_insensitively_to_model_ids() -> None:
    resolved = resolve_allowed_class_ids(
        {0: "person", 1: "Backpack", 2: "bottle"},
        (" PERSON ", "backpack"),
    )

    assert resolved == frozenset({0, 1})


def test_class_allowlist_rejects_unknown_or_ambiguous_labels() -> None:
    with pytest.raises(DetectionError, match="Unknown allowed class"):
        resolve_allowed_class_ids({0: "person"}, ("car",))

    with pytest.raises(DetectionError, match="ambiguous"):
        resolve_allowed_class_ids({0: "Person", 1: " person "}, ("person",))


def test_normalization_filters_classes_clips_boxes_and_preserves_order() -> None:
    rows = [
        (-5.0, 10.0, 50.0, 120.0, 0.95, 0),
        (10.0, 20.0, 40.0, 60.0, 0.90, 2),
        (15.0, 25.0, 70.0, 80.0, 0.85, 1),
    ]

    batch = normalize_detections(
        _packet(),
        rows,
        class_names={0: "person", 1: "backpack", 2: "car"},
        allowed_class_ids=frozenset({0, 1}),
        max_detections=10,
    )

    assert batch.source_id == "camera-1"
    assert batch.frame_sequence == 4
    assert batch.captured_at_monotonic_ns == 999
    assert (batch.frame_width, batch.frame_height) == (200, 100)
    assert [item.class_name for item in batch.detections] == ["person", "backpack"]
    assert batch.detections[0].box.x1 == 0.0
    assert batch.detections[0].box.y2 == 100.0
    assert batch.rejected_count == 1


def test_normalization_stops_after_the_configured_accepted_limit() -> None:
    batch = normalize_detections(
        _packet(),
        [
            (1, 1, 10, 10, 0.9, 0),
            (2, 2, 20, 20, 0.8, 0),
            (3, 3, 30, 30, 0.7, 0),
        ],
        class_names={0: "person"},
        allowed_class_ids=frozenset({0}),
        max_detections=2,
    )

    assert [item.confidence for item in batch.detections] == [0.9, 0.8]


@pytest.mark.parametrize(
    "row",
    [
        (1, 2, 3),
        (1, 2, 3, 4, 0.5, 0, 7),
        (True, 2, 3, 4, 0.5, 0),
        (1, 2, math.nan, 4, 0.5, 0),
        (1, 2, 3, math.inf, 0.5, 0),
        (1, 2, 3, 4, math.nan, 0),
        (1, 2, 3, 4, -0.1, 0),
        (1, 2, 3, 4, 1.1, 0),
        (1, 2, 3, 4, 0.5, True),
        (1, 2, 3, 4, 0.5, -1),
        (1, 2, 3, 4, 0.5, 0.5),
        (1, 2, 3, 4, 0.5, 99),
        (20, 20, 10, 30, 0.5, 0),
        (-20, 10, -10, 30, 0.5, 0),
    ],
)
def test_malformed_backend_rows_are_counted_not_emitted(row: tuple[object, ...]) -> None:
    batch = normalize_detections(
        _packet(),
        [row],
        class_names={0: "person"},
        allowed_class_ids=frozenset({0}),
        max_detections=10,
    )

    assert batch.detections == ()
    assert batch.rejected_count == 1


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.zeros((0, 10, 3), dtype=np.uint8),
    ],
)
def test_invalid_frame_packets_are_rejected(frame: np.ndarray) -> None:
    with pytest.raises(DetectionError, match="Invalid frame packet"):
        normalize_detections(
            _packet(frame),
            [],
            class_names={0: "person"},
            allowed_class_ids=frozenset({0}),
            max_detections=10,
        )
