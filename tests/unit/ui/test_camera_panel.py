from __future__ import annotations

import numpy as np
import pytest
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtTest import QSignalSpy

from aot.capture.types import FramePacket, SourceState
from aot.capture.worker import WorkerSnapshot
from aot.config.types import SourceType
from aot.detection.types import BoundingBox, Detection, DetectionBatch
from aot.runtime.types import ProcessedFrame, SourceRuntimeSnapshot
from aot.ui.camera_panel import CameraPanel


def snapshot(
    state: SourceState,
    *,
    sequence: int | None = 1,
    last_error: str | None = None,
    processing_error: str | None = None,
) -> SourceRuntimeSnapshot:
    result = None
    if sequence is not None:
        packet = FramePacket(
            "camera-1",
            sequence,
            sequence,
            np.full((40, 80, 3), sequence, dtype=np.uint8),
        )
        result = ProcessedFrame(packet, None, processing_error)
    return SourceRuntimeSnapshot(
        "camera-1",
        "Front Door",
        SourceType.USB,
        WorkerSnapshot("camera-1", state, 0, 0, last_error, state is SourceState.OPEN),
        result,
    )


def selectable_snapshot(sequence: int = 1, *, malformed: bool = False):
    frame = np.zeros((100, 200, 3), dtype=np.float32 if malformed else np.uint8)
    packet = FramePacket("camera-1", sequence, sequence, frame)
    batch = DetectionBatch(
        "camera-1",
        sequence,
        sequence,
        200,
        100,
        (Detection(0, "person", 0.9, BoundingBox(20, 20, 180, 80)),),
        0,
    )
    result = ProcessedFrame(packet, batch, None)
    return SourceRuntimeSnapshot(
        "camera-1",
        "Front Door",
        SourceType.USB,
        WorkerSnapshot("camera-1", SourceState.OPEN, 0, 0, None, True),
        result,
    )


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (SourceState.OPEN, "Live"),
        (SourceState.RECONNECTING, "Reconnecting"),
        (SourceState.EOF, "End of stream"),
        (SourceState.FAILED, "Unavailable"),
    ],
)
def test_panel_maps_source_states_to_stable_safe_text(qtbot, state, expected) -> None:
    panel = CameraPanel()
    qtbot.addWidget(panel)
    panel.apply_snapshot(snapshot(state, last_error="PRIVATE-CREDENTIAL"))

    assert panel.name_label.text() == "Front Door"
    assert panel.source_label.text() == "camera-1 · USB"
    assert expected in panel.status_label.text()
    assert "PRIVATE-CREDENTIAL" not in panel.status_label.text()
    assert panel.status_label.property("state") == state.value


def test_unconfigured_panel_has_explicit_empty_state(qtbot) -> None:
    panel = CameraPanel()
    qtbot.addWidget(panel)
    panel.apply_snapshot(None)

    assert panel.name_label.text() == "Unconfigured camera"
    assert panel.source_label.text() == "No source assigned"
    assert panel.status_label.text() == "Unconfigured"
    assert panel.image_label.pixmap() is None


def test_panel_replaces_pixmap_and_preserves_aspect_ratio(qtbot) -> None:
    panel = CameraPanel()
    panel.resize(320, 240)
    qtbot.addWidget(panel)
    panel.show()
    panel.apply_snapshot(snapshot(SourceState.OPEN, sequence=1))
    first_key = panel.image_label.pixmap().cacheKey()
    panel.apply_snapshot(snapshot(SourceState.OPEN, sequence=2))
    second = panel.image_label.pixmap()

    assert second.cacheKey() != first_key
    assert second.width() / second.height() == pytest.approx(2.0, rel=0.05)
    assert panel.image_label.alignment() == Qt.AlignCenter
    assert len(panel.findChildren(type(panel.image_label))) == 4


def test_processing_failure_uses_fixed_summary(qtbot) -> None:
    panel = CameraPanel()
    qtbot.addWidget(panel)
    panel.apply_snapshot(
        snapshot(
            SourceState.OPEN,
            processing_error="PRIVATE-MODEL-PATH",
        )
    )

    assert panel.status_label.text() == "Live · Detection unavailable"
    assert "PRIVATE" not in panel.status_label.text()


def test_click_maps_aspect_fit_point_and_emits_exact_displayed_frame(qtbot) -> None:
    panel = CameraPanel()
    qtbot.addWidget(panel)
    panel.image_label.setFixedSize(200, 200)
    current = selectable_snapshot()
    panel.apply_snapshot(current)
    panel.set_selection_enabled(True)
    selected = QSignalSpy(panel.selection_requested)

    qtbot.mouseClick(panel.image_label, Qt.LeftButton, pos=QPoint(100, 100))

    assert len(selected) == 1
    assert selected[0][0].packet.source_id == current.result.packet.source_id
    assert np.array_equal(selected[0][0].packet.frame, current.result.packet.frame)
    assert selected[0][1:] == [100.0, 50.0]


def test_letterbox_margin_click_does_not_emit(qtbot) -> None:
    panel = CameraPanel()
    qtbot.addWidget(panel)
    panel.image_label.setFixedSize(200, 200)
    panel.apply_snapshot(selectable_snapshot())
    panel.set_selection_enabled(True)
    selected = QSignalSpy(panel.selection_requested)

    qtbot.mouseClick(panel.image_label, Qt.LeftButton, pos=QPoint(100, 25))

    assert len(selected) == 0


def test_new_snapshot_replaces_retained_selectable_frame(qtbot) -> None:
    panel = CameraPanel()
    qtbot.addWidget(panel)
    panel.image_label.setFixedSize(200, 200)
    first = selectable_snapshot(1)
    second = selectable_snapshot(2)
    panel.apply_snapshot(first)
    panel.apply_snapshot(second)
    panel.set_selection_enabled(True)
    selected = QSignalSpy(panel.selection_requested)

    qtbot.mouseClick(panel.image_label, Qt.LeftButton, pos=QPoint(100, 100))

    assert len(selected) == 1
    assert selected[0][0].packet.sequence == second.result.packet.sequence


@pytest.mark.parametrize("snapshot_value", [None, snapshot(SourceState.OPEN, sequence=None)])
def test_unconfigured_or_missing_result_cannot_emit(qtbot, snapshot_value) -> None:
    panel = CameraPanel()
    qtbot.addWidget(panel)
    panel.image_label.setFixedSize(200, 200)
    panel.apply_snapshot(selectable_snapshot())
    panel.apply_snapshot(snapshot_value)
    panel.set_selection_enabled(True)
    selected = QSignalSpy(panel.selection_requested)

    qtbot.mouseClick(panel.image_label, Qt.LeftButton, pos=QPoint(100, 100))

    assert len(selected) == 0


def test_malformed_preview_and_disabled_selection_cannot_emit(qtbot) -> None:
    panel = CameraPanel()
    qtbot.addWidget(panel)
    panel.image_label.setFixedSize(200, 200)
    selected = QSignalSpy(panel.selection_requested)
    panel.apply_snapshot(selectable_snapshot(malformed=True))
    panel.set_selection_enabled(True)
    qtbot.mouseClick(panel.image_label, Qt.LeftButton, pos=QPoint(100, 100))
    panel.apply_snapshot(selectable_snapshot())
    panel.set_selection_enabled(False)
    qtbot.mouseClick(panel.image_label, Qt.LeftButton, pos=QPoint(100, 100))

    assert len(selected) == 0


def test_resize_recomputes_mapping_without_replacing_frame(qtbot) -> None:
    panel = CameraPanel()
    qtbot.addWidget(panel)
    current = selectable_snapshot()
    panel.image_label.setFixedSize(400, 200)
    panel.apply_snapshot(current)
    panel.set_selection_enabled(True)
    selected = QSignalSpy(panel.selection_requested)

    qtbot.mouseClick(panel.image_label, Qt.LeftButton, pos=QPoint(200, 100))

    assert selected[0][0].packet.sequence == current.result.packet.sequence
    assert selected[0][1:] == [100.0, 50.0]


def test_rendered_selection_owns_frame_against_later_source_mutation(qtbot) -> None:
    panel = CameraPanel()
    qtbot.addWidget(panel)
    panel.image_label.setFixedSize(200, 200)
    current = selectable_snapshot()
    panel.apply_snapshot(current)
    panel.set_selection_enabled(True)
    current.result.packet.frame[:] = 255
    selected = QSignalSpy(panel.selection_requested)

    qtbot.mouseClick(panel.image_label, Qt.LeftButton, pos=QPoint(100, 100))

    assert len(selected) == 1
    displayed = selected[0][0]
    assert displayed is not current.result
    assert np.all(displayed.packet.frame == 0)
    assert displayed.packet.frame.flags.writeable is False


def test_odd_sized_first_bottom_margin_pixel_does_not_emit(qtbot) -> None:
    panel = CameraPanel()
    qtbot.addWidget(panel)
    panel.image_label.setFixedSize(641, 600)
    panel.apply_snapshot(selectable_snapshot())
    panel.set_selection_enabled(True)
    selected = QSignalSpy(panel.selection_requested)

    qtbot.mouseClick(panel.image_label, Qt.LeftButton, pos=QPoint(320, 540))

    assert len(selected) == 0
