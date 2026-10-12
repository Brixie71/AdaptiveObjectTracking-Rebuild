from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QSignalSpy

from aot.selection.types import PixelBox, SelectedTarget
from aot.ui.selected_target_panel import SelectedTargetPanel


def selected_target() -> SelectedTarget:
    crop = np.zeros((2, 3, 3), dtype=np.uint8)
    crop[:, :, 1] = 200
    return SelectedTarget(
        target_id="target-1",
        source_id="camera-a",
        frame_sequence=17,
        captured_at_monotonic_ns=123_456,
        selected_at_utc=datetime(2026, 10, 12, 9, 30, tzinfo=timezone.utc),
        class_id=0,
        class_name="person",
        confidence=0.91,
        initial_box=PixelBox(1, 1, 4, 3),
        reference_crop=crop,
    )


def test_panel_starts_empty_with_disengage_disabled(qtbot) -> None:
    panel = SelectedTargetPanel()
    qtbot.addWidget(panel)

    assert panel.status_label.text() == "No target selected"
    assert panel.crop_label.pixmap() is None
    assert not panel.disengage_button.isEnabled()
    assert all(
        label.text() == ""
        for label in (
            panel.target_id_label,
            panel.class_label,
            panel.source_label,
            panel.frame_label,
            panel.selected_at_label,
            panel.box_label,
        )
    )


def test_apply_target_shows_safe_reference_metadata_and_crop(qtbot) -> None:
    panel = SelectedTargetPanel()
    panel.crop_label.setFixedSize(180, 120)
    qtbot.addWidget(panel)

    panel.apply_target(selected_target())

    assert panel.status_label.text() == "Target selected"
    assert panel.target_id_label.text() == "Target: target-1"
    assert panel.class_label.text() == "Class: person (91%)"
    assert panel.source_label.text() == "Source: camera-a"
    assert panel.frame_label.text() == "Frame: 17"
    assert panel.selected_at_label.text() == "Selected: 2026-10-12T09:30:00+00:00"
    assert panel.box_label.text() == "Box: (1, 1)-(4, 3)"
    assert panel.crop_label.pixmap() is not None
    assert panel.crop_label.pixmap().width() / panel.crop_label.pixmap().height() == 1.5
    assert panel.crop_label.alignment() == Qt.AlignCenter
    assert panel.disengage_button.isEnabled()

    visible_text = " ".join(label.text() for label in panel.findChildren(type(panel.status_label)))
    assert "array(" not in visible_text
    assert "123456" not in visible_text
    assert "PRIVATE" not in visible_text


def test_dynamic_metadata_is_always_plain_text(qtbot) -> None:
    panel = SelectedTargetPanel()
    qtbot.addWidget(panel)
    target = selected_target()
    object.__setattr__(target, "class_name", "<b>person</b>")
    object.__setattr__(target, "source_id", "<i>camera-a</i>")

    panel.apply_target(target)

    assert panel.class_label.textFormat() == Qt.PlainText
    assert panel.source_label.textFormat() == Qt.PlainText
    assert panel.class_label.text() == "Class: <b>person</b> (91%)"


def test_disengage_button_emits_intent_once_without_clearing_state(qtbot) -> None:
    panel = SelectedTargetPanel()
    qtbot.addWidget(panel)
    panel.apply_target(selected_target())
    requested = QSignalSpy(panel.disengage_requested)

    qtbot.mouseClick(panel.disengage_button, Qt.LeftButton)

    assert len(requested) == 1
    assert panel.status_label.text() == "Target selected"
    assert panel.crop_label.pixmap() is not None


def test_apply_none_clears_prior_crop_and_metadata(qtbot) -> None:
    panel = SelectedTargetPanel()
    qtbot.addWidget(panel)
    panel.apply_target(selected_target())

    panel.apply_target(None)

    assert panel.status_label.text() == "No target selected"
    assert panel.crop_label.pixmap() is None
    assert not panel.disengage_button.isEnabled()
    assert all(
        label.text() == ""
        for label in (
            panel.target_id_label,
            panel.class_label,
            panel.source_label,
            panel.frame_label,
            panel.selected_at_label,
            panel.box_label,
        )
    )
