from __future__ import annotations

import numpy as np
import pytest
from PyQt5.QtCore import Qt

from aot.capture.types import FramePacket, SourceState
from aot.capture.worker import WorkerSnapshot
from aot.config.types import SourceType
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
