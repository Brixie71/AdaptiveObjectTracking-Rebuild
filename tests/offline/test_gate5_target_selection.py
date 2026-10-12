from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, QTimer, Qt

from aot.capture.types import FramePacket, SourceState
from aot.capture.worker import WorkerSnapshot
from aot.config.types import (
    AppSettings,
    CameraSourceConfig,
    ComputeConfig,
    DetectionConfig,
    SourceType,
)
from aot.detection.types import BoundingBox, Detection, DetectionBatch
from aot.runtime.live_runtime import LiveRuntime
from aot.selection.target_selector import TargetSelector
from aot.selection.types import PixelBox
from aot.ui.main_window import MainWindow


FIXTURE = Path(__file__).parents[1] / "fixtures" / "media" / "gate2-three-frame.avi"


def load_frames() -> tuple[np.ndarray, ...]:
    capture = cv2.VideoCapture(str(FIXTURE))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    assert len(frames) == 3
    return tuple(frames)


class StaticThreeSourceCapture:
    def __init__(self, frames: tuple[np.ndarray, ...]) -> None:
        self.frames = frames
        self.packets: dict[str, FramePacket] = {}
        self.states: dict[str, SourceState] = {}
        self.stopped = False

    def start(self, sources) -> None:
        for index, source in enumerate(sources):
            self.states[source.source_id] = SourceState.OPEN
            self.packets[source.source_id] = FramePacket(
                source.source_id,
                index + 1,
                time.monotonic_ns(),
                self.frames[index].copy(),
            )

    def latest(self, source_id: str) -> FramePacket:
        return self.packets[source_id]

    def snapshot(self):
        return {
            source_id: WorkerSnapshot(
                source_id,
                state,
                1,
                0,
                None,
                not self.stopped,
            )
            for source_id, state in self.states.items()
        }

    def stop(self) -> None:
        self.stopped = True
        self.states = {source_id: SourceState.CLOSED for source_id in self.states}


class SelectionDetector:
    model_id = "deterministic-selection"
    class_names = ("person", "backpack", "bottle")

    def __init__(self) -> None:
        self.closed = False

    def detect(self, packet: FramePacket) -> DetectionBatch:
        height, width = packet.frame.shape[:2]
        return DetectionBatch(
            packet.source_id,
            packet.sequence,
            packet.captured_at_monotonic_ns,
            width,
            height,
            (
                Detection(0, "person", 0.9, BoundingBox(20, 20, 120, 120)),
                Detection(1, "backpack", 0.8, BoundingBox(-5, -5, 80, 80)),
                Detection(2, "bottle", 0.9, BoundingBox(30, 30, 70, 70)),
            ),
            0,
        )

    def close(self) -> None:
        self.closed = True


def app_settings() -> AppSettings:
    sources = tuple(
        CameraSourceConfig(
            source_id,
            name,
            SourceType.OFFLINE,
            True,
            offline_path=FIXTURE,
        )
        for source_id, name in (
            ("camera-a", "Camera A"),
            ("camera-b", "Camera B"),
            ("camera-c", "Camera C"),
        )
    )
    return AppSettings(
        1,
        Path("runtime"),
        "INFO",
        ComputeConfig(),
        sources,
        detection=DetectionConfig(
            enabled=True,
            model_id="deterministic-selection",
            allowed_classes=("person", "backpack", "bottle"),
        ),
    )


def assert_crop_matches_displayed_frame(window: MainWindow, index: int, selector) -> None:
    result = window.camera_panels[index]._displayed_result
    target = selector.current
    box = target.initial_box
    expected = result.packet.frame[box.top : box.bottom, box.left : box.right]
    assert target.source_id == result.packet.source_id
    assert target.frame_sequence == result.packet.sequence
    assert target.captured_at_monotonic_ns == result.packet.captured_at_monotonic_ns
    assert np.array_equal(target.reference_crop, expected)


def test_gate5_selects_replaces_preserves_and_disengages_through_native_ui(
    qtbot,
) -> None:
    frames = load_frames()
    captures = []
    detector = SelectionDetector()

    def capture_factory(_config):
        manager = StaticThreeSourceCapture(frames)
        captures.append(manager)
        return manager

    ids = iter(("target-a", "target-b", "target-c"))
    selector = TargetSelector(
        id_factory=lambda: next(ids),
        clock=lambda: datetime(2026, 10, 12, 10, 0, tzinfo=timezone.utc),
    )
    runtime = LiveRuntime(app_settings(), capture_factory, lambda: detector)
    window = MainWindow(runtime, target_selector=selector, refresh_interval_ms=10)
    qtbot.addWidget(window)
    for panel in window.camera_panels:
        panel.image_label.setFixedSize(640, 600)
    window.show()

    heartbeat = []
    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: heartbeat.append(1))
    timer.start()
    window.start_button.click()
    qtbot.waitUntil(lambda: runtime.running)
    qtbot.waitUntil(
        lambda: all(panel._displayed_result is not None for panel in window.camera_panels)
    )
    qtbot.waitUntil(lambda: len(heartbeat) >= 3)

    # The 640x480 frame is centered in the 640x600 label with 60px top padding.
    qtbot.mouseClick(
        window.camera_panels[0].image_label,
        Qt.LeftButton,
        pos=QPoint(40, 100),
    )
    assert selector.current.target_id == "target-a"
    assert selector.current.class_name == "bottle"
    assert selector.current.initial_box == PixelBox(30, 30, 70, 70)
    assert window.selected_target_panel.class_label.text() == "Class: bottle (90%)"
    assert_crop_matches_displayed_frame(window, 0, selector)

    current = selector.current
    qtbot.mouseClick(
        window.camera_panels[0].image_label,
        Qt.LeftButton,
        pos=QPoint(320, 20),
    )
    assert selector.current is current
    qtbot.mouseClick(
        window.camera_panels[0].image_label,
        Qt.LeftButton,
        pos=QPoint(500, 460),
    )
    assert selector.current is current
    assert window.operation_status_label.text() == "No selectable target at this point"

    qtbot.mouseClick(
        window.camera_panels[1].image_label,
        Qt.LeftButton,
        pos=QPoint(5, 65),
    )
    assert selector.current.target_id == "target-b"
    assert selector.current.class_name == "backpack"
    assert selector.current.initial_box == PixelBox(0, 0, 80, 80)
    assert_crop_matches_displayed_frame(window, 1, selector)

    qtbot.mouseClick(
        window.camera_panels[2].image_label,
        Qt.LeftButton,
        pos=QPoint(100, 160),
    )
    assert selector.current.target_id == "target-c"
    assert selector.current.source_id == "camera-c"
    assert selector.current.frame_sequence == 3
    assert selector.current.class_name == "person"
    assert selector.current.confidence == 0.9
    assert selector.current.initial_box == PixelBox(20, 20, 120, 120)
    assert window.selected_target_panel.source_label.text() == "Source: camera-c"
    assert window.selected_target_panel.frame_label.text() == "Frame: 3"
    assert_crop_matches_displayed_frame(window, 2, selector)

    qtbot.mouseClick(window.selected_target_panel.disengage_button, Qt.LeftButton)
    assert selector.current is None
    assert window.selected_target_panel.crop_label.pixmap() is None
    assert window.operation_status_label.text() == "Target disengaged"

    window.stop_button.click()
    qtbot.waitUntil(lambda: not runtime.running)
    qtbot.waitUntil(lambda: captures[0].stopped)
    assert detector.closed is True
    assert runtime._processing is None
    window.close()
    assert window.wait_for_shutdown() is True
