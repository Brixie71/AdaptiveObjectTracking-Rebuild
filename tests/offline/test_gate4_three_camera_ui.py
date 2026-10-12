from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
from PyQt5.QtCore import QTimer

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
from aot.ui.main_window import MainWindow
from aot.ui.rendering import describe_overlays


FIXTURE = Path(__file__).parents[1] / "fixtures" / "media" / "gate2-three-frame.avi"


def load_frames():
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


class OfflineCaptureManager:
    def __init__(self, frames) -> None:
        self.frames = frames
        self.packets = {}
        self.states = {}
        self.stopped = False
        self.started = False

    def start(self, sources) -> None:
        self.started = True
        for index, source in enumerate(sources):
            self.states[source.source_id] = SourceState.OPEN
            self.packets[source.source_id] = FramePacket(
                source.source_id,
                1,
                time.monotonic_ns(),
                self.frames[index].copy(),
            )

    def latest(self, source_id):
        return self.packets[source_id]

    def snapshot(self):
        return {
            source_id: WorkerSnapshot(
                source_id,
                state,
                1 if state is SourceState.OPEN else 0,
                0,
                "Source operation failed" if state is SourceState.FAILED else None,
                not self.stopped,
            )
            for source_id, state in self.states.items()
        }

    def advance(self, source_id: str, sequence: int, frame_index: int) -> None:
        self.packets[source_id] = FramePacket(
            source_id,
            sequence,
            time.monotonic_ns(),
            self.frames[frame_index].copy(),
        )

    def stop(self) -> None:
        self.stopped = True
        self.states = {source_id: SourceState.CLOSED for source_id in self.states}


class BlockingDetector:
    model_id = "deterministic"
    class_names = ("person",)

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed = False
        self._first = True

    def detect(self, packet: FramePacket) -> DetectionBatch:
        if self._first:
            self._first = False
            self.entered.set()
            self.release.wait(2.0)
        height, width = packet.frame.shape[:2]
        return DetectionBatch(
            packet.source_id,
            packet.sequence,
            packet.captured_at_monotonic_ns,
            width,
            height,
            (
                Detection(
                    0,
                    "person",
                    0.9,
                    BoundingBox(5, 5, 50, 60),
                ),
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
            model_id="deterministic",
            allowed_classes=("person",),
        ),
    )


def test_three_camera_ui_responsiveness_isolation_and_clean_shutdown(qtbot) -> None:
    frames = load_frames()
    captures = []
    detector = BlockingDetector()

    def capture_factory(_config):
        manager = OfflineCaptureManager(frames)
        captures.append(manager)
        return manager

    runtime = LiveRuntime(app_settings(), capture_factory, lambda: detector)
    window = MainWindow(
        runtime,
        target_selector=TargetSelector(),
        refresh_interval_ms=10,
    )
    qtbot.addWidget(window)
    window.show()
    heartbeat = []
    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: heartbeat.append(1))
    timer.start()

    window.start_button.click()
    assert detector.entered.wait(0.5)
    qtbot.waitUntil(lambda: len(heartbeat) >= 3)
    assert window.stop_button.isEnabled() is True
    detector.release.set()
    qtbot.waitUntil(lambda: runtime.running)
    qtbot.waitUntil(
        lambda: all(panel.image_label.pixmap() is not None for panel in window.camera_panels)
    )

    first_snapshot = runtime.snapshot()
    assert [item.source_id for item in first_snapshot.sources] == [
        "camera-a",
        "camera-b",
        "camera-c",
    ]
    for source in first_snapshot.sources:
        assert source.result.packet.source_id == source.source_id
        assert source.result.detections.source_id == source.source_id
        assert source.result.detections.frame_sequence == source.result.packet.sequence
        assert describe_overlays(source.result)[0].label == "person 90%"

    manager = captures[0]
    for index, source_id in enumerate(("camera-a", "camera-b", "camera-c")):
        manager.advance(source_id, 2, (index + 1) % 3)
    qtbot.waitUntil(
        lambda: all(
            item.result is not None and item.result.packet.sequence == 2
            for item in runtime.snapshot().sources
        )
    )

    manager.states["camera-a"] = SourceState.FAILED
    manager.states["camera-b"] = SourceState.EOF
    manager.advance("camera-c", 3, 1)
    qtbot.waitUntil(
        lambda: window.camera_panels[0].status_label.text() == "Unavailable"
        and window.camera_panels[1].status_label.text() == "End of stream"
    )
    qtbot.waitUntil(
        lambda: runtime.snapshot().sources[2].result.packet.sequence == 3
    )
    visible = " ".join(panel.status_label.text() for panel in window.camera_panels)
    assert "PRIVATE" not in visible
    assert window.stop_button.isEnabled()

    window.stop_button.click()
    qtbot.waitUntil(lambda: not runtime.running)
    qtbot.waitUntil(lambda: manager.stopped)
    assert detector.closed is True
    assert runtime._processing is None
    window.close()
    assert window.wait_for_shutdown() is True
    runtime.stop()
    assert manager.stopped is True
