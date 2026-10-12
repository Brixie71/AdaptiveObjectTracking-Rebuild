from __future__ import annotations

import threading

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget

from aot.capture.types import SourceState
from aot.capture.worker import WorkerSnapshot
from aot.config.types import SourceType
from aot.runtime.types import LiveRuntimeSnapshot, SourceRuntimeSnapshot
from aot.ui.main_window import MainWindow


def source_snapshot(source_id: str) -> SourceRuntimeSnapshot:
    return SourceRuntimeSnapshot(
        source_id,
        source_id.upper(),
        SourceType.USB,
        WorkerSnapshot(source_id, SourceState.OPEN, 0, 0, None, True),
        None,
    )


class WindowRuntime:
    def __init__(self, sources=("a", "b")) -> None:
        self.running = False
        self.requires_cleanup = False
        self.source_ids = sources
        self.snapshot_calls = 0
        self.start_calls = 0
        self.stop_calls = 0
        self.start_entered = threading.Event()
        self.start_release = threading.Event()
        self.start_release.set()

    def start(self):
        self.start_calls += 1
        self.start_entered.set()
        self.start_release.wait(1.0)
        self.running = True

    def stop(self):
        self.stop_calls += 1
        self.running = False

    def snapshot(self):
        self.snapshot_calls += 1
        return LiveRuntimeSnapshot(
            self.running, tuple(source_snapshot(item) for item in self.source_ids)
        )


def test_window_has_three_stable_slots_and_refreshes_one_snapshot(qtbot) -> None:
    runtime = WindowRuntime(("one", "two"))
    window = MainWindow(runtime, refresh_interval_ms=1000)
    qtbot.addWidget(window)
    panels = tuple(window.camera_panels)
    before_refresh = runtime.snapshot_calls

    window.refresh_once()

    assert len(panels) == 3
    assert tuple(window.camera_panels) == panels
    assert runtime.snapshot_calls == before_refresh + 1
    assert panels[0].source_label.text().startswith("one")
    assert panels[1].source_label.text().startswith("two")
    assert panels[2].status_label.text() == "Unconfigured"
    assert window.start_button.isEnabled() is True
    assert window.stop_button.isEnabled() is False
    window.close()


def test_window_start_stop_controls_follow_controller(qtbot) -> None:
    runtime = WindowRuntime(("one",))
    window = MainWindow(runtime, refresh_interval_ms=1000)
    qtbot.addWidget(window)
    window.show()

    qtbot.mouseClick(window.start_button, __import__("PyQt5.QtCore").QtCore.Qt.LeftButton)
    qtbot.waitUntil(lambda: runtime.running)
    qtbot.waitUntil(lambda: window.stop_button.isEnabled())
    assert window.start_button.isEnabled() is False

    qtbot.mouseClick(window.stop_button, __import__("PyQt5.QtCore").QtCore.Qt.LeftButton)
    qtbot.waitUntil(lambda: not runtime.running)
    qtbot.waitUntil(lambda: window.start_button.isEnabled())
    window.close()


def test_settings_opens_without_stopping_runtime(qtbot) -> None:
    runtime = WindowRuntime(("one",))
    runtime.running = True
    dialogs: list[QWidget] = []

    def create_settings(parent):
        dialog = QWidget(parent)
        dialogs.append(dialog)
        return dialog

    window = MainWindow(runtime, settings_factory=create_settings)
    qtbot.addWidget(window)
    window.show()
    window.settings_button.click()

    assert len(dialogs) == 1
    assert dialogs[0].isVisible()
    assert runtime.stop_calls == 0
    window.close()
    qtbot.waitUntil(lambda: runtime.stop_calls == 1)


def test_close_during_blocked_start_keeps_event_loop_responsive(qtbot) -> None:
    runtime = WindowRuntime(("one",))
    runtime.start_release.clear()
    window = MainWindow(runtime)
    qtbot.addWidget(window)
    heartbeats: list[int] = []
    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: heartbeats.append(1))
    timer.start()

    window.start_button.click()
    assert runtime.start_entered.wait(0.5)
    window.close()
    qtbot.waitUntil(lambda: len(heartbeats) >= 2)
    runtime.start_release.set()
    qtbot.waitUntil(lambda: runtime.stop_calls == 1)

    assert runtime.start_calls == 1
    assert runtime.stop_calls == 1


def test_window_displays_sanitized_lifecycle_failure(qtbot) -> None:
    class FailingRuntime(WindowRuntime):
        def start(self):
            raise RuntimeError("PRIVATE-MODEL-PATH")

    window = MainWindow(FailingRuntime(("one",)))
    qtbot.addWidget(window)
    window.start_button.click()
    qtbot.waitUntil(lambda: bool(window.operation_status_label.text()))

    assert window.operation_status_label.text() == "Unable to start live view"
    assert "PRIVATE" not in window.operation_status_label.text()
    window.close()


def test_settings_dialog_is_reused_while_probe_remains_active(qtbot) -> None:
    runtime = WindowRuntime(("one",))
    entered = threading.Event()
    release = threading.Event()
    calls = []

    from aot.config.types import CameraSourceConfig, SourceType
    from aot.ui.settings_dialog import SettingsDialog

    sources = (CameraSourceConfig("one", "One", SourceType.USB, True, device_index=0),)

    def probe(_sources):
        calls.append(1)
        entered.set()
        release.wait(1.0)
        return ()

    def settings_factory(parent):
        return SettingsDialog(sources, probe, lambda: (), lambda: None, parent)

    window = MainWindow(runtime, settings_factory=settings_factory)
    qtbot.addWidget(window)
    window.show()
    window.open_settings()
    first = window._settings_window
    first.hardware_testing.run_sources_button.click()
    assert entered.wait(0.5)
    first.close()
    window.open_settings()

    assert window._settings_window is first
    window._settings_window.hardware_testing.run_sources_button.click()
    assert calls == [1]
    release.set()
    qtbot.waitUntil(lambda: first.hardware_testing.run_sources_button.isEnabled())
    window.close()
