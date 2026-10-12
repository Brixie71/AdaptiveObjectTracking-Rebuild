from __future__ import annotations

import threading

from PyQt5.QtCore import QThread
from PyQt5.QtTest import QSignalSpy

from aot.ui.controller import RuntimeController


class BlockingRuntime:
    def __init__(self) -> None:
        self.running = False
        self.start_entered = threading.Event()
        self.start_release = threading.Event()
        self.stop_entered = threading.Event()
        self.stop_release = threading.Event()
        self.start_threads = []
        self.stop_threads = []
        self.start_calls = 0
        self.stop_calls = 0
        self.start_error = False

    def start(self) -> None:
        self.start_calls += 1
        self.start_threads.append(QThread.currentThread())
        self.start_entered.set()
        self.start_release.wait(1.0)
        if self.start_error:
            raise RuntimeError("PRIVATE-START-DETAIL")
        self.running = True

    def stop(self) -> None:
        self.stop_calls += 1
        self.stop_threads.append(QThread.currentThread())
        self.stop_entered.set()
        self.stop_release.wait(1.0)
        self.running = False


def test_controller_serializes_commands_and_runs_start_off_gui_thread(qapp, qtbot) -> None:
    runtime = BlockingRuntime()
    controller = RuntimeController(runtime)
    started = QSignalSpy(controller.started)

    assert controller.request_start() is True
    assert runtime.start_entered.wait(0.5)
    assert controller.busy is True
    assert controller.can_start is False
    assert controller.can_stop is False
    assert controller.request_start() is False
    assert runtime.start_calls == 1
    runtime.start_release.set()
    qtbot.waitUntil(lambda: len(started) == 1)

    assert runtime.start_threads[0] is not qapp.thread()
    assert controller.can_start is False
    assert controller.can_stop is True
    controller.close()
    runtime.stop_release.set()


def test_controller_stop_and_fixed_failure_summaries(qapp, qtbot) -> None:
    runtime = BlockingRuntime()
    runtime.start_error = True
    runtime.start_release.set()
    controller = RuntimeController(runtime)
    failures = QSignalSpy(controller.failed)

    assert controller.request_start() is True
    qtbot.waitUntil(lambda: len(failures) == 1)
    assert failures[0] == ["Unable to start live view"]
    assert "PRIVATE" not in failures[0][0]

    runtime.start_error = False
    runtime.running = True
    controller.sync_runtime_state()
    stopped = QSignalSpy(controller.stopped)
    assert controller.request_stop() is True
    assert runtime.stop_entered.wait(0.5)
    assert controller.request_stop() is False
    runtime.stop_release.set()
    qtbot.waitUntil(lambda: len(stopped) == 1)
    assert runtime.stop_threads[0] is not qapp.thread()
    controller.close()


def test_close_during_blocked_start_stops_runtime_without_late_signals(qapp, qtbot) -> None:
    runtime = BlockingRuntime()
    controller = RuntimeController(runtime)
    started = QSignalSpy(controller.started)
    stopped = QSignalSpy(controller.stopped)
    controller.request_start()
    assert runtime.start_entered.wait(0.5)

    controller.close()
    runtime.start_release.set()
    runtime.stop_release.set()
    qtbot.waitUntil(lambda: runtime.stop_calls == 1)

    assert len(started) == 0
    assert len(stopped) == 0
