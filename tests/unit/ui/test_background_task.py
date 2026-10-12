from __future__ import annotations

import threading

from PyQt5.QtCore import QThread, QThreadPool
from PyQt5.QtTest import QSignalSpy

from aot.ui.background_task import FunctionTask


def test_function_task_runs_off_gui_thread_and_emits_one_success(qapp, qtbot) -> None:
    pool = QThreadPool()
    pool.setMaxThreadCount(1)
    worker_threads = []
    task = FunctionTask(lambda: worker_threads.append(QThread.currentThread()) or 42)
    success = QSignalSpy(task.signals.succeeded)
    failure = QSignalSpy(task.signals.failed)

    pool.start(task)
    qtbot.waitUntil(lambda: len(success) == 1)

    assert success[0] == [42]
    assert len(failure) == 0
    assert worker_threads[0] is not qapp.thread()
    assert not hasattr(task, "queue")


def test_function_task_sanitizes_failure(qapp, qtbot) -> None:
    def fail():
        raise RuntimeError("PRIVATE-CREDENTIAL")

    pool = QThreadPool()
    task = FunctionTask(fail)
    failure = QSignalSpy(task.signals.failed)
    pool.start(task)
    qtbot.waitUntil(lambda: len(failure) == 1)

    assert failure[0] == ["Background operation failed"]
    assert "PRIVATE" not in failure[0][0]


def test_cancel_callbacks_discards_late_result(qapp, qtbot) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocked():
        entered.set()
        release.wait(1.0)
        return "late"

    pool = QThreadPool()
    task = FunctionTask(blocked)
    success = QSignalSpy(task.signals.succeeded)
    pool.start(task)
    assert entered.wait(0.5)
    task.cancel_callbacks()
    release.set()
    assert pool.waitForDone(1000)
    qapp.processEvents()

    assert len(success) == 0
