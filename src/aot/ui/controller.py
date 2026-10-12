"""Asynchronous Qt command bridge for the live runtime."""

from __future__ import annotations

import threading
import time

from PyQt5.QtCore import QObject, QThreadPool, Qt, pyqtSignal, pyqtSlot

from aot.runtime.live_runtime import LiveRuntime
from aot.ui.background_task import FunctionTask, submit_function_task


class RuntimeController(QObject):
    started = pyqtSignal()
    stopped = pyqtSignal()
    failed = pyqtSignal(str)
    controls_changed = pyqtSignal(bool, bool, bool)

    def __init__(
        self,
        runtime: LiveRuntime,
        parent: QObject | None = None,
        *,
        thread_pool: QThreadPool | None = None,
    ) -> None:
        super().__init__(parent)
        self._runtime = runtime
        self._pool = thread_pool or QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._busy = False
        self._operation: str | None = None
        self._task: FunctionTask | None = None
        self._closed = False
        self._guard = threading.Lock()
        self._worker_start_succeeded = False
        self._stop_after_start = False
        self._had_failure = False

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def can_start(self) -> bool:
        return (
            not self._closed
            and not self._busy
            and not self._runtime.running
            and not self._requires_cleanup()
        )

    @property
    def can_stop(self) -> bool:
        if self._closed:
            return False
        if self._busy:
            return self._operation == "start" and not self._stop_after_start
        return self._runtime.running or self._requires_cleanup()

    def _requires_cleanup(self) -> bool:
        return bool(getattr(self._runtime, "requires_cleanup", False))

    def sync_runtime_state(self) -> None:
        self._emit_controls()

    def request_start(self) -> bool:
        if not self.can_start:
            return False
        self._begin("start", self._start_runtime)
        return True

    def request_stop(self) -> bool:
        if not self.can_stop:
            return False
        if self._busy:
            self._stop_after_start = True
            self._emit_controls()
            return True
        self._begin("stop", self._runtime.stop)
        return True

    def _start_runtime(self) -> str:
        self._runtime.start()
        with self._guard:
            self._worker_start_succeeded = True
            must_stop = self._closed or self._stop_after_start
        if must_stop:
            self._runtime.stop()
            return "stopped"
        return "started"

    def _begin(self, operation: str, function) -> None:
        self._busy = True
        self._operation = operation
        self._worker_start_succeeded = False
        if operation == "start":
            self._stop_after_start = False
        task = FunctionTask(function)
        task.signals.succeeded.connect(self._on_success, Qt.QueuedConnection)
        task.signals.failed.connect(self._on_failure, Qt.QueuedConnection)
        self._task = task
        self._emit_controls()
        submit_function_task(self._pool, task)

    @pyqtSlot(object)
    def _on_success(self, result) -> None:
        operation = self._operation
        self._finish_operation()
        if self._closed:
            return
        if operation == "start" and result != "stopped":
            self.started.emit()
        else:
            self.stopped.emit()

    @pyqtSlot(str)
    def _on_failure(self, _summary: str) -> None:
        operation = self._operation
        self._had_failure = True
        self._finish_operation()
        if self._closed:
            return
        if operation == "start":
            self.failed.emit("Unable to start live view")
        else:
            self.failed.emit("Unable to stop live view")

    def _finish_operation(self) -> None:
        self._busy = False
        self._task = None
        self._operation = None
        self._stop_after_start = False
        self._emit_controls()

    def _emit_controls(self) -> None:
        self.controls_changed.emit(self.can_start, self.can_stop, self._busy)

    def close(self) -> None:
        with self._guard:
            if self._closed:
                return
            self._closed = True
            operation = self._operation
            start_already_succeeded = self._worker_start_succeeded
        if self._task is not None:
            self._task.cancel_callbacks()

        should_submit_stop = (
            operation is None
            and (self._runtime.running or self._requires_cleanup())
        ) or (operation == "start" and start_already_succeeded)
        if should_submit_stop:
            cleanup = FunctionTask(self._runtime.stop)
            cleanup.cancel_callbacks()
            self._task = cleanup
            submit_function_task(self._pool, cleanup)
        self._emit_controls()

    def wait_for_done(self, timeout_ms: int = 10_000) -> bool:
        """Wait only after the GUI loop exits and report close-task success."""
        deadline = time.monotonic() + max(timeout_ms, 0) / 1000
        if not self._pool.waitForDone(max(timeout_ms, 0)):
            return False
        task = self._task
        if task is not None:
            remaining = max(0.0, deadline - time.monotonic())
            if not task.wait(remaining):
                return False
            if task.succeeded is False:
                return False
        return not self._had_failure and not self._requires_cleanup()
