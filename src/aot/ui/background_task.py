"""Small fixed-result QRunnable used by native UI controllers."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from typing import Any

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, Qt, pyqtSignal, pyqtSlot


class FunctionTaskSignals(QObject):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    finished = pyqtSignal(object)


class FunctionTask(QRunnable):
    """Run one callable and expose only success or a fixed safe failure."""

    def __init__(self, function: Callable[[], Any]) -> None:
        super().__init__()
        self.signals = FunctionTaskSignals()
        self._function = function
        self._callback_lock = threading.Lock()
        self._callbacks_enabled = True
        self.succeeded: bool | None = None
        # Python owns the runnable and its signal object.  Deleting the wrapper
        # from its worker thread can strand Qt while queued callbacks unwind.
        self.setAutoDelete(False)

    def cancel_callbacks(self) -> None:
        with self._callback_lock:
            self._callbacks_enabled = False

    def _can_emit(self) -> bool:
        with self._callback_lock:
            return self._callbacks_enabled

    def run(self) -> None:
        try:
            result = self._function()
        except Exception:
            self.succeeded = False
            if self._can_emit():
                self.signals.failed.emit("Background operation failed")
        else:
            self.succeeded = True
            if self._can_emit():
                self.signals.succeeded.emit(result)
        finally:
            self.signals.finished.emit(self)


class _TaskRegistry(QObject):
    """Keep Python-owned runnables alive and retire them on the GUI thread."""

    def __init__(self) -> None:
        super().__init__()
        self._active: dict[int, FunctionTask] = {}
        self._retired: deque[FunctionTask] = deque(maxlen=8)

    def submit(self, pool: QThreadPool, task: FunctionTask) -> None:
        self._active[id(task)] = task
        task.signals.finished.connect(self._release, Qt.QueuedConnection)
        pool.start(task)

    @pyqtSlot(object)
    def _release(self, task: FunctionTask) -> None:
        retained = self._active.pop(id(task), None)
        if retained is not None:
            self._retired.append(retained)


_TASK_REGISTRY = _TaskRegistry()


def submit_function_task(pool: QThreadPool, task: FunctionTask) -> None:
    """Submit through a bounded registry so QRunnable cleanup stays UI-owned."""
    _TASK_REGISTRY.submit(pool, task)
