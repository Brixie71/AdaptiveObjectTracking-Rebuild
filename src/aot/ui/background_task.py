"""Small fixed-result QRunnable used by native UI controllers."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal


class FunctionTaskSignals(QObject):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)


class FunctionTask(QRunnable):
    """Run one callable and expose only success or a fixed safe failure."""

    def __init__(self, function: Callable[[], Any]) -> None:
        super().__init__()
        self.signals = FunctionTaskSignals()
        self._function = function
        self._callback_lock = threading.Lock()
        self._callbacks_enabled = True
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
            if self._can_emit():
                self.signals.failed.emit("Background operation failed")
            return
        if self._can_emit():
            self.signals.succeeded.emit(result)
