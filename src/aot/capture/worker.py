"""Independent capture worker and bounded reconnect policy."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from aot.capture.buffer import NewestFrameBuffer
from aot.capture.types import CaptureError, EndOfStream, FrameSource, SourceState


class ReconnectPolicy:
    def __init__(self, initial_seconds: float, maximum_seconds: float) -> None:
        if initial_seconds <= 0 or maximum_seconds < initial_seconds:
            raise ValueError("Reconnect delays must be positive and ordered")
        self._initial = float(initial_seconds)
        self._maximum = float(maximum_seconds)
        self._next = self._initial

    def next_delay(self) -> float:
        delay = self._next
        self._next = min(self._next * 2, self._maximum)
        return delay

    def reset(self) -> None:
        self._next = self._initial


@dataclass(frozen=True)
class WorkerSnapshot:
    source_id: str
    state: SourceState
    queue_depth: int
    dropped_frames: int
    last_error: str | None
    alive: bool


class CaptureWorker:
    def __init__(
        self,
        source: FrameSource,
        buffer: NewestFrameBuffer,
        reconnect_policy: ReconnectPolicy,
        shutdown_timeout_seconds: float,
        *,
        wait_for_stop: Callable[[float], bool] | None = None,
    ) -> None:
        self.source = source
        self.buffer = buffer
        self._reconnect_policy = reconnect_policy
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._stop_event = threading.Event()
        self._wait_for_stop = wait_for_stop or self._stop_event.wait
        self._thread: threading.Thread | None = None
        self._snapshot_lock = threading.Lock()
        self._attempt_lock = threading.Lock()
        self._attempt_closed = True
        self._closer_started = False
        self._state = SourceState.CLOSED
        self._last_error: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Capture worker has already been started")
        self._thread = threading.Thread(
            target=self._run,
            name=f"capture-{self.source.source_id}",
            daemon=True,
        )
        self._thread.start()

    def _set_status(
        self,
        state: SourceState,
        last_error: str | None = None,
    ) -> None:
        with self._snapshot_lock:
            self._state = state
            self._last_error = last_error

    def _begin_attempt(self) -> None:
        with self._attempt_lock:
            self._attempt_closed = False

    def _close_attempt(self) -> None:
        with self._attempt_lock:
            if self._attempt_closed:
                return
            self._attempt_closed = True
        try:
            self.source.close()
        except Exception:
            self._set_status(SourceState.FAILED, "Source cleanup failed")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._begin_attempt()
            retry = False
            try:
                self._set_status(SourceState.OPENING)
                self.source.open()
                with self._attempt_lock:
                    if self._stop_event.is_set() and self._attempt_closed:
                        self._attempt_closed = False
                if self._stop_event.is_set():
                    return
                self._set_status(SourceState.OPEN)
                while not self._stop_event.is_set():
                    packet = self.source.read()
                    self.buffer.put(packet)
                    self._reconnect_policy.reset()
                    self._set_status(SourceState.OPEN)
            except EndOfStream:
                self._set_status(SourceState.EOF)
                return
            except Exception:
                if self.source.reconnectable and not self._stop_event.is_set():
                    retry = True
                    self._set_status(
                        SourceState.RECONNECTING,
                        "Source operation failed",
                    )
                else:
                    self._set_status(SourceState.FAILED, "Source operation failed")
                    return
            finally:
                self._close_attempt()

            if retry:
                delay = self._reconnect_policy.next_delay()
                if self._wait_for_stop(delay):
                    self._set_status(SourceState.CLOSED, self._last_error)
                    return

        self._set_status(SourceState.CLOSED)

    def stop(self) -> bool:
        deadline = time.monotonic() + self._shutdown_timeout_seconds
        self.request_stop()
        stopped = self.join(max(0.0, deadline - time.monotonic()))
        if stopped:
            self._set_status(SourceState.CLOSED, self._last_error)
        return stopped

    def request_stop(self) -> None:
        """Signal shutdown and release the current attempt without joining."""
        self._stop_event.set()
        with self._attempt_lock:
            should_start_closer = not self._closer_started
            self._closer_started = True
        if should_start_closer:
            threading.Thread(
                target=self._close_attempt,
                name=f"capture-close-{self.source.source_id}",
                daemon=True,
            ).start()

    def join(self, timeout: float | None = None) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> WorkerSnapshot:
        with self._snapshot_lock:
            state = self._state
            last_error = self._last_error
        return WorkerSnapshot(
            source_id=self.source.source_id,
            state=state,
            queue_depth=self.buffer.size,
            dropped_frames=self.buffer.dropped_count,
            last_error=last_error,
            alive=self.is_alive,
        )
