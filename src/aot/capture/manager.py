"""Ownership boundary for a set of independent capture workers."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

from aot.capture.buffer import NewestFrameBuffer
from aot.capture.factory import create_frame_source
from aot.capture.types import FramePacket, FrameSource
from aot.capture.worker import CaptureWorker, ReconnectPolicy, WorkerSnapshot
from aot.config.types import CameraSourceConfig, CaptureRuntimeConfig


class ManagerShutdownError(RuntimeError):
    def __init__(self, source_ids: Iterable[str]) -> None:
        self.source_ids = tuple(source_ids)
        super().__init__(
            "Capture workers did not stop: " + ", ".join(self.source_ids)
        )


class CaptureManager:
    def __init__(
        self,
        runtime_config: CaptureRuntimeConfig,
        source_factory: Callable[[CameraSourceConfig], FrameSource] = create_frame_source,
    ) -> None:
        self._runtime_config = runtime_config
        self._source_factory = source_factory
        self._workers: dict[str, CaptureWorker] = {}
        self._started = False

    def start(self, sources: Iterable[CameraSourceConfig]) -> None:
        if self._started:
            raise RuntimeError("Capture manager is already started")
        enabled = [source for source in sources if source.enabled]
        source_ids = [source.source_id for source in enabled]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("Enabled capture source IDs must be unique")

        for config in enabled:
            source = self._source_factory(config)
            buffer = NewestFrameBuffer(self._runtime_config.queue_capacity)
            policy = ReconnectPolicy(
                self._runtime_config.reconnect_initial_seconds,
                self._runtime_config.reconnect_max_seconds,
            )
            self._workers[config.source_id] = CaptureWorker(
                source,
                buffer,
                policy,
                self._runtime_config.shutdown_timeout_seconds,
            )
        self._started = True
        for worker in self._workers.values():
            worker.start()

    def snapshot(self) -> dict[str, WorkerSnapshot]:
        return {
            source_id: worker.snapshot()
            for source_id, worker in self._workers.items()
        }

    def latest(self, source_id: str) -> FramePacket | None:
        return self._workers[source_id].buffer.get_latest()

    def stop(self) -> None:
        for worker in self._workers.values():
            worker.request_stop()

        deadline = time.monotonic() + self._runtime_config.shutdown_timeout_seconds
        stuck: list[str] = []
        for source_id, worker in self._workers.items():
            remaining = max(0.0, deadline - time.monotonic())
            if not worker.join(remaining):
                stuck.append(source_id)

        if stuck:
            raise ManagerShutdownError(stuck)
