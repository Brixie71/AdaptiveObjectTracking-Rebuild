import threading
import time
from collections import deque

import numpy as np
import pytest

from aot.capture.buffer import NewestFrameBuffer
from aot.capture.usb import UsbFrameSource
from aot.config.types import CameraSourceConfig
from aot.capture.types import (
    CaptureOpenError,
    CaptureProperties,
    CaptureReadError,
    EndOfStream,
    FramePacket,
    SourceState,
)
from aot.capture.worker import CaptureWorker, ReconnectPolicy
from aot.config.types import SourceType


PROPERTIES = CaptureProperties(1, 1, 15.0, 1, 1, 15.0)


def packet(sequence: int) -> FramePacket:
    return FramePacket("source-1", sequence, sequence, np.zeros((1, 1, 3), np.uint8))


class ScriptedSource:
    def __init__(self, *, source_type=SourceType.OFFLINE, opens=(), reads=()) -> None:
        self.source_id = "source-1"
        self.source_type = source_type
        self.reconnectable = source_type is not SourceType.OFFLINE
        self.state = SourceState.CLOSED
        self.properties = None
        self.opens = deque(opens)
        self.reads = deque(reads)
        self.open_count = 0
        self.close_count = 0
        self.open_thread_id = None
        self.read_thread_id = None

    def open(self):
        self.open_count += 1
        self.open_thread_id = threading.get_ident()
        effect = self.opens.popleft() if self.opens else None
        if isinstance(effect, Exception):
            self.state = SourceState.FAILED
            raise effect
        if callable(effect):
            effect()
        self.state = SourceState.OPEN
        self.properties = PROPERTIES
        return PROPERTIES

    def read(self):
        self.read_thread_id = threading.get_ident()
        effect = self.reads.popleft() if self.reads else EndOfStream(
            self.source_id, "done"
        )
        if callable(effect):
            effect = effect()
        if isinstance(effect, EndOfStream):
            self.state = SourceState.EOF
            raise effect
        if isinstance(effect, Exception):
            self.state = SourceState.FAILED
            raise effect
        return effect

    def close(self):
        self.close_count += 1
        self.state = SourceState.CLOSED


def test_reconnect_policy_doubles_caps_and_resets() -> None:
    policy = ReconnectPolicy(0.25, 5.0)

    assert [policy.next_delay() for _ in range(7)] == [0.25, 0.5, 1.0, 2.0, 4.0, 5.0, 5.0]
    policy.reset()
    assert policy.next_delay() == 0.25


def test_worker_opens_reads_publishes_and_closes_in_its_thread() -> None:
    source = ScriptedSource(reads=[packet(0), EndOfStream("source-1", "done")])
    buffer = NewestFrameBuffer(2)
    worker = CaptureWorker(source, buffer, ReconnectPolicy(0.01, 0.02), 1.0)
    caller_thread = threading.get_ident()

    worker.start()
    assert worker.join(1.0)

    assert buffer.get_latest().sequence == 0
    assert source.open_thread_id == source.read_thread_id
    assert source.open_thread_id != caller_thread
    assert source.close_count == 1
    assert worker.snapshot().state is SourceState.EOF


@pytest.mark.parametrize("source_type", [SourceType.USB, SourceType.RTSP])
def test_live_sources_retry_open_failures_with_bounded_backoff(source_type) -> None:
    stop_after_delays = []

    def wait_for_stop(delay: float) -> bool:
        stop_after_delays.append(delay)
        return len(stop_after_delays) == 7

    source = ScriptedSource(
        source_type=source_type,
        opens=[CaptureOpenError("source-1", "unavailable")] * 7,
    )
    worker = CaptureWorker(
        source,
        NewestFrameBuffer(1),
        ReconnectPolicy(0.25, 5.0),
        1.0,
        wait_for_stop=wait_for_stop,
    )

    worker.start()
    assert worker.join(1.0)

    assert stop_after_delays == [0.25, 0.5, 1.0, 2.0, 4.0, 5.0, 5.0]
    assert source.open_count == 7


def test_successful_frame_resets_reconnect_delay() -> None:
    delays = []

    def wait_for_stop(delay: float) -> bool:
        delays.append(delay)
        return len(delays) == 2

    source = ScriptedSource(
        source_type=SourceType.USB,
        opens=[CaptureOpenError("source-1", "first"), None, CaptureOpenError("source-1", "third")],
        reads=[packet(0), CaptureReadError("source-1", "lost")],
    )
    worker = CaptureWorker(
        source,
        NewestFrameBuffer(1),
        ReconnectPolicy(0.25, 5.0),
        1.0,
        wait_for_stop=wait_for_stop,
    )

    worker.start()
    assert worker.join(1.0)

    assert delays == [0.25, 0.25]


def test_offline_eof_never_retries() -> None:
    source = ScriptedSource(reads=[EndOfStream("source-1", "done")])
    waits = []
    worker = CaptureWorker(
        source,
        NewestFrameBuffer(1),
        ReconnectPolicy(0.25, 5.0),
        1.0,
        wait_for_stop=lambda delay: waits.append(delay) or False,
    )

    worker.start()
    assert worker.join(1.0)

    assert waits == []
    assert source.open_count == 1
    assert worker.snapshot().state is SourceState.EOF


def test_errors_are_contained_and_snapshot_never_exposes_secret() -> None:
    secret = "rtsp" + "://operator:hidden@camera.invalid/live"
    source = ScriptedSource(
        source_type=SourceType.RTSP,
        opens=[CaptureOpenError("source-1", secret)],
    )
    worker = CaptureWorker(
        source,
        NewestFrameBuffer(1),
        ReconnectPolicy(0.01, 0.01),
        1.0,
        wait_for_stop=lambda _delay: True,
    )

    worker.start()
    assert worker.join(1.0)
    snapshot = worker.snapshot()

    assert snapshot.last_error == "Source operation failed"
    assert secret not in repr(snapshot)


@pytest.mark.parametrize("blocked_operation", ["open", "read"])
def test_stop_unblocks_open_or_read_and_closes_once(blocked_operation) -> None:
    entered = threading.Event()
    released = threading.Event()

    def block():
        entered.set()
        assert released.wait(1.0)
        if blocked_operation == "read":
            return packet(0)

    class BlockingSource(ScriptedSource):
        def close(self):
            if self.close_count == 0:
                self.close_count += 1
                released.set()
                self.state = SourceState.CLOSED

    source = BlockingSource(
        source_type=SourceType.USB,
        opens=[block] if blocked_operation == "open" else [],
        reads=[block] if blocked_operation == "read" else [],
    )
    worker = CaptureWorker(source, NewestFrameBuffer(1), ReconnectPolicy(0.1, 0.2), 1.0)
    worker.start()
    assert entered.wait(1.0)

    started = time.monotonic()
    assert worker.stop()

    assert time.monotonic() - started < 1.0
    assert source.close_count == 1


def test_stop_interrupts_backoff() -> None:
    source = ScriptedSource(
        source_type=SourceType.USB,
        opens=[CaptureOpenError("source-1", "unavailable")],
    )
    worker = CaptureWorker(source, NewestFrameBuffer(1), ReconnectPolicy(10.0, 10.0), 1.0)
    worker.start()
    deadline = time.monotonic() + 1.0
    while worker.snapshot().state is not SourceState.RECONNECTING:
        assert time.monotonic() < deadline

    started = time.monotonic()
    assert worker.stop()

    assert time.monotonic() - started < 1.0


def test_snapshot_reports_queue_depth_and_dropped_frames() -> None:
    source = ScriptedSource(
        reads=[packet(0), packet(1), packet(2), EndOfStream("source-1", "done")]
    )
    worker = CaptureWorker(source, NewestFrameBuffer(2), ReconnectPolicy(0.1, 0.2), 1.0)

    worker.start()
    assert worker.join(1.0)
    snapshot = worker.snapshot()

    assert snapshot.queue_depth == 2
    assert snapshot.dropped_frames == 1


def test_stop_during_backend_construction_releases_late_handle() -> None:
    factory_entered = threading.Event()
    factory_released = threading.Event()

    class LateBackend:
        release_count = 0

        def is_opened(self):
            return True

        def set(self, *_args):
            return True

        def get(self, *_args):
            return 1.0

        def read(self):
            return False, None

        def release(self):
            self.release_count += 1

    backend = LateBackend()

    def delayed_factory(_value):
        factory_entered.set()
        assert factory_released.wait(1.0)
        return backend

    source = UsbFrameSource(
        CameraSourceConfig(
            "source-1", "USB", SourceType.USB, device_index=0, width=1, height=1
        ),
        backend_factory=delayed_factory,
    )
    worker = CaptureWorker(source, NewestFrameBuffer(1), ReconnectPolicy(0.1, 0.2), 0.05)
    worker.start()
    assert factory_entered.wait(1.0)

    assert worker.stop() is False
    factory_released.set()
    assert worker.join(1.0)

    assert backend.release_count == 1
    assert source._backend is None


def test_shutdown_timeout_includes_slow_close() -> None:
    read_entered = threading.Event()
    release_read = threading.Event()

    class SlowCloseSource(ScriptedSource):
        def read(self):
            read_entered.set()
            release_read.wait(1.0)
            return packet(0)

        def close(self):
            time.sleep(0.2)
            release_read.set()
            super().close()

    source = SlowCloseSource(source_type=SourceType.USB)
    worker = CaptureWorker(source, NewestFrameBuffer(1), ReconnectPolicy(0.1, 0.2), 0.05)
    worker.start()
    assert read_entered.wait(1.0)

    started = time.monotonic()
    assert worker.stop() is False
    elapsed = time.monotonic() - started

    assert elapsed < 0.12
    assert worker.join(1.0)
