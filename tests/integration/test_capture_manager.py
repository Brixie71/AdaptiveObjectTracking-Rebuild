import threading
import time

import numpy as np
import pytest

from aot.capture.manager import CaptureManager, ManagerShutdownError
from aot.capture.types import (
    CaptureOpenError,
    CaptureProperties,
    EndOfStream,
    FramePacket,
    SourceState,
)
from aot.config.types import CameraSourceConfig, CaptureRuntimeConfig, SourceType


PROPERTIES = CaptureProperties(1, 1, 15.0, 1, 1, 15.0)


def config(source_id: str, source_type: SourceType, *, enabled=True):
    fields = {}
    if source_type is SourceType.USB:
        fields["device_index"] = 0
    elif source_type is SourceType.RTSP:
        fields["rtsp_env_key"] = "CAMERA_URL"
    else:
        fields["offline_path"] = __file__
    return CameraSourceConfig(source_id, source_id, source_type, enabled=enabled, **fields)


class IntegrationSource:
    def __init__(self, source_id: str, mode: str, source_type: SourceType) -> None:
        self.source_id = source_id
        self.source_type = source_type
        self.reconnectable = source_type is not SourceType.OFFLINE
        self.state = SourceState.CLOSED
        self.properties = None
        self.mode = mode
        self.open_count = 0
        self.close_count = 0
        self.sequence = 0
        self.closed = threading.Event()

    def open(self):
        self.open_count += 1
        self.closed.clear()
        if self.mode == "fail":
            self.state = SourceState.FAILED
            raise CaptureOpenError(self.source_id, "unavailable")
        self.state = SourceState.OPEN
        self.properties = PROPERTIES
        return PROPERTIES

    def read(self):
        if self.mode == "offline" and self.sequence > 0:
            self.state = SourceState.EOF
            raise EndOfStream(self.source_id, "done")
        if self.mode == "healthy":
            time.sleep(0.001)
        packet = FramePacket(
            self.source_id,
            self.sequence,
            time.monotonic_ns(),
            np.zeros((1, 1, 3), np.uint8),
        )
        self.sequence += 1
        return packet

    def close(self):
        if not self.closed.is_set():
            self.close_count += 1
            self.closed.set()
        self.state = SourceState.CLOSED


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached")


def test_sources_run_independently_and_disabled_sources_are_skipped() -> None:
    configs = [
        config("healthy", SourceType.USB),
        config("failing", SourceType.RTSP),
        config("offline", SourceType.OFFLINE),
        config("disabled", SourceType.USB, enabled=False),
    ]
    sources = {
        "healthy": IntegrationSource("healthy", "healthy", SourceType.USB),
        "failing": IntegrationSource("failing", "fail", SourceType.RTSP),
        "offline": IntegrationSource("offline", "offline", SourceType.OFFLINE),
    }
    factory_calls = []

    def factory(source_config):
        factory_calls.append(source_config.source_id)
        return sources[source_config.source_id]

    manager = CaptureManager(
        CaptureRuntimeConfig(
            queue_capacity=2,
            reconnect_initial_seconds=0.01,
            reconnect_max_seconds=0.02,
            shutdown_timeout_seconds=1.0,
        ),
        factory,
    )
    manager.start(configs)
    wait_until(lambda: sources["healthy"].sequence >= 3)
    wait_until(lambda: sources["failing"].open_count >= 2)
    wait_until(lambda: manager.snapshot()["offline"].state is SourceState.EOF)

    snapshots = manager.snapshot()
    assert factory_calls == ["healthy", "failing", "offline"]
    assert snapshots["healthy"].state is SourceState.OPEN
    assert snapshots["failing"].state is SourceState.RECONNECTING
    assert snapshots["offline"].state is SourceState.EOF
    assert manager.latest("healthy").source_id == "healthy"
    assert manager.latest("offline").source_id == "offline"
    with pytest.raises(KeyError):
        manager.latest("disabled")

    with pytest.raises(RuntimeError, match="already started"):
        manager.start(configs)

    manager.stop()
    manager.stop()
    assert not any(snapshot.alive for snapshot in manager.snapshot().values())
    assert all(source.close_count >= 1 for source in sources.values())


def test_shutdown_error_names_only_stuck_source_and_stops_others() -> None:
    entered = threading.Event()
    release = threading.Event()

    class StubbornSource(IntegrationSource):
        def open(self):
            entered.set()
            release.wait(1.0)
            return super().open()

        def close(self):
            self.close_count += 1

    stubborn = StubbornSource("stuck", "healthy", SourceType.USB)
    normal = IntegrationSource("normal", "healthy", SourceType.USB)
    sources = {"stuck": stubborn, "normal": normal}
    manager = CaptureManager(
        CaptureRuntimeConfig(shutdown_timeout_seconds=0.05),
        lambda item: sources[item.source_id],
    )
    manager.start([config("stuck", SourceType.USB), config("normal", SourceType.USB)])
    assert entered.wait(1.0)

    with pytest.raises(ManagerShutdownError) as caught:
        manager.stop()

    assert caught.value.source_ids == ("stuck",)
    assert "CAMERA_URL" not in str(caught.value)
    assert not manager.snapshot()["normal"].alive

    release.set()
    wait_until(lambda: not manager.snapshot()["stuck"].alive)
    manager.stop()
