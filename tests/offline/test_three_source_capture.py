import time
from pathlib import Path

from aot.capture.backend import OpenCvCaptureBackend
from aot.capture.manager import CaptureManager
from aot.capture.offline import OfflineFrameSource
from aot.capture.types import SourceState
from aot.config.types import CameraSourceConfig, CaptureRuntimeConfig, SourceType


FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "media"
    / "gate2-three-frame.avi"
)


class ObservedBackend:
    def __init__(self, source) -> None:
        self.delegate = OpenCvCaptureBackend(source)
        self.release_count = 0

    def is_opened(self):
        return self.delegate.is_opened()

    def set(self, property_name, value):
        return self.delegate.set(property_name, value)

    def get(self, property_name):
        return self.delegate.get(property_name)

    def read(self):
        time.sleep(0.01)
        return self.delegate.read()

    def release(self):
        self.release_count += 1
        self.delegate.release()


def test_three_offline_sources_reach_eof_independently_and_shutdown_cleanly() -> None:
    configs = tuple(
        CameraSourceConfig(
            source_id=f"offline-{index}",
            name=f"Offline {index}",
            source_type=SourceType.OFFLINE,
            enabled=True,
            offline_path=FIXTURE,
            width=320,
            height=240,
            fps=5.0,
            rotation=0,
        )
        for index in range(1, 4)
    )
    backends = []

    def source_factory(config):
        def backend_factory(value):
            backend = ObservedBackend(value)
            backends.append(backend)
            return backend

        return OfflineFrameSource(config, backend_factory=backend_factory)

    manager = CaptureManager(
        CaptureRuntimeConfig(queue_capacity=2, shutdown_timeout_seconds=1.0),
        source_factory,
    )
    observed_states = {config.source_id: set() for config in configs}
    manager.start(configs)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        snapshots = manager.snapshot()
        for source_id, snapshot in snapshots.items():
            observed_states[source_id].add(snapshot.state)
        if all(snapshot.state is SourceState.EOF for snapshot in snapshots.values()):
            break
        time.sleep(0.001)
    else:
        raise AssertionError("offline sources did not all reach EOF")

    packets = {config.source_id: manager.latest(config.source_id) for config in configs}
    assert all(SourceState.OPEN in states for states in observed_states.values())
    assert all(SourceState.EOF in states for states in observed_states.values())
    assert set(packets) == {"offline-1", "offline-2", "offline-3"}
    assert all(packet is not None for packet in packets.values())
    assert all(packet.source_id == source_id for source_id, packet in packets.items())
    assert all(packet.frame.shape == (240, 320, 3) for packet in packets.values())

    manager.stop()

    assert not any(snapshot.alive for snapshot in manager.snapshot().values())
    assert len(backends) == 3
    assert all(backend.release_count == 1 for backend in backends)
