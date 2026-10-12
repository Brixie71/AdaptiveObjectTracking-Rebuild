import json
from collections import deque

import numpy as np
import pytest

from aot.capture.backend import BackendProperty
from aot.capture.types import (
    CaptureOpenError,
    CaptureProperties,
    CaptureReadError,
    EndOfStream,
    FramePacket,
    SourceState,
)
from aot.config.types import CameraSourceConfig, SourceType
from aot.hardware_testing.source_probe import probe_source, probe_sources
from aot.hardware_testing.types import HardwareTestStatus
from aot.hardware_testing.usb_discovery import discover_usb_devices


PROPERTIES = CaptureProperties(640, 480, 15.0, 1280, 720, 30.0)


def packet(sequence: int) -> FramePacket:
    return FramePacket("camera-1", sequence, sequence, np.zeros((1, 1, 3), np.uint8))


class ProbeSource:
    source_id = "camera-1"
    source_type = SourceType.RTSP
    reconnectable = True
    state = SourceState.CLOSED
    properties = None

    def __init__(self, *, open_error=None, reads=()) -> None:
        self.open_error = open_error
        self.reads = deque(reads)
        self.close_count = 0

    def open(self):
        if self.open_error:
            raise self.open_error
        self.state = SourceState.OPEN
        self.properties = PROPERTIES
        return PROPERTIES

    def read(self):
        effect = self.reads.popleft()
        if isinstance(effect, Exception):
            raise effect
        return effect

    def close(self):
        self.close_count += 1
        self.state = SourceState.CLOSED


@pytest.mark.parametrize("frame_count", [0, -1, True, 1.5])
def test_probe_requires_positive_integer_frame_count(frame_count) -> None:
    with pytest.raises(ValueError):
        probe_source(ProbeSource(), frame_count=frame_count)


def test_complete_probe_reports_metrics_properties_and_success() -> None:
    source = ProbeSource(reads=[packet(0), packet(2), packet(3)])
    clock_values = iter([0, 1_000_000, 1_000_000, 4_000_000, 4_000_000, 14_000_000])

    result = probe_source(source, frame_count=3, clock_ns=lambda: next(clock_values))
    serialized = result.to_dict()

    assert result.status is HardwareTestStatus.SUCCESS
    assert result.frames_requested == 3
    assert result.frames_received == 3
    assert result.mean_capture_latency_ms == pytest.approx(14 / 3)
    assert result.p95_capture_latency_ms == 10.0
    assert result.dropped_frames == 1
    assert serialized["source_id"] == "camera-1"
    assert serialized["source_type"] == "rtsp"
    assert serialized["requested_properties"] == {"width": 640, "height": 480, "fps": 15.0}
    assert serialized["negotiated_properties"] == {"width": 1280, "height": 720, "fps": 30.0}
    assert serialized["started_at_utc"].endswith("+00:00")
    assert serialized["ended_at_utc"].endswith("+00:00")
    assert source.close_count == 1


def test_partial_capture_failure_is_degraded_and_sanitized() -> None:
    secret = "rtsp" + "://operator:probe-password@camera.invalid/live?token=hidden"
    source = ProbeSource(
        reads=[packet(0), CaptureReadError("camera-1", secret)]
    )
    clock_values = iter([0, 1_000_000, 1_000_000, 2_000_000])

    result = probe_source(source, frame_count=3, clock_ns=lambda: next(clock_values))

    assert result.status is HardwareTestStatus.DEGRADED
    assert result.frames_received == 1
    assert result.summary == "Source probe ended before all frames were captured"
    assert secret not in json.dumps(result.to_dict())
    assert "operator" not in json.dumps(result.to_dict())
    assert "probe-password" not in json.dumps(result.to_dict())
    assert "hidden" not in json.dumps(result.to_dict())
    assert source.close_count == 1


@pytest.mark.parametrize(
    ("source", "expected_summary"),
    [
        (
            ProbeSource(open_error=CaptureOpenError("camera-1", "secret input")),
            "Source could not be opened",
        ),
        (
            ProbeSource(reads=[CaptureReadError("camera-1", "no frame")]),
            "Source probe captured no frames",
        ),
        (
            ProbeSource(reads=[EndOfStream("camera-1", "early eof")]),
            "Source probe captured no frames",
        ),
    ],
)
def test_open_failure_zero_frames_and_early_eof_are_failed(source, expected_summary) -> None:
    result = probe_source(source, frame_count=1, clock_ns=lambda: 0)

    assert result.status is HardwareTestStatus.FAILED
    assert result.summary == expected_summary
    assert source.close_count == 1


def test_unsupported_source_construction_is_reported_without_exception_details() -> None:
    config = CameraSourceConfig("future", "Future", SourceType.USB, device_index=0)
    secret = "rtsp" + "://operator:factory-password@camera.invalid/live"

    results = probe_sources(
        [config],
        source_factory=lambda _config: (_ for _ in ()).throw(RuntimeError(secret)),
        frame_count=1,
    )

    assert results[0].status is HardwareTestStatus.UNSUPPORTED
    assert results[0].summary == "Source type is unsupported"
    assert secret not in json.dumps(results[0].to_dict())


class DiscoveryBackend:
    def __init__(self, index: int, opened: bool) -> None:
        self.index = index
        self.opened = opened
        self.release_count = 0

    def is_opened(self):
        return self.opened

    def get(self, prop):
        return {
            BackendProperty.WIDTH: 640 + self.index,
            BackendProperty.HEIGHT: 480,
            BackendProperty.FPS: 30,
        }[prop]

    def release(self):
        self.release_count += 1


def test_usb_discovery_probes_range_returns_safe_properties_and_releases_all() -> None:
    backends = []

    def factory(index):
        backend = DiscoveryBackend(index, opened=index in {0, 2})
        backends.append(backend)
        return backend

    candidates = discover_usb_devices(max_index=3, backend_factory=factory)

    assert [backend.index for backend in backends] == [0, 1, 2]
    assert [candidate.device_index for candidate in candidates] == [0, 2]
    assert candidates[1].width == 642
    assert all(backend.release_count == 1 for backend in backends)
    assert set(candidates[0].to_dict()) == {"device_index", "width", "height", "fps"}


@pytest.mark.parametrize("max_index", [-1, True, 1.5])
def test_usb_discovery_requires_nonnegative_integer_limit(max_index) -> None:
    with pytest.raises(ValueError):
        discover_usb_devices(max_index=max_index)
