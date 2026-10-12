from __future__ import annotations

import numpy as np
import pytest

from aot.capture.manager import ManagerShutdownError
from aot.capture.types import FramePacket, SourceState
from aot.capture.worker import WorkerSnapshot
from aot.config.types import (
    AppSettings,
    CameraSourceConfig,
    ComputeConfig,
    DetectionConfig,
    SourceType,
)
from aot.runtime.live_runtime import (
    LiveRuntime,
    LiveRuntimeConfigError,
    LiveRuntimeShutdownError,
)
from aot.runtime.types import ProcessedFrame


def source(source_id: str, *, enabled: bool = True) -> CameraSourceConfig:
    return CameraSourceConfig(
        source_id, source_id.upper(), SourceType.USB, enabled, device_index=0
    )


def settings(*sources: CameraSourceConfig, detection: bool = False) -> AppSettings:
    return AppSettings(
        1,
        __import__("pathlib").Path("runtime"),
        "INFO",
        ComputeConfig(),
        tuple(sources),
        detection=DetectionConfig(enabled=detection, model_id="model" if detection else None),
    )


class FakeCapture:
    def __init__(self, events: list[str], ids: tuple[str, ...]) -> None:
        self.events = events
        self.ids = ids
        self.stop_error: Exception | None = None

    def start(self, configured_sources) -> None:
        self.events.append("capture.start")
        assert tuple(item.source_id for item in configured_sources) == self.ids

    def latest(self, _source_id: str):
        return None

    def snapshot(self):
        return {
            source_id: WorkerSnapshot(
                source_id, SourceState.OPEN, 0, 0, None, True
            )
            for source_id in self.ids
        }

    def stop(self) -> None:
        self.events.append("capture.stop")
        if self.stop_error is not None:
            raise self.stop_error


class FakeDetector:
    model_id = "model"
    class_names = ()

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.closed = False

    def close(self) -> None:
        self.events.append("detector.close")
        self.closed = True


class FakeProcessing:
    def __init__(self, events: list[str], store) -> None:
        self.events = events
        self.store = store
        self.alive = False
        self.stuck = False

    def start(self) -> None:
        self.events.append("processing.start")
        self.alive = True

    def request_stop(self) -> None:
        self.events.append("processing.request_stop")

    def join(self, _timeout=None) -> bool:
        self.events.append("processing.join")
        if not self.stuck:
            self.alive = False
        return not self.stuck

    @property
    def is_alive(self) -> bool:
        return self.alive


def runtime_with_fakes(app_settings: AppSettings):
    events: list[str] = []
    captures: list[FakeCapture] = []
    detectors: list[FakeDetector] = []
    processors: list[FakeProcessing] = []
    ids = tuple(item.source_id for item in app_settings.sources if item.enabled)

    def capture_factory(_config):
        capture = FakeCapture(events, ids)
        captures.append(capture)
        return capture

    def detector_factory():
        detector = FakeDetector(events)
        detectors.append(detector)
        return detector

    def processing_factory(_ids, _supplier, store, _detector):
        processing = FakeProcessing(events, store)
        processors.append(processing)
        return processing

    runtime = LiveRuntime(
        app_settings,
        capture_factory,
        detector_factory,
        processing_worker_factory=processing_factory,
    )
    return runtime, events, captures, detectors, processors


@pytest.mark.parametrize("count", [0, 4])
def test_invalid_enabled_source_count_fails_before_factories(count: int) -> None:
    calls: list[str] = []
    runtime = LiveRuntime(
        settings(*(source(f"s{index}") for index in range(count))),
        lambda _config: calls.append("capture"),
        lambda: calls.append("detector"),
    )

    with pytest.raises(LiveRuntimeConfigError):
        runtime.start()
    assert calls == []


def test_disabled_detection_does_not_invoke_detector_factory() -> None:
    runtime, events, _captures, detectors, _processors = runtime_with_fakes(
        settings(source("a"), source("disabled", enabled=False))
    )
    runtime.start()

    assert detectors == []
    assert events[:2] == ["capture.start", "processing.start"]
    runtime.stop()


def test_start_preserves_order_builds_snapshot_and_rejects_second_start() -> None:
    runtime, events, _captures, detectors, processors = runtime_with_fakes(
        settings(source("b"), source("a"), detection=True)
    )
    runtime.start()
    packet = FramePacket("b", 3, 9, np.zeros((1, 1, 3), dtype=np.uint8))
    processors[0].store.put(ProcessedFrame(packet, None, None))

    snapshot = runtime.snapshot()
    assert runtime.running is True
    assert [item.source_id for item in snapshot.sources] == ["b", "a"]
    assert [item.display_name for item in snapshot.sources] == ["B", "A"]
    assert snapshot.sources[0].result.packet.sequence == 3
    assert events.index("capture.start") < events.index("processing.start")
    assert len(detectors) == 1
    with pytest.raises(RuntimeError, match="already running"):
        runtime.start()
    runtime.stop()


def test_stop_order_restart_and_repeated_stop_are_safe() -> None:
    runtime, events, captures, detectors, processors = runtime_with_fakes(
        settings(source("a"), detection=True)
    )
    runtime.start()
    runtime.stop()

    assert events[-4:] == [
        "processing.request_stop",
        "processing.join",
        "capture.stop",
        "detector.close",
    ]
    runtime.stop()
    assert events.count("detector.close") == 1

    runtime.start()
    assert len(captures) == len(detectors) == len(processors) == 2
    runtime.stop()


def test_stuck_processing_still_stops_capture_and_does_not_close_detector() -> None:
    runtime, events, _captures, detectors, processors = runtime_with_fakes(
        settings(source("a"), source("b"), detection=True)
    )
    runtime.start()
    processors[0].stuck = True

    with pytest.raises(LiveRuntimeShutdownError) as caught:
        runtime.stop()

    assert caught.value.source_ids == ("a", "b")
    assert events[-1] == "capture.stop"
    assert detectors[0].closed is False
    assert "PRIVATE" not in str(caught.value)

    processors[0].stuck = False
    runtime.stop()
    assert detectors[0].closed is True


def test_capture_shutdown_error_is_sanitized_after_other_cleanup() -> None:
    runtime, events, captures, detectors, _processors = runtime_with_fakes(
        settings(source("a"), source("b"), detection=True)
    )
    runtime.start()
    captures[0].stop_error = ManagerShutdownError(["b"])

    with pytest.raises(LiveRuntimeShutdownError) as caught:
        runtime.stop()

    assert caught.value.source_ids == ("b",)
    assert detectors[0].closed is True
    assert events.index("processing.join") < events.index("capture.stop")
    assert events.index("capture.stop") < events.index("detector.close")
