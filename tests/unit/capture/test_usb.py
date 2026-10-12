import numpy as np
import pytest

from aot.capture.backend import BackendProperty
from aot.capture.types import CaptureOpenError, CaptureReadError, SourceState
from aot.capture.usb import UsbFrameSource
from aot.config.types import CameraSourceConfig, SourceType

from fakes import FakeCaptureBackend, RecordingBackendFactory


def usb_config(**changes) -> CameraSourceConfig:
    values = {
        "source_id": "usb-1",
        "name": "Workbench camera",
        "source_type": SourceType.USB,
        "device_index": 3,
        "width": 4,
        "height": 2,
        "fps": 20.0,
        "rotation": 0,
    }
    values.update(changes)
    return CameraSourceConfig(**values)


def test_usb_source_accepts_only_complete_usb_configuration() -> None:
    factory = RecordingBackendFactory(FakeCaptureBackend())

    with pytest.raises(ValueError):
        UsbFrameSource(usb_config(source_type=SourceType.RTSP), factory)
    with pytest.raises(ValueError):
        UsbFrameSource(usb_config(device_index=None), factory)


def test_open_configures_backend_before_reading_negotiated_properties() -> None:
    negotiated = {
        BackendProperty.WIDTH: 1280.0,
        BackendProperty.HEIGHT: 720.0,
        BackendProperty.FPS: 30.0,
    }
    backend = FakeCaptureBackend(negotiated=negotiated)
    factory = RecordingBackendFactory(backend)
    source = UsbFrameSource(usb_config(), factory)

    properties = source.open()

    assert factory.inputs == [3]
    assert backend.events == [
        ("is_opened",),
        ("set", BackendProperty.WIDTH, 4.0),
        ("set", BackendProperty.HEIGHT, 2.0),
        ("set", BackendProperty.FPS, 20.0),
        ("get", BackendProperty.WIDTH),
        ("get", BackendProperty.HEIGHT),
        ("get", BackendProperty.FPS),
    ]
    assert properties.negotiated_width == 1280
    assert properties.negotiated_height == 720
    assert properties.negotiated_fps == 30.0
    assert source.state is SourceState.OPEN


def test_open_failure_is_sanitized_and_releases_backend() -> None:
    backend = FakeCaptureBackend(opened=False)
    source = UsbFrameSource(usb_config(), RecordingBackendFactory(backend))

    with pytest.raises(CaptureOpenError) as caught:
        source.open()

    assert caught.value.source_id == "usb-1"
    assert "3" not in caught.value.summary
    assert source.state is SourceState.FAILED
    assert backend.release_count == 1


def test_reads_are_normalized_sequenced_and_timestamped() -> None:
    frame = np.zeros((1, 2), dtype=np.uint8)
    frame[:, 1] = 255
    backend = FakeCaptureBackend(reads=[(True, frame), (True, frame)])
    timestamps = iter([100, 200])
    source = UsbFrameSource(
        usb_config(),
        RecordingBackendFactory(backend),
        clock_ns=lambda: next(timestamps),
    )
    source.open()

    first = source.read()
    second = source.read()

    assert (first.sequence, second.sequence) == (0, 1)
    assert (first.captured_at_monotonic_ns, second.captured_at_monotonic_ns) == (100, 200)
    assert first.source_id == "usb-1"
    assert first.frame.shape == (2, 4, 3)
    assert first.frame.dtype == np.uint8


@pytest.mark.parametrize(
    "backend_result",
    [(False, None), (True, None), (True, np.array([], dtype=np.uint8))],
)
def test_failed_or_empty_reads_raise_sanitized_error(backend_result) -> None:
    backend = FakeCaptureBackend(reads=[backend_result])
    source = UsbFrameSource(usb_config(), RecordingBackendFactory(backend))
    source.open()

    with pytest.raises(CaptureReadError) as caught:
        source.read()

    assert caught.value.source_id == "usb-1"
    assert vars(caught.value).keys() == {"source_id", "summary"}
    assert source.state is SourceState.FAILED


def test_close_is_idempotent_and_releases_acquired_backend_once() -> None:
    backend = FakeCaptureBackend()
    source = UsbFrameSource(usb_config(), RecordingBackendFactory(backend))
    source.open()

    source.close()
    source.close()

    assert backend.release_count == 1
    assert source.state is SourceState.CLOSED
