import logging
import traceback

import numpy as np
import pytest

from aot.capture.rtsp import RtspFrameSource
from aot.capture.types import CaptureOpenError, CaptureReadError, SourceState
from aot.config.types import CameraSourceConfig, SourceType

from fakes import FakeCaptureBackend, RecordingBackendFactory


def rtsp_config(**changes) -> CameraSourceConfig:
    values = {
        "source_id": "cctv-1",
        "name": "Entrance",
        "source_type": SourceType.RTSP,
        "rtsp_env_key": "AOT_TEST_CAMERA_URL",
        "width": 4,
        "height": 2,
        "fps": 15.0,
        "rotation": 0,
    }
    values.update(changes)
    return CameraSourceConfig(**values)


def credential_url(password: str = "first") -> str:
    return "rtsp" + "://operator:" + password + "@camera.invalid/live"


def test_rtsp_retains_environment_key_but_never_secret(caplog) -> None:
    secret = credential_url()
    environment = {"AOT_TEST_CAMERA_URL": secret}
    factory = RecordingBackendFactory(FakeCaptureBackend())
    source = RtspFrameSource(rtsp_config(), environment, factory)

    with caplog.at_level(logging.DEBUG):
        source.open()

    assert source.rtsp_env_key == "AOT_TEST_CAMERA_URL"
    assert factory.inputs == [secret]
    assert secret not in repr(source)
    assert secret not in repr(vars(source))
    assert secret not in caplog.text


def test_url_is_resolved_immediately_before_every_open_attempt() -> None:
    environment = {"AOT_TEST_CAMERA_URL": credential_url("first")}
    factory = RecordingBackendFactory(FakeCaptureBackend())
    source = RtspFrameSource(rtsp_config(), environment, factory)

    source.open()
    source.close()
    environment["AOT_TEST_CAMERA_URL"] = credential_url("second")
    source.open()

    assert factory.inputs == [credential_url("first"), credential_url("second")]


def test_backend_exception_is_sanitized_and_backend_reference_is_cleared(caplog) -> None:
    secret = credential_url()

    def failing_factory(source_value):
        raise RuntimeError(f"backend rejected {source_value}")

    source = RtspFrameSource(
        rtsp_config(),
        {"AOT_TEST_CAMERA_URL": secret},
        failing_factory,
    )

    with caplog.at_level(logging.DEBUG), pytest.raises(CaptureOpenError) as caught:
        source.open()

    assert caught.value.summary == "RTSP source could not be opened"
    assert secret not in str(caught.value)
    assert secret not in repr(vars(caught.value))
    assert secret not in repr(vars(source))
    assert secret not in caplog.text
    assert source.state is SourceState.FAILED
    assert secret not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_backend_read_exception_has_no_secret_bearing_exception_chain() -> None:
    secret = credential_url()

    class FailingReadBackend(FakeCaptureBackend):
        def read(self):
            raise RuntimeError(f"backend lost {secret}")

    source = RtspFrameSource(
        rtsp_config(),
        {"AOT_TEST_CAMERA_URL": secret},
        RecordingBackendFactory(FailingReadBackend()),
    )
    source.open()

    with pytest.raises(CaptureReadError) as caught:
        source.read()

    assert secret not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_environment_dependency_does_not_use_a_secret_retaining_closure() -> None:
    environment = {"AOT_TEST_CAMERA_URL": credential_url()}
    source = RtspFrameSource(
        rtsp_config(),
        environment,
        RecordingBackendFactory(FakeCaptureBackend()),
    )

    assert getattr(source._environment_get, "__closure__", None) is None


def test_rtsp_reads_normalized_packets_and_is_reconnectable() -> None:
    frame = np.zeros((1, 2), dtype=np.uint8)
    backend = FakeCaptureBackend(reads=[(True, frame)])
    source = RtspFrameSource(
        rtsp_config(),
        {"AOT_TEST_CAMERA_URL": credential_url()},
        RecordingBackendFactory(backend),
        clock_ns=lambda: 789,
    )
    source.open()

    packet = source.read()

    assert source.reconnectable is True
    assert packet.frame.shape == (2, 4, 3)
    assert packet.captured_at_monotonic_ns == 789


def test_failed_read_releases_and_clears_backend_once() -> None:
    backend = FakeCaptureBackend(reads=[(False, None)])
    source = RtspFrameSource(
        rtsp_config(),
        {"AOT_TEST_CAMERA_URL": credential_url()},
        RecordingBackendFactory(backend),
    )
    source.open()

    with pytest.raises(CaptureReadError):
        source.read()
    source.close()

    assert backend.release_count == 1
    assert source.state is SourceState.CLOSED


def test_unavailable_backend_releases_once_per_acquired_handle() -> None:
    backend = FakeCaptureBackend(opened=False)
    source = RtspFrameSource(
        rtsp_config(),
        {"AOT_TEST_CAMERA_URL": credential_url()},
        RecordingBackendFactory(backend),
    )

    with pytest.raises(CaptureOpenError):
        source.open()
    source.close()

    assert backend.release_count == 1
