from pathlib import Path

import pytest

from aot.capture.factory import create_frame_source
from aot.capture.offline import OfflineFrameSource
from aot.capture.rtsp import RtspFrameSource
from aot.capture.types import CaptureError
from aot.capture.usb import UsbFrameSource
from aot.config.types import CameraSourceConfig, SourceType

from fakes import FakeCaptureBackend, RecordingBackendFactory


@pytest.mark.parametrize(
    ("config", "expected_type"),
    [
        (
            CameraSourceConfig("usb", "USB", SourceType.USB, device_index=0),
            UsbFrameSource,
        ),
        (
            CameraSourceConfig(
                "rtsp", "RTSP", SourceType.RTSP, rtsp_env_key="CAMERA_URL"
            ),
            RtspFrameSource,
        ),
        (
            CameraSourceConfig(
                "offline",
                "Offline",
                SourceType.OFFLINE,
                offline_path=Path("C:/video.mp4"),
            ),
            OfflineFrameSource,
        ),
    ],
)
def test_factory_selects_exact_adapter_and_propagates_injections(config, expected_type) -> None:
    backend_factory = RecordingBackendFactory(FakeCaptureBackend())
    clock = lambda: 123
    environment = {"CAMERA_URL": credential_url()}

    source = create_frame_source(
        config,
        environment=environment,
        backend_factory=backend_factory,
        clock_ns=clock,
    )

    assert type(source) is expected_type
    assert source._backend_factory is backend_factory
    assert source._clock_ns is clock


def credential_url() -> str:
    return "rtsp" + "://operator:factory-secret@camera.invalid/live"


@pytest.mark.parametrize(
    "config",
    [
        CameraSourceConfig("usb", "USB", SourceType.USB),
        CameraSourceConfig(
            "usb-extra",
            "USB",
            SourceType.USB,
            device_index=0,
            rtsp_env_key="CAMERA_URL",
        ),
        CameraSourceConfig("rtsp", "RTSP", SourceType.RTSP),
        CameraSourceConfig(
            "rtsp-extra",
            "RTSP",
            SourceType.RTSP,
            device_index=0,
            rtsp_env_key="CAMERA_URL",
        ),
        CameraSourceConfig("offline", "Offline", SourceType.OFFLINE),
        CameraSourceConfig(
            "offline-extra",
            "Offline",
            SourceType.OFFLINE,
            device_index=0,
            offline_path=Path("C:/video.mp4"),
        ),
    ],
)
def test_factory_rejects_missing_or_inconsistent_source_fields(config) -> None:
    with pytest.raises(CaptureError):
        create_frame_source(config)


def test_unknown_future_source_type_raises_project_owned_error() -> None:
    config = CameraSourceConfig(
        "future",
        "Future",
        "future-source",  # type: ignore[arg-type]
    )

    with pytest.raises(CaptureError) as caught:
        create_frame_source(config)

    assert caught.value.source_id == "future"
    assert caught.value.summary == "Unsupported source type"
