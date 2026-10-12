from dataclasses import FrozenInstanceError

import pytest

from aot.config.types import (
    AppSettings,
    CameraSourceConfig,
    ComputeConfig,
    ComputePreference,
    DetectionConfig,
    SourceType,
)


def test_camera_source_defaults_are_canonical() -> None:
    source = CameraSourceConfig(
        source_id="camera-1",
        name="Camera 1",
        source_type=SourceType.USB,
        device_index=0,
    )

    assert (source.width, source.height, source.fps, source.rotation) == (
        640,
        480,
        15.0,
        0,
    )
    assert source.enabled is False


@pytest.mark.parametrize(
    ("enum_type", "valid_values"),
    [
        (SourceType, {"usb", "rtsp", "offline"}),
        (ComputePreference, {"auto", "cpu", "cuda"}),
    ],
)
def test_configuration_enums_accept_only_supported_values(
    enum_type: type[SourceType] | type[ComputePreference],
    valid_values: set[str],
) -> None:
    assert {member.value for member in enum_type} == valid_values

    with pytest.raises(ValueError):
        enum_type("unsupported")


def test_configuration_records_are_immutable(tmp_path) -> None:
    compute = ComputeConfig()
    detection = DetectionConfig()
    source = CameraSourceConfig(
        source_id="camera-1",
        name="Camera 1",
        source_type=SourceType.USB,
        device_index=0,
    )
    settings = AppSettings(
        schema_version=1,
        runtime_dir=tmp_path,
        log_level="INFO",
        compute=compute,
        sources=(source,),
        detection=detection,
    )

    for record, attribute in (
        (compute, "cuda_index"),
        (detection, "enabled"),
        (source, "enabled"),
        (settings, "log_level"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(record, attribute, None)
