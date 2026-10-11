from __future__ import annotations

import os
from pathlib import Path
from textwrap import dedent

import pytest

from aot.config.loader import ConfigError, load_settings, resolve_rtsp_url
from aot.config.types import CameraSourceConfig, SourceType


VALID_APP = """\
schema_version: 1
runtime_dir: runtime
log_level: INFO
compute:
  preference: auto
  cuda_index: 0
"""

VALID_CAMERAS = """\
schema_version: 1
sources:
  - source_id: usb-1
    name: USB 1
    type: usb
    enabled: true
    device_index: 0
  - source_id: cctv-1
    name: CCTV 1
    type: rtsp
    rtsp_env_key: CAMERA_1_URL
    width: 1280
    height: 720
    fps: 25
    rotation: 90
  - source_id: demo-1
    name: Offline demo
    type: offline
    offline_path: media/demo.mp4
"""


def _write_configs(
    root: Path,
    *,
    app: str = VALID_APP,
    cameras: str = VALID_CAMERAS,
) -> tuple[Path, Path]:
    app_path = root / "app.yaml"
    cameras_path = root / "cameras.yaml"
    app_path.write_text(dedent(app), encoding="utf-8")
    cameras_path.write_text(dedent(cameras), encoding="utf-8")
    return app_path, cameras_path


def test_load_settings_parses_all_sources_and_resolves_relative_paths(tmp_path) -> None:
    app_path, cameras_path = _write_configs(tmp_path)

    settings = load_settings(app_path, cameras_path)

    assert settings.schema_version == 1
    assert settings.runtime_dir == tmp_path / "runtime"
    assert settings.compute.preference.value == "auto"
    assert [source.source_type for source in settings.sources] == [
        SourceType.USB,
        SourceType.RTSP,
        SourceType.OFFLINE,
    ]
    assert settings.sources[0].device_index == 0
    assert settings.sources[1].rtsp_env_key == "CAMERA_1_URL"
    assert settings.sources[1].width == 1280
    assert settings.sources[1].rotation == 90
    assert settings.sources[2].offline_path == tmp_path / "media" / "demo.mp4"


@pytest.mark.parametrize(
    ("app", "cameras", "unknown_key"),
    [
        (VALID_APP + "unexpected: true\n", VALID_CAMERAS, "unexpected"),
        (
            VALID_APP.replace("  cuda_index: 0", "  cuda_index: 0\n  mystery: 1"),
            VALID_CAMERAS,
            "mystery",
        ),
        (
            VALID_APP,
            VALID_CAMERAS.replace("    device_index: 0", "    device_index: 0\n    extra: value"),
            "extra",
        ),
    ],
)
def test_unknown_keys_report_path_and_key(
    tmp_path,
    app: str,
    cameras: str,
    unknown_key: str,
) -> None:
    app_path, cameras_path = _write_configs(tmp_path, app=app, cameras=cameras)

    with pytest.raises(ConfigError) as error:
        load_settings(app_path, cameras_path)

    assert unknown_key in str(error.value)
    assert str(app_path if unknown_key != "extra" else cameras_path) in str(error.value)


@pytest.mark.parametrize(
    "sources",
    [
        """
        - source_id: ""
          name: Blank
          type: usb
          device_index: 0
        """,
        """
        - source_id: duplicate
          name: First
          type: usb
          device_index: 0
        - source_id: duplicate
          name: Second
          type: usb
          device_index: 1
        """,
    ],
)
def test_source_ids_must_be_nonblank_and_unique(tmp_path, sources: str) -> None:
    cameras = "schema_version: 1\nsources:\n" + dedent(sources)
    app_path, cameras_path = _write_configs(tmp_path, cameras=cameras)

    with pytest.raises(ConfigError):
        load_settings(app_path, cameras_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("width", 0), ("height", -1), ("fps", 0)],
)
def test_capture_dimensions_and_fps_must_be_positive(
    tmp_path,
    field: str,
    value: int,
) -> None:
    cameras = VALID_CAMERAS.replace(
        "    device_index: 0",
        f"    device_index: 0\n    {field}: {value}",
    )
    app_path, cameras_path = _write_configs(tmp_path, cameras=cameras)

    with pytest.raises(ConfigError) as error:
        load_settings(app_path, cameras_path)

    assert field in str(error.value)


def test_rotation_must_be_a_supported_quarter_turn(tmp_path) -> None:
    cameras = VALID_CAMERAS.replace("    device_index: 0", "    device_index: 0\n    rotation: 45")
    app_path, cameras_path = _write_configs(tmp_path, cameras=cameras)

    with pytest.raises(ConfigError) as error:
        load_settings(app_path, cameras_path)

    assert "rotation" in str(error.value)


@pytest.mark.parametrize(
    "source",
    [
        "source_id: usb\nname: USB\ntype: usb",
        "source_id: rtsp\nname: RTSP\ntype: rtsp",
        "source_id: offline\nname: Offline\ntype: offline",
        "source_id: usb\nname: USB\ntype: usb\ndevice_index: 0\nrtsp_env_key: CAMERA_1_URL",
    ],
)
def test_source_specific_fields_are_required_and_exclusive(tmp_path, source: str) -> None:
    indented = "\n".join(f"    {line}" for line in source.splitlines())
    cameras = f"schema_version: 1\nsources:\n  - {indented.lstrip()}\n"
    app_path, cameras_path = _write_configs(tmp_path, cameras=cameras)

    with pytest.raises(ConfigError):
        load_settings(app_path, cameras_path)


@pytest.mark.parametrize("document", ["app", "cameras"])
def test_only_schema_version_one_is_supported(tmp_path, document: str) -> None:
    app = VALID_APP.replace("schema_version: 1", "schema_version: 2") if document == "app" else VALID_APP
    cameras = (
        VALID_CAMERAS.replace("schema_version: 1", "schema_version: 2")
        if document == "cameras"
        else VALID_CAMERAS
    )
    app_path, cameras_path = _write_configs(tmp_path, app=app, cameras=cameras)

    with pytest.raises(ConfigError) as error:
        load_settings(app_path, cameras_path)

    assert "schema_version" in str(error.value)


def test_environment_file_does_not_override_existing_value(tmp_path, monkeypatch) -> None:
    app_path, cameras_path = _write_configs(tmp_path)
    env_path = tmp_path / ".env"
    env_path.write_text("CAMERA_1_URL=rtsp://from-file\n", encoding="utf-8")
    monkeypatch.setenv("CAMERA_1_URL", "rtsp://already-set")

    load_settings(app_path, cameras_path, env_path)

    assert os.environ["CAMERA_1_URL"] == "rtsp://already-set"


def test_resolve_rtsp_url_returns_only_the_requested_environment_value() -> None:
    source = CameraSourceConfig(
        source_id="cctv-1",
        name="CCTV 1",
        source_type=SourceType.RTSP,
        rtsp_env_key="CAMERA_1_URL",
    )

    assert resolve_rtsp_url(source, {"CAMERA_1_URL": "rtsp://camera/live"}) == "rtsp://camera/live"


def test_missing_rtsp_environment_value_does_not_leak_other_values(monkeypatch) -> None:
    monkeypatch.setenv("UNRELATED_SECRET", "do-not-print")
    source = CameraSourceConfig(
        source_id="cctv-1",
        name="CCTV 1",
        source_type=SourceType.RTSP,
        rtsp_env_key="CAMERA_1_URL",
    )

    with pytest.raises(ConfigError) as error:
        resolve_rtsp_url(source, {})

    assert "CAMERA_1_URL" in str(error.value)
    assert "do-not-print" not in str(error.value)
