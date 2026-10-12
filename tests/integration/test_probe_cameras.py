import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from aot.capture.types import CaptureProperties, CaptureReadError, FramePacket, SourceState
from aot.config.types import (
    AppSettings,
    CameraSourceConfig,
    ComputeConfig,
    SourceType,
)
from aot.hardware_testing.types import (
    HardwareTestStatus,
    SourceTestResult,
    UsbDeviceCandidate,
)
from scripts.probe_cameras import main


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def camera(source_id, *, enabled=True):
    return CameraSourceConfig(
        source_id,
        source_id,
        SourceType.USB,
        enabled=enabled,
        device_index=0,
    )


def settings(*sources):
    return AppSettings(1, Path("runtime"), "INFO", ComputeConfig(), tuple(sources))


def result(source_id, status=HardwareTestStatus.SUCCESS):
    return SourceTestResult(
        source_id=source_id,
        source_type=SourceType.USB,
        status=status,
        properties=CaptureProperties(640, 480, 15, 640, 480, 15),
        frames_requested=1,
        frames_received=1 if status is HardwareTestStatus.SUCCESS else 0,
        mean_capture_latency_ms=1.0,
        p95_capture_latency_ms=1.0,
        dropped_frames=0,
        started_at_utc=NOW,
        ended_at_utc=NOW,
        summary="Probe complete" if status is HardwareTestStatus.SUCCESS else "Probe unavailable",
    )


def loader_for(value):
    return lambda *_args, **_kwargs: value


def test_human_and_json_output_are_sanitized(capsys) -> None:
    app_settings = settings(camera("usb-1"))
    probe = lambda sources, **_kwargs: tuple(result(item.source_id) for item in sources)

    assert main([], settings_loader=loader_for(app_settings), probe_sources_fn=probe) == 0
    human = capsys.readouterr().out
    assert "[SUCCESS] usb-1: Probe complete" in human

    assert main(["--json"], settings_loader=loader_for(app_settings), probe_sources_fn=probe) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["sources"][0]["source_id"] == "usb-1"
    assert "rtsp://" not in json.dumps(payload)


def test_usb_discovery_and_source_filtering(capsys) -> None:
    app_settings = settings(camera("one"), camera("two"), camera("disabled", enabled=False))
    selected = []

    def probe(sources, **_kwargs):
        selected.extend(item.source_id for item in sources)
        return tuple(result(item.source_id) for item in sources)

    exit_code = main(
        ["--source-id", "two", "--discover-usb", "--max-usb-index", "4", "--json"],
        settings_loader=loader_for(app_settings),
        probe_sources_fn=probe,
        discover_usb_fn=lambda **kwargs: (UsbDeviceCandidate(2, 640, 480, 30),),
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert selected == ["two"]
    assert payload["usb_devices"] == [{"device_index": 2, "width": 640, "height": 480, "fps": 30}]


def test_explicit_filter_can_probe_disabled_configured_source(capsys) -> None:
    app_settings = settings(camera("disabled", enabled=False))
    selected = []

    def probe(sources, **_kwargs):
        selected.extend(sources)
        return (result("disabled"),)

    assert main(
        ["--source-id", "disabled"],
        settings_loader=loader_for(app_settings),
        probe_sources_fn=probe,
    ) == 0
    capsys.readouterr()
    assert [item.source_id for item in selected] == ["disabled"]


def test_no_enabled_source_and_unknown_filter_fail_safely(capsys) -> None:
    app_settings = settings(camera("known", enabled=False))

    assert main([], settings_loader=loader_for(app_settings)) == 1
    assert "No enabled camera sources" in capsys.readouterr().out

    assert main(
        ["--source-id", "missing"],
        settings_loader=loader_for(app_settings),
    ) == 1
    assert "Unknown source ID: missing" in capsys.readouterr().out


class CliProbeSource:
    source_type = SourceType.USB
    reconnectable = True
    state = SourceState.CLOSED
    properties = CaptureProperties(1, 1, 15, 1, 1, 15)

    def __init__(self, source_id, succeeds):
        self.source_id = source_id
        self.succeeds = succeeds
        self.close_count = 0

    def open(self):
        return self.properties

    def read(self):
        if not self.succeeds:
            raise CaptureReadError(self.source_id, "private backend detail")
        return FramePacket(self.source_id, 0, 0, np.zeros((1, 1, 3), np.uint8))

    def close(self):
        self.close_count += 1


def test_one_failed_probe_does_not_skip_cleanup_or_later_sources(capsys) -> None:
    first = CliProbeSource("first", False)
    second = CliProbeSource("second", True)
    source_map = {"first": first, "second": second}

    exit_code = main(
        ["--frames", "1", "--json"],
        settings_loader=loader_for(settings(camera("first"), camera("second"))),
        source_factory=lambda item: source_map[item.source_id],
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert [item["status"] for item in payload["sources"]] == ["failed", "success"]
    assert first.close_count == second.close_count == 1
    assert "private backend detail" not in json.dumps(payload)


@pytest.mark.parametrize(
    "status",
    [HardwareTestStatus.DEGRADED, HardwareTestStatus.FAILED, HardwareTestStatus.UNSUPPORTED],
)
def test_non_success_status_returns_one(status, capsys) -> None:
    code = main(
        [],
        settings_loader=loader_for(settings(camera("one"))),
        probe_sources_fn=lambda _sources, **_kwargs: (result("one", status),),
    )
    capsys.readouterr()
    assert code == 1
