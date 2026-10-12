"""Opt-in acceptance checks for authorized USB and RTSP camera devices."""

from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from aot.capture.factory import create_frame_source
from aot.capture.manager import CaptureManager
from aot.config.loader import load_settings
from aot.config.types import SourceType
from aot.hardware_testing.source_probe import probe_source
from aot.hardware_testing.types import HardwareTestStatus


pytestmark = pytest.mark.hardware

if os.environ.get("AOT_RUN_HARDWARE_TESTS") != "1":
    pytest.skip(
        "Set AOT_RUN_HARDWARE_TESTS=1 to run authorized camera checks",
        allow_module_level=True,
    )


def _selected_settings_and_sources():
    camera_path_text = os.environ.get("AOT_HARDWARE_CAMERA_CONFIG")
    source_ids_text = os.environ.get("AOT_HARDWARE_SOURCE_IDS")
    assert camera_path_text, "AOT_HARDWARE_CAMERA_CONFIG is required"
    assert source_ids_text, "AOT_HARDWARE_SOURCE_IDS is required"
    settings = load_settings(
        Path("config/app.example.yaml"),
        Path(camera_path_text),
    )
    configured = {source.source_id: source for source in settings.sources}
    source_ids = tuple(item.strip() for item in source_ids_text.split(",") if item.strip())
    assert source_ids, "At least one hardware source ID is required"
    assert all(source_id in configured for source_id in source_ids), "Unknown hardware source ID"
    selected = tuple(configured[source_id] for source_id in source_ids)
    assert all(
        source.source_type in {SourceType.USB, SourceType.RTSP}
        for source in selected
    ), "Hardware checks accept only USB and RTSP sources"
    return settings, selected


def test_selected_camera_sources_open_read_report_properties_and_close() -> None:
    _settings, selected = _selected_settings_and_sources()
    frame_count = int(os.environ.get("AOT_HARDWARE_FRAME_COUNT", "30"))

    for config in selected:
        result = probe_source(create_frame_source(config), frame_count=frame_count)
        assert result.status is HardwareTestStatus.SUCCESS, (
            f"Hardware source {config.source_id} did not pass: {result.summary}"
        )
        assert result.frames_received == frame_count
        assert result.properties is not None
        assert result.properties.negotiated_width > 0
        assert result.properties.negotiated_height > 0


def test_three_configured_sources_run_and_shutdown_independently() -> None:
    settings, selected = _selected_settings_and_sources()
    if len(selected) < 3:
        pytest.skip("Three configured source IDs are required for concurrency acceptance")
    selected = tuple(replace(source, enabled=True) for source in selected[:3])
    manager = CaptureManager(settings.capture)
    manager.start(selected)
    seen = set()
    timeout = float(os.environ.get("AOT_HARDWARE_TIMEOUT_SECONDS", "15"))
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline and len(seen) < 3:
            for source in selected:
                if manager.latest(source.source_id) is not None:
                    seen.add(source.source_id)
            time.sleep(0.01)
        assert seen == {source.source_id for source in selected}, (
            "Not every selected hardware source produced a frame"
        )
    finally:
        manager.stop()
    assert not any(snapshot.alive for snapshot in manager.snapshot().values())
