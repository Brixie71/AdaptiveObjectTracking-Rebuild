from __future__ import annotations

import threading
from datetime import datetime, timezone

from PyQt5.QtCore import QTimer

from aot.config.types import CameraSourceConfig, SourceType
from aot.hardware_testing.types import (
    DetectorTestResult,
    HardwareTestStatus,
    SourceTestResult,
    UsbDeviceCandidate,
)
from aot.ui.hardware_testing_panel import HardwareTestingPanel


NOW = datetime(2026, 10, 12, tzinfo=timezone.utc)


def configured_sources():
    return (
        CameraSourceConfig(
            "front",
            "Front Door",
            SourceType.RTSP,
            True,
            rtsp_env_key="PRIVATE_RTSP_ENV",
        ),
        CameraSourceConfig(
            "usb",
            "Workshop",
            SourceType.USB,
            True,
            device_index=7,
        ),
    )


def source_result(source_id: str, status: HardwareTestStatus) -> SourceTestResult:
    return SourceTestResult(
        source_id,
        SourceType.USB,
        status,
        None,
        3,
        3 if status is HardwareTestStatus.SUCCESS else 0,
        1.0,
        1.5,
        0,
        NOW,
        NOW,
        f"Source probe {status.value}",
    )


def detector_result(status=HardwareTestStatus.SUCCESS) -> DetectorTestResult:
    return DetectorTestResult(
        "yolo",
        status,
        "cpu",
        1,
        2,
        2,
        4.0,
        5.0,
        1,
        NOW,
        NOW,
        f"Detector probe {status.value}",
    )


def table_text(panel: HardwareTestingPanel) -> str:
    values = []
    for row in range(panel.results_table.rowCount()):
        for column in range(panel.results_table.columnCount()):
            item = panel.results_table.item(row, column)
            if item is not None:
                values.append(item.text())
    return " ".join(values)


def test_panel_lists_only_non_sensitive_source_metadata_and_actions(qtbot) -> None:
    panel = HardwareTestingPanel(configured_sources(), lambda _sources: (), lambda: (), lambda: None)
    qtbot.addWidget(panel)
    visible = " ".join(label.text() for label in panel.source_labels)

    assert "Front Door" in visible and "front" in visible and "RTSP" in visible
    assert "Workshop" in visible and "usb" in visible and "USB" in visible
    assert "PRIVATE_RTSP_ENV" not in visible
    assert "7" not in visible
    assert panel.run_sources_button.text() == "Run Source Tests"
    assert panel.discover_usb_button.text() == "Discover USB"
    assert panel.run_detector_button.text() == "Run Detector Test"


def test_probe_runs_async_disables_all_actions_and_blocks_duplicates(qtbot) -> None:
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def probe(sources):
        calls.append(tuple(s.source_id for s in sources))
        entered.set()
        release.wait(1.0)
        return tuple(
            source_result(f"s-{index}", status)
            for index, status in enumerate(HardwareTestStatus)
        )

    panel = HardwareTestingPanel(configured_sources(), probe, lambda: (), lambda: None)
    qtbot.addWidget(panel)
    heartbeat = []
    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: heartbeat.append(1))
    timer.start()

    panel.run_sources_button.click()
    assert entered.wait(0.5)
    panel.run_sources_button.click()
    assert not any(
        button.isEnabled()
        for button in (
            panel.run_sources_button,
            panel.discover_usb_button,
            panel.run_detector_button,
        )
    )
    qtbot.waitUntil(lambda: len(heartbeat) >= 2)
    assert len(calls) == 1
    release.set()
    qtbot.waitUntil(lambda: panel.run_sources_button.isEnabled())

    rendered = table_text(panel)
    for status in HardwareTestStatus:
        assert status.value in rendered


def test_usb_and_detector_results_render_project_owned_fields(qtbot) -> None:
    panel = HardwareTestingPanel(
        configured_sources(),
        lambda _sources: (),
        lambda: (UsbDeviceCandidate(2, 640, 480, 30.0),),
        lambda: detector_result(),
    )
    qtbot.addWidget(panel)

    panel.discover_usb_button.click()
    qtbot.waitUntil(lambda: "640×480" in table_text(panel))
    panel.run_detector_button.click()
    qtbot.waitUntil(lambda: "yolo" in table_text(panel))

    assert "cpu" in table_text(panel)
    assert "success" in table_text(panel)


def test_close_ignores_late_probe_results(qtbot) -> None:
    entered = threading.Event()
    release = threading.Event()

    def probe(_sources):
        entered.set()
        release.wait(1.0)
        return (source_result("late", HardwareTestStatus.SUCCESS),)

    panel = HardwareTestingPanel(configured_sources(), probe, lambda: (), lambda: None)
    qtbot.addWidget(panel)
    panel.run_sources_button.click()
    assert entered.wait(0.5)
    panel.close()
    release.set()
    qtbot.wait(30)

    assert "late" not in table_text(panel)


def test_stale_generation_cannot_replace_newer_result(qtbot) -> None:
    panel = HardwareTestingPanel(configured_sources(), lambda _sources: (), lambda: (), lambda: None)
    qtbot.addWidget(panel)
    panel._generation = 2
    panel._apply_success(2, "sources", (source_result("new", HardwareTestStatus.SUCCESS),))
    panel._apply_success(1, "sources", (source_result("old", HardwareTestStatus.FAILED),))

    assert "new" in table_text(panel)
    assert "old" not in table_text(panel)


def test_gate4_hardware_panel_does_not_persist_profiles(qtbot, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    panel = HardwareTestingPanel(configured_sources(), lambda _sources: (), lambda: (), lambda: None)
    qtbot.addWidget(panel)

    assert "Gate 10" in panel.profile_notice.text()
    assert list(tmp_path.iterdir()) == []
