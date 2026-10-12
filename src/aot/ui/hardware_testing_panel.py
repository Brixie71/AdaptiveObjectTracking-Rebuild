"""Asynchronous, sanitized hardware checks for the Settings window."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from PyQt5.QtCore import QThreadPool, Qt, pyqtSlot
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from aot.config.types import CameraSourceConfig
from aot.hardware_testing.types import (
    DetectorTestResult,
    SourceTestResult,
    UsbDeviceCandidate,
)
from aot.ui.background_task import FunctionTask, submit_function_task


class HardwareTestingPanel(QWidget):
    """Run existing probe services without exposing secrets or blocking Qt."""

    def __init__(
        self,
        sources: Iterable[CameraSourceConfig],
        probe_sources_fn: Callable,
        discover_usb_fn: Callable,
        probe_detector_fn: Callable,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._sources = tuple(sources)
        self._probe_sources = probe_sources_fn
        self._discover_usb = discover_usb_fn
        self._probe_detector = probe_detector_fn
        self._pool = QThreadPool.globalInstance()
        self._task: FunctionTask | None = None
        self._generation = 0
        self._closed = False

        layout = QVBoxLayout(self)
        heading = QLabel("Hardware Testing", self)
        heading.setObjectName("hardwareTestingHeading")
        layout.addWidget(heading)
        layout.addWidget(QLabel("Configured sources", self))
        self.source_labels: list[QLabel] = []
        for source in self._sources:
            label = QLabel(
                f"{source.name} — {source.source_id} · {source.source_type.value.upper()}",
                self,
            )
            self.source_labels.append(label)
            layout.addWidget(label)

        actions = QHBoxLayout()
        self.run_sources_button = QPushButton("Run Source Tests", self)
        self.discover_usb_button = QPushButton("Discover USB", self)
        self.run_detector_button = QPushButton("Run Detector Test", self)
        self._action_buttons = (
            self.run_sources_button,
            self.discover_usb_button,
            self.run_detector_button,
        )
        for button in self._action_buttons:
            actions.addWidget(button)
        layout.addLayout(actions)

        self.results_table = QTableWidget(0, 5, self)
        self.results_table.setHorizontalHeaderLabels(
            ("Test", "Identifier", "Type / device", "Status", "Summary")
        )
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.results_table, 1)
        self.profile_notice = QLabel(
            "Named profiles can be saved after successful testing in Gate 10.", self
        )
        layout.addWidget(self.profile_notice)

        self.run_sources_button.clicked.connect(
            lambda: self._start_job(
                "sources", lambda: self._probe_sources(self._sources)
            )
        )
        self.discover_usb_button.clicked.connect(
            lambda: self._start_job("usb", self._discover_usb)
        )
        self.run_detector_button.clicked.connect(
            lambda: self._start_job("detector", self._probe_detector)
        )

    def _set_busy(self, busy: bool) -> None:
        for button in self._action_buttons:
            button.setEnabled(not busy and not self._closed)

    def _start_job(self, kind: str, function: Callable) -> None:
        if self._closed or self._task is not None:
            return
        self._generation += 1
        generation = self._generation
        task = FunctionTask(lambda: (generation, kind, function()))
        task.signals.succeeded.connect(self._on_success, Qt.QueuedConnection)
        task.signals.failed.connect(self._on_failure, Qt.QueuedConnection)
        task.signals.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._task = task
        self._set_busy(True)
        submit_function_task(self._pool, task)

    @pyqtSlot(object)
    def _on_success(self, payload) -> None:
        generation, kind, result = payload
        self._apply_success(generation, kind, result)

    def _apply_success(self, generation: int, kind: str, result) -> None:
        if self._closed or generation != self._generation:
            return
        self.results_table.setRowCount(0)
        if kind == "sources":
            for item in tuple(result):
                self._append_source(item)
        elif kind == "usb":
            for item in tuple(result):
                self._append_usb(item)
        elif kind == "detector" and result is not None:
            self._append_detector(result)

    @pyqtSlot(str)
    def _on_failure(self, _summary: str) -> None:
        if self._closed:
            return
        self.results_table.setRowCount(0)
        self._append_row(
            "Operation", "hardware-test", "local", "failed", "Hardware test could not be completed"
        )

    @pyqtSlot(object)
    def _on_finished(self, task: FunctionTask) -> None:
        if task is self._task:
            self._task = None
            self._set_busy(False)

    def _append_source(self, result: SourceTestResult) -> None:
        source_type = getattr(result.source_type, "value", str(result.source_type))
        self._append_row(
            "Source",
            result.source_id,
            source_type,
            result.status.value,
            result.summary,
        )

    def _append_usb(self, result: UsbDeviceCandidate) -> None:
        self._append_row(
            "USB discovery",
            f"USB {result.device_index}",
            "usb",
            "success",
            f"{result.width}×{result.height} at {result.fps:g} FPS",
        )

    def _append_detector(self, result: DetectorTestResult) -> None:
        self._append_row(
            "Detector",
            result.model_id,
            result.device,
            result.status.value,
            result.summary,
        )

    def _append_row(
        self, test: str, identifier: str, type_or_device: str, status: str, summary: str
    ) -> None:
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        for column, value in enumerate(
            (test, identifier, type_or_device, status, summary)
        ):
            self.results_table.setItem(row, column, QTableWidgetItem(str(value)))

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._generation += 1
        if self._task is not None:
            self._task.cancel_callbacks()
        self._set_busy(False)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.shutdown()
        event.accept()
