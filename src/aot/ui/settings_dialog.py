"""Native Settings dialog ownership."""

from __future__ import annotations

from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import QDialog, QTabWidget, QVBoxLayout

from aot.ui.hardware_testing_panel import HardwareTestingPanel


class SettingsDialog(QDialog):
    def __init__(
        self,
        sources,
        probe_sources_fn,
        discover_usb_fn,
        probe_detector_fn,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        self.hardware_testing = HardwareTestingPanel(
            sources,
            probe_sources_fn,
            discover_usb_fn,
            probe_detector_fn,
            self,
        )
        self.tabs.addTab(self.hardware_testing, "Hardware Testing")
        layout.addWidget(self.tabs)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.hardware_testing.shutdown()
        event.accept()
