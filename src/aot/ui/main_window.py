"""Responsive native three-camera application window."""

from __future__ import annotations

from collections.abc import Callable

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from aot.runtime.live_runtime import LiveRuntime
from aot.ui.camera_panel import CameraPanel
from aot.ui.controller import RuntimeController


class MainWindow(QMainWindow):
    """Own widgets and render snapshots; lifecycle work stays in the controller."""

    def __init__(
        self,
        runtime: LiveRuntime,
        parent=None,
        *,
        settings_factory: Callable[[QWidget], QWidget] | None = None,
        refresh_interval_ms: int = 50,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Adaptive Object Tracking")
        self._runtime = runtime
        self._settings_factory = settings_factory
        self._settings_window: QWidget | None = None
        self.controller = RuntimeController(runtime, self)

        central = QWidget(self)
        root = QVBoxLayout(central)
        controls = QHBoxLayout()
        self.start_button = QPushButton("Start", central)
        self.stop_button = QPushButton("Stop", central)
        self.settings_button = QPushButton("Settings", central)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)
        controls.addWidget(self.settings_button)
        root.addLayout(controls)

        grid = QGridLayout()
        self.camera_panels = tuple(CameraPanel(central) for _ in range(3))
        grid.addWidget(self.camera_panels[0], 0, 0)
        grid.addWidget(self.camera_panels[1], 0, 1)
        grid.addWidget(self.camera_panels[2], 1, 0, 1, 2)
        root.addLayout(grid, 1)
        self.setCentralWidget(central)

        self.start_button.clicked.connect(self.controller.request_start)
        self.stop_button.clicked.connect(self.controller.request_stop)
        self.settings_button.clicked.connect(self.open_settings)
        self.controller.controls_changed.connect(self._apply_controls)
        self.controller.started.connect(self.refresh_once)
        self.controller.stopped.connect(self.refresh_once)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(refresh_interval_ms)
        self._refresh_timer.timeout.connect(self.refresh_once)
        self._refresh_timer.start()
        self._apply_controls(
            self.controller.can_start,
            self.controller.can_stop,
            self.controller.busy,
        )
        self.refresh_once()

    def _apply_controls(self, can_start: bool, can_stop: bool, _busy: bool) -> None:
        self.start_button.setEnabled(can_start)
        self.stop_button.setEnabled(can_stop)

    def refresh_once(self) -> None:
        snapshot = self._runtime.snapshot()
        for index, panel in enumerate(self.camera_panels):
            source = snapshot.sources[index] if index < len(snapshot.sources) else None
            panel.apply_snapshot(source)
        self.controller.sync_runtime_state()

    def open_settings(self) -> None:
        if self._settings_factory is None:
            return
        if self._settings_window is not None:
            self._settings_window.show()
            self._settings_window.raise_()
            self._settings_window.activateWindow()
            return
        dialog = self._settings_factory(self)
        dialog.setAttribute(__import__("PyQt5.QtCore").QtCore.Qt.WA_DeleteOnClose, True)
        dialog.destroyed.connect(self._settings_destroyed)
        self._settings_window = dialog
        dialog.show()

    def _settings_destroyed(self, _object=None) -> None:
        self._settings_window = None

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._refresh_timer.stop()
        self.controller.close()
        event.accept()
