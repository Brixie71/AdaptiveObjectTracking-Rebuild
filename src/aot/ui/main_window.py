"""Responsive native three-camera application window."""

from __future__ import annotations

from collections.abc import Callable

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from aot.runtime.live_runtime import LiveRuntime
from aot.selection.target_selector import TargetSelectionError, TargetSelector
from aot.ui.camera_panel import CameraPanel
from aot.ui.controller import RuntimeController
from aot.ui.selected_target_panel import SelectedTargetPanel


class MainWindow(QMainWindow):
    """Own widgets and render snapshots; lifecycle work stays in the controller."""

    def __init__(
        self,
        runtime: LiveRuntime,
        parent=None,
        *,
        target_selector: TargetSelector,
        settings_factory: Callable[[QWidget], QWidget] | None = None,
        refresh_interval_ms: int = 50,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Adaptive Object Tracking")
        self._runtime = runtime
        self._target_selector = target_selector
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
        self.operation_status_label = QLabel("", central)
        self.operation_status_label.setObjectName("operationStatusLabel")
        root.addWidget(self.operation_status_label)

        grid = QGridLayout()
        self.camera_panels = tuple(CameraPanel(central) for _ in range(3))
        grid.addWidget(self.camera_panels[0], 0, 0)
        grid.addWidget(self.camera_panels[1], 0, 1)
        grid.addWidget(self.camera_panels[2], 1, 0, 1, 2)
        root.addLayout(grid, 1)
        self.selected_target_panel = SelectedTargetPanel(central)
        self.selected_target_panel.apply_target(self._target_selector.current)
        root.addWidget(self.selected_target_panel)
        self.setCentralWidget(central)

        self.start_button.clicked.connect(self.controller.request_start)
        self.stop_button.clicked.connect(self.controller.request_stop)
        self.settings_button.clicked.connect(self.open_settings)
        self.controller.controls_changed.connect(self._apply_controls)
        self.controller.started.connect(self.refresh_once)
        self.controller.stopped.connect(self.refresh_once)
        self.controller.started.connect(lambda: self.operation_status_label.clear())
        self.controller.stopped.connect(lambda: self.operation_status_label.clear())
        self.controller.failed.connect(self.operation_status_label.setText)
        for panel in self.camera_panels:
            panel.selection_requested.connect(self._select_target)
        self.selected_target_panel.disengage_requested.connect(self._disengage_target)

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
            panel.set_selection_enabled(snapshot.running)
        self.controller.sync_runtime_state()

    def _select_target(self, result, frame_x: float, frame_y: float) -> None:
        try:
            target = self._target_selector.select(result, frame_x, frame_y)
        except TargetSelectionError:
            self.operation_status_label.setText("Target selection unavailable")
            return
        if target is None:
            self.operation_status_label.setText("No selectable target at this point")
            return
        self.selected_target_panel.apply_target(target)
        self.operation_status_label.setText("Target selected")

    def _disengage_target(self) -> None:
        self._target_selector.disengage()
        self.selected_target_panel.apply_target(None)
        self.operation_status_label.setText("Target disengaged")

    def open_settings(self) -> None:
        if self._settings_factory is None:
            return
        if self._settings_window is not None:
            self._settings_window.show()
            self._settings_window.raise_()
            self._settings_window.activateWindow()
            return
        dialog = self._settings_factory(self)
        self._settings_window = dialog
        dialog.show()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._refresh_timer.stop()
        if self._settings_window is not None:
            shutdown = getattr(self._settings_window, "shutdown", None)
            if callable(shutdown):
                shutdown()
        self.controller.close()
        event.accept()

    def wait_for_shutdown(self) -> bool:
        """Complete the already-requested close after the Qt loop exits."""
        return self.controller.wait_for_done()
