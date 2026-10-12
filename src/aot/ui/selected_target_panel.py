"""Presentation-only panel for the current in-memory target reference."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QResizeEvent
from PyQt5.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout

from aot.selection.types import SelectedTarget
from aot.ui.rendering import frame_to_qimage


class SelectedTargetPanel(QFrame):
    """Show safe metadata and an owned crop for the selected target."""

    disengage_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("selectedTargetPanel")
        self.setFrameShape(QFrame.StyledPanel)

        heading = QLabel("Selected target", self)
        heading.setObjectName("selectedTargetHeading")
        self.status_label = QLabel(self)
        self.status_label.setObjectName("selectedTargetStatus")
        self.crop_label = QLabel(self)
        self.crop_label.setObjectName("selectedTargetCrop")
        self.crop_label.setAlignment(Qt.AlignCenter)
        self.crop_label.setMinimumSize(160, 100)
        self.crop_label.setStyleSheet("background: #121820; color: #8996a8;")
        self.target_id_label = QLabel(self)
        self.class_label = QLabel(self)
        self.source_label = QLabel(self)
        self.frame_label = QLabel(self)
        self.selected_at_label = QLabel(self)
        self.box_label = QLabel(self)
        for label in self._metadata_labels:
            label.setTextFormat(Qt.PlainText)
        self.disengage_button = QPushButton("Disable / Disengage Target", self)
        self.disengage_button.clicked.connect(self.disengage_requested.emit)
        self._image: QImage | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(heading)
        layout.addWidget(self.status_label)
        layout.addWidget(self.crop_label, 1)
        for label in self._metadata_labels:
            layout.addWidget(label)
        layout.addWidget(self.disengage_button)

        self.apply_target(None)

    @property
    def _metadata_labels(self) -> tuple[QLabel, ...]:
        return (
            self.target_id_label,
            self.class_label,
            self.source_label,
            self.frame_label,
            self.selected_at_label,
            self.box_label,
        )

    def apply_target(self, target: SelectedTarget | None) -> None:
        """Replace the displayed selection, or return to the explicit empty state."""
        if target is None:
            self._image = None
            self.crop_label.clear()
            self.crop_label.setText("No target crop")
            self.status_label.setText("No target selected")
            for label in self._metadata_labels:
                label.clear()
            self.disengage_button.setEnabled(False)
            return

        self._image = frame_to_qimage(target.reference_crop)
        self.status_label.setText("Target selected")
        self.target_id_label.setText(f"Target: {target.target_id}")
        self.class_label.setText(
            f"Class: {target.class_name} ({target.confidence:.0%})"
        )
        self.source_label.setText(f"Source: {target.source_id}")
        self.frame_label.setText(f"Frame: {target.frame_sequence}")
        self.selected_at_label.setText(
            f"Selected: {target.selected_at_utc.isoformat()}"
        )
        box = target.initial_box
        self.box_label.setText(
            f"Box: ({box.left}, {box.top})-({box.right}, {box.bottom})"
        )
        self.disengage_button.setEnabled(True)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._image is None:
            return
        pixmap = QPixmap.fromImage(self._image).scaled(
            self.crop_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.crop_label.setPixmap(pixmap)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_pixmap()
