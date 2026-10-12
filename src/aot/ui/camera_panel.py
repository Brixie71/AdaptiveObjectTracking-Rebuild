"""One native camera tile driven exclusively by runtime snapshots."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QResizeEvent
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout

from aot.capture.types import SourceState
from aot.runtime.types import SourceRuntimeSnapshot
from aot.ui.rendering import FrameRenderingError, render_processed_frame


_STATUS_TEXT = {
    SourceState.CLOSED: "Stopped",
    SourceState.OPENING: "Connecting",
    SourceState.OPEN: "Live",
    SourceState.RECONNECTING: "Reconnecting",
    SourceState.EOF: "End of stream",
    SourceState.FAILED: "Unavailable",
}


class CameraPanel(QFrame):
    """Present one source without performing capture or model work."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("cameraPanel")
        self.setFrameShape(QFrame.StyledPanel)
        self.name_label = QLabel("Unconfigured camera", self)
        self.name_label.setObjectName("cameraName")
        self.source_label = QLabel("No source assigned", self)
        self.source_label.setObjectName("cameraSource")
        self.status_label = QLabel("Unconfigured", self)
        self.status_label.setObjectName("cameraStatus")
        self.image_label = QLabel("No preview", self)
        self.image_label.setObjectName("cameraImage")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(160, 100)
        self.image_label.setStyleSheet("background: #121820; color: #8996a8;")
        self._image: QImage | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(self.name_label)
        layout.addWidget(self.source_label)
        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.status_label)

        self.setStyleSheet(
            "QLabel#cameraStatus[state='open'] { color: #38b87c; }"
            "QLabel#cameraStatus[state='reconnecting'] { color: #d7a93e; }"
            "QLabel#cameraStatus[state='eof'] { color: #8aa0ba; }"
            "QLabel#cameraStatus[state='failed'] { color: #df6868; }"
            "QLabel#cameraStatus[state='unconfigured'] { color: #8996a8; }"
        )

    def apply_snapshot(self, snapshot: SourceRuntimeSnapshot | None) -> None:
        if snapshot is None:
            self.name_label.setText("Unconfigured camera")
            self.source_label.setText("No source assigned")
            self._set_status("Unconfigured", "unconfigured")
            self._image = None
            self.image_label.clear()
            self.image_label.setText("No preview")
            return

        self.name_label.setText(snapshot.display_name)
        self.source_label.setText(
            f"{snapshot.source_id} · {snapshot.source_type.value.upper()}"
        )
        status = _STATUS_TEXT[snapshot.capture.state]
        if (
            snapshot.capture.state is SourceState.OPEN
            and snapshot.result is not None
            and snapshot.result.processing_error is not None
        ):
            status = "Live · Detection unavailable"
        self._set_status(status, snapshot.capture.state.value)

        if snapshot.result is not None:
            try:
                self._image = render_processed_frame(snapshot.result)
            except FrameRenderingError:
                self._image = None
                self.image_label.clear()
                self.image_label.setText("Preview unavailable")
            else:
                self._refresh_pixmap()

    def _set_status(self, text: str, state: str) -> None:
        self.status_label.setText(text)
        self.status_label.setProperty("state", state)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _refresh_pixmap(self) -> None:
        if self._image is None:
            return
        pixmap = QPixmap.fromImage(self._image).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(pixmap)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_pixmap()
