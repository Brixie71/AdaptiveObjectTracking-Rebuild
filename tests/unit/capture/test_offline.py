from pathlib import Path

import numpy as np
import pytest

from aot.capture.offline import OfflineFrameSource
from aot.capture.types import CaptureOpenError, EndOfStream, SourceState
from aot.config.types import CameraSourceConfig, SourceType

from fakes import FakeCaptureBackend, RecordingBackendFactory


def offline_config(path: Path, **changes) -> CameraSourceConfig:
    values = {
        "source_id": "clip-1",
        "name": "Acceptance clip",
        "source_type": SourceType.OFFLINE,
        "offline_path": path,
        "width": 4,
        "height": 2,
        "fps": 15.0,
        "rotation": 0,
    }
    values.update(changes)
    return CameraSourceConfig(**values)


def test_offline_source_accepts_only_complete_offline_configuration(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.touch()
    factory = RecordingBackendFactory(FakeCaptureBackend())

    with pytest.raises(ValueError):
        OfflineFrameSource(offline_config(video, source_type=SourceType.USB), factory)
    with pytest.raises(ValueError):
        OfflineFrameSource(offline_config(video, offline_path=None), factory)
    with pytest.raises(ValueError):
        OfflineFrameSource(offline_config(Path("relative.mp4")), factory)


@pytest.mark.parametrize("path_kind", ["missing", "directory"])
def test_missing_or_non_file_path_is_rejected_before_backend_open(tmp_path, path_kind) -> None:
    path = tmp_path / "input"
    if path_kind == "directory":
        path.mkdir()
    factory = RecordingBackendFactory(FakeCaptureBackend())
    source = OfflineFrameSource(offline_config(path), factory)

    with pytest.raises(CaptureOpenError):
        source.open()

    assert factory.inputs == []
    assert source.state is SourceState.FAILED


def test_open_uses_resolved_absolute_file_path(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.touch()
    factory = RecordingBackendFactory(FakeCaptureBackend())
    source = OfflineFrameSource(offline_config(video), factory)

    source.open()

    assert factory.inputs == [str(video.resolve())]
    assert source.reconnectable is False


def test_read_normalizes_frames_then_reports_terminal_eof(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.touch()
    frame = np.array([[0, 255]], dtype=np.uint8)
    backend = FakeCaptureBackend(reads=[(True, frame), (False, None)])
    source = OfflineFrameSource(
        offline_config(video),
        RecordingBackendFactory(backend),
        clock_ns=lambda: 456,
    )
    source.open()

    packet = source.read()
    assert packet.sequence == 0
    assert packet.captured_at_monotonic_ns == 456
    assert packet.frame.shape == (2, 4, 3)

    with pytest.raises(EndOfStream) as caught:
        source.read()

    assert caught.value.source_id == "clip-1"
    assert source.state is SourceState.EOF
    assert source.reconnectable is False


@pytest.mark.parametrize("reach_eof", [False, True])
def test_close_releases_once_from_open_or_eof_state(tmp_path, reach_eof) -> None:
    video = tmp_path / "clip.mp4"
    video.touch()
    backend = FakeCaptureBackend(reads=[(False, None)])
    source = OfflineFrameSource(offline_config(video), RecordingBackendFactory(backend))
    source.open()
    if reach_eof:
        with pytest.raises(EndOfStream):
            source.read()

    source.close()
    source.close()

    assert backend.release_count == 1
    assert source.state is SourceState.CLOSED
