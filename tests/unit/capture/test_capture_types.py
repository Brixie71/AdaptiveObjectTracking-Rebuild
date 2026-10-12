from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from aot.capture.types import (
    CaptureError,
    CaptureOpenError,
    CaptureProperties,
    CaptureReadError,
    EndOfStream,
    FramePacket,
    FrameSource,
    SourceState,
)
from aot.config.types import SourceType


def test_capture_states_cover_the_complete_source_lifecycle() -> None:
    assert {state.value for state in SourceState} == {
        "closed",
        "opening",
        "open",
        "reconnecting",
        "eof",
        "failed",
    }


def test_capture_records_are_immutable_and_preserve_boundary_values() -> None:
    properties = CaptureProperties(
        requested_width=640,
        requested_height=480,
        requested_fps=15.0,
        negotiated_width=1280,
        negotiated_height=720,
        negotiated_fps=30.0,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    packet = FramePacket(
        source_id="usb-1",
        sequence=0,
        captured_at_monotonic_ns=123,
        frame=frame,
    )

    assert properties.negotiated_width == 1280
    assert packet.sequence == 0
    assert packet.frame is frame
    with pytest.raises(FrozenInstanceError):
        packet.sequence = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "error_type",
    [CaptureError, CaptureOpenError, CaptureReadError, EndOfStream],
)
def test_capture_errors_expose_only_source_id_and_sanitized_summary(error_type) -> None:
    error = error_type("cctv-1", "RTSP source could not be opened")

    assert error.source_id == "cctv-1"
    assert error.summary == "RTSP source could not be opened"
    assert str(error) == "RTSP source could not be opened"
    assert vars(error) == {
        "source_id": "cctv-1",
        "summary": "RTSP source could not be opened",
    }


def test_minimal_source_satisfies_runtime_checkable_protocol() -> None:
    class MinimalSource:
        source_id = "offline-1"
        source_type = SourceType.OFFLINE
        reconnectable = False
        state = SourceState.CLOSED
        properties = None

        def open(self):
            raise NotImplementedError

        def read(self):
            raise NotImplementedError

        def close(self):
            return None

    assert isinstance(MinimalSource(), FrameSource)
