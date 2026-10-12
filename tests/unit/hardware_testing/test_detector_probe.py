from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pytest

from aot.capture.types import FramePacket
from aot.detection.types import DetectionBatch, DetectorInferenceError
from aot.hardware_testing.detector_probe import probe_detector
from aot.hardware_testing.types import HardwareTestStatus


def _packet() -> FramePacket:
    return FramePacket(
        source_id="hardware-test",
        sequence=0,
        captured_at_monotonic_ns=1,
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
    )


def _batch(count: int = 0) -> DetectionBatch:
    return DetectionBatch(
        source_id="hardware-test",
        frame_sequence=0,
        captured_at_monotonic_ns=1,
        frame_width=640,
        frame_height=480,
        detections=tuple([None] * count),
        rejected_count=0,
    )


class ProbeDetector:
    model_id = "detector-test"
    class_names = ("person",)

    def __init__(
        self,
        outcomes: Iterable[DetectionBatch | Exception],
        *,
        close_error: Exception | None = None,
    ) -> None:
        self._outcomes = iter(outcomes)
        self.close_error = close_error
        self.detect_count = 0
        self.close_count = 0

    def detect(self, _packet: FramePacket) -> DetectionBatch:
        self.detect_count += 1
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self) -> None:
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error


def test_detector_probe_reports_successful_nearest_rank_timing() -> None:
    detector = ProbeDetector([_batch(), _batch(), _batch(2), _batch(1), _batch()])
    clock = iter([0, 10_000_000, 20_000_000, 40_000_000, 50_000_000, 80_000_000])

    result = probe_detector(
        detector,
        _packet(),
        device="cpu",
        warmup_runs=2,
        measured_runs=3,
        clock_ns=lambda: next(clock),
    )

    assert result.status is HardwareTestStatus.SUCCESS
    assert result.warmup_runs == 2
    assert result.measured_runs == 3
    assert result.successful_runs == 3
    assert result.mean_inference_ms == 20.0
    assert result.p95_inference_ms == 30.0
    assert result.detection_count == 3
    assert detector.close_count == 1
    assert result.to_dict()["status"] == "success"


def test_zero_detections_still_proves_successful_inference() -> None:
    detector = ProbeDetector([_batch()])

    result = probe_detector(
        detector,
        _packet(),
        device="cpu",
        warmup_runs=0,
        measured_runs=1,
        clock_ns=iter([0, 1_000_000]).__next__,
    )

    assert result.status is HardwareTestStatus.SUCCESS
    assert result.detection_count == 0


def test_partial_inference_failure_is_degraded_and_sanitized() -> None:
    secret = "rtsp://admin:camera-pass@example.invalid/live"
    detector = ProbeDetector([_batch(1), DetectorInferenceError(secret)])

    result = probe_detector(
        detector,
        _packet(),
        device="cuda:0",
        warmup_runs=0,
        measured_runs=3,
        clock_ns=iter([0, 1_000_000, 2_000_000]).__next__,
    )

    assert result.status is HardwareTestStatus.DEGRADED
    assert result.successful_runs == 1
    assert secret not in result.summary
    assert secret not in str(result.to_dict())


def test_first_inference_or_cleanup_failure_is_failed() -> None:
    secret = "token=private-model-token"
    first_failure = ProbeDetector([RuntimeError(secret)])

    inference_result = probe_detector(
        first_failure,
        _packet(),
        device="cpu",
        warmup_runs=0,
        measured_runs=1,
        clock_ns=iter([0, 1]).__next__,
    )

    cleanup_failure = ProbeDetector([_batch()], close_error=RuntimeError(secret))
    cleanup_result = probe_detector(
        cleanup_failure,
        _packet(),
        device="cpu",
        warmup_runs=0,
        measured_runs=1,
        clock_ns=iter([0, 1]).__next__,
    )

    assert inference_result.status is HardwareTestStatus.FAILED
    assert cleanup_result.status is HardwareTestStatus.FAILED
    assert secret not in inference_result.summary
    assert secret not in cleanup_result.summary
    assert first_failure.close_count == 1
    assert cleanup_failure.close_count == 1


@pytest.mark.parametrize(
    ("warmup_runs", "measured_runs"),
    [(-1, 1), (True, 1), (0, 0), (0, -1), (0, True), (0.5, 1), (0, 1.5)],
)
def test_detector_probe_rejects_invalid_run_counts(warmup_runs, measured_runs) -> None:
    with pytest.raises(ValueError):
        probe_detector(
            ProbeDetector([]),
            _packet(),
            device="cpu",
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )
