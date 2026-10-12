from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np

from aot.capture.types import FramePacket
from aot.detection.types import DetectionBatch
from aot.runtime.processing_worker import ProcessingWorker
from aot.runtime.result_store import LatestResultStore


def packet(source_id: str, sequence: int) -> FramePacket:
    return FramePacket(
        source_id, sequence, sequence, np.zeros((2, 3, 3), dtype=np.uint8)
    )


def wait_until(predicate, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached")


def test_round_robin_raw_processing_is_bounded_and_ignores_duplicates() -> None:
    source_ids = ("a", "b")
    store = LatestResultStore(source_ids)
    packets = {
        "a": deque((packet("a", 1), packet("a", 1), packet("a", 2))),
        "b": deque((packet("b", 4), packet("b", 4), packet("b", 5))),
    }
    polls: list[str] = []

    def supply(source_id: str) -> FramePacket | None:
        polls.append(source_id)
        return packets[source_id].popleft() if packets[source_id] else None

    worker = ProcessingWorker(source_ids, supply, store, detector=None, poll_seconds=0.001)
    worker.start()
    wait_until(lambda: all(not queue for queue in packets.values()))
    worker.request_stop()

    assert worker.join(1.0) is True
    assert polls[:6] == ["a", "b", "a", "b", "a", "b"]
    assert store.get("a").packet.sequence == 2
    assert store.get("b").packet.sequence == 5
    assert store.get("a").detections is None
    assert not hasattr(worker, "_queue")


class RecordingDetector:
    def __init__(self) -> None:
        self.thread_ids: list[int] = []
        self.closed = False

    def detect(self, captured: FramePacket) -> DetectionBatch:
        self.thread_ids.append(threading.get_ident())
        if captured.source_id == "bad":
            raise RuntimeError("PRIVATE-MARKER")
        height, width = captured.frame.shape[:2]
        return DetectionBatch(
            captured.source_id,
            captured.sequence,
            captured.captured_at_monotonic_ns,
            width,
            height,
            (),
            0,
        )

    def close(self) -> None:
        self.closed = True


def test_detector_runs_off_caller_thread_and_failure_is_source_local() -> None:
    source_ids = ("bad", "good")
    store = LatestResultStore(source_ids)
    packets = {source_id: deque((packet(source_id, 1),)) for source_id in source_ids}
    detector = RecordingDetector()
    caller_thread = threading.get_ident()

    worker = ProcessingWorker(
        source_ids,
        lambda source_id: packets[source_id].popleft() if packets[source_id] else None,
        store,
        detector,
        poll_seconds=0.001,
    )
    worker.start()
    wait_until(lambda: store.get("good") is not None)
    worker.request_stop()

    assert worker.join(1.0) is True
    assert all(thread_id != caller_thread for thread_id in detector.thread_ids)
    assert worker.snapshot_error("bad") == "Detector inference failed"
    assert "PRIVATE-MARKER" not in repr(worker.snapshot_error("bad"))
    assert store.get("good").detections.frame_sequence == 1
    assert detector.closed is False


def test_request_stop_interrupts_idle_wait_and_join_is_idempotent() -> None:
    store = LatestResultStore(["a"])
    worker = ProcessingWorker(["a"], lambda _source_id: None, store, poll_seconds=30.0)
    worker.start()
    worker.request_stop()

    assert worker.join(0.5) is True
    assert worker.join(0.5) is True
    assert worker.is_alive is False


def test_blocked_detector_makes_join_bounded_without_closing_detector() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingDetector(RecordingDetector):
        def detect(self, captured: FramePacket) -> DetectionBatch:
            entered.set()
            release.wait(2.0)
            return super().detect(captured)

    detector = BlockingDetector()
    supplied = False

    def supply(_source_id: str) -> FramePacket | None:
        nonlocal supplied
        if supplied:
            return None
        supplied = True
        return packet("a", 1)

    worker = ProcessingWorker(
        ["a"], supply, LatestResultStore(["a"]), detector, poll_seconds=0.001
    )
    worker.start()
    assert entered.wait(0.5)
    worker.request_stop()
    started = time.monotonic()

    assert worker.join(0.02) is False
    assert time.monotonic() - started < 0.2
    assert detector.closed is False

    release.set()
    assert worker.join(1.0) is True


def test_unknown_error_source_is_rejected() -> None:
    worker = ProcessingWorker(
        ["a"], lambda _source_id: None, LatestResultStore(["a"])
    )
    try:
        worker.snapshot_error("missing")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown source must raise KeyError")
