from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from aot.capture.buffer import NewestFrameBuffer
from aot.capture.types import FramePacket


def packet(sequence: int) -> FramePacket:
    return FramePacket("source-1", sequence, sequence, np.zeros((1, 1, 3), np.uint8))


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_capacity_must_be_a_positive_integer(capacity) -> None:
    with pytest.raises(ValueError):
        NewestFrameBuffer(capacity)


def test_frames_remain_ordered_before_overflow_and_latest_drains_buffer() -> None:
    buffer = NewestFrameBuffer(3)
    for sequence in range(3):
        assert buffer.put(packet(sequence)) == 0

    assert [item.sequence for item in buffer._snapshot_for_test()] == [0, 1, 2]
    assert buffer.size == 3
    assert buffer.get_latest().sequence == 2
    assert buffer.size == 0
    assert buffer.dropped_count == 2
    assert buffer.get_latest() is None


def test_overflow_drops_oldest_and_reports_exact_counts() -> None:
    buffer = NewestFrameBuffer(2)

    assert buffer.put(packet(0)) == 0
    assert buffer.put(packet(1)) == 0
    assert buffer.put(packet(2)) == 1
    assert [item.sequence for item in buffer._snapshot_for_test()] == [1, 2]
    assert buffer.dropped_count == 1

    assert buffer.get_latest().sequence == 2
    assert buffer.dropped_count == 2


def test_clear_discards_contents_without_counting_runtime_drops() -> None:
    buffer = NewestFrameBuffer(2)
    buffer.put(packet(1))
    buffer.put(packet(2))

    buffer.clear()

    assert buffer.size == 0
    assert buffer.dropped_count == 0


def test_concurrent_access_never_exceeds_capacity() -> None:
    buffer = NewestFrameBuffer(4)

    def produce(offset: int) -> None:
        for value in range(200):
            buffer.put(packet(offset + value))
            assert buffer.size <= 4

    def consume() -> None:
        for _ in range(200):
            buffer.get_latest()
            assert buffer.size <= 4

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(produce, offset) for offset in (0, 1000, 2000)]
        futures.append(executor.submit(consume))
        for future in futures:
            future.result(timeout=5)

    assert buffer.size <= 4
