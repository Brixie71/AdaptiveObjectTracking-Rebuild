"""Thread-safe bounded newest-frame buffering."""

from __future__ import annotations

from collections import deque
from threading import Lock

from aot.capture.types import FramePacket


class NewestFrameBuffer:
    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Buffer capacity must be a positive integer")
        self._capacity = capacity
        self._packets: deque[FramePacket] = deque()
        self._dropped_count = 0
        self._lock = Lock()

    def put(self, packet: FramePacket) -> int:
        with self._lock:
            dropped = 0
            if len(self._packets) == self._capacity:
                self._packets.popleft()
                self._dropped_count += 1
                dropped = 1
            self._packets.append(packet)
            return dropped

    def get_latest(self) -> FramePacket | None:
        with self._lock:
            if not self._packets:
                return None
            latest = self._packets[-1]
            discarded = len(self._packets) - 1
            self._dropped_count += discarded
            self._packets.clear()
            return latest

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._packets)

    @property
    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    def clear(self) -> None:
        with self._lock:
            self._packets.clear()

    def _snapshot_for_test(self) -> tuple[FramePacket, ...]:
        with self._lock:
            return tuple(self._packets)
