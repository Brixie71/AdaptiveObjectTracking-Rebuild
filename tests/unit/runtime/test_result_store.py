import numpy as np
import pytest

from aot.capture.types import FramePacket
from aot.runtime.result_store import LatestResultStore
from aot.runtime.types import ProcessedFrame


def result(source_id: str, sequence: int) -> ProcessedFrame:
    packet = FramePacket(
        source_id, sequence, sequence * 10, np.zeros((1, 1, 3), dtype=np.uint8)
    )
    return ProcessedFrame(packet, None, None)


def test_store_rejects_unknown_sources() -> None:
    store = LatestResultStore(["camera-1"])
    with pytest.raises(KeyError):
        store.get("camera-2")
    with pytest.raises(KeyError):
        store.put(result("camera-2", 1))


def test_only_newer_source_local_sequences_replace_the_result() -> None:
    store = LatestResultStore(["camera-1"])
    newest = result("camera-1", 4)

    assert store.put(newest) is True
    assert store.put(result("camera-1", 4)) is False
    assert store.put(result("camera-1", 3)) is False
    assert store.get("camera-1") is newest
    assert store.put(result("camera-1", 5)) is True
    assert store.get("camera-1").packet.sequence == 5


def test_snapshot_is_an_independent_mapping() -> None:
    store = LatestResultStore(["camera-1", "camera-2"])
    store.put(result("camera-1", 1))

    snapshot = store.snapshot()
    snapshot["camera-1"] = None
    snapshot["extra"] = result("extra", 1)

    assert store.get("camera-1") is not None
    assert store.snapshot() == {"camera-1": store.get("camera-1"), "camera-2": None}


def test_source_ids_must_be_unique_and_non_empty() -> None:
    with pytest.raises(ValueError):
        LatestResultStore([])
    with pytest.raises(ValueError):
        LatestResultStore(["camera-1", "camera-1"])
