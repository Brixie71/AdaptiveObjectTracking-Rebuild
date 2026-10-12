from collections.abc import Iterable

import numpy as np


class FakeCaptureBackend:
    def __init__(
        self,
        *,
        opened: bool = True,
        reads: Iterable[tuple[bool, np.ndarray | None]] = (),
        negotiated: dict[object, float] | None = None,
    ) -> None:
        self.opened = opened
        self.reads = list(reads)
        self.negotiated = negotiated or {}
        self.events: list[tuple[object, ...]] = []
        self.release_count = 0

    def is_opened(self) -> bool:
        self.events.append(("is_opened",))
        return self.opened

    def set(self, property_name, value: float) -> bool:
        self.events.append(("set", property_name, value))
        return True

    def get(self, property_name) -> float:
        self.events.append(("get", property_name))
        return self.negotiated.get(property_name, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.events.append(("read",))
        if not self.reads:
            return False, None
        return self.reads.pop(0)

    def release(self) -> None:
        self.events.append(("release",))
        self.release_count += 1


class RecordingBackendFactory:
    def __init__(self, backend: FakeCaptureBackend) -> None:
        self.backend = backend
        self.inputs: list[int | str] = []

    def __call__(self, source: int | str) -> FakeCaptureBackend:
        self.inputs.append(source)
        return self.backend
