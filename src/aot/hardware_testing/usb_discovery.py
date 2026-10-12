"""Bounded USB camera index discovery."""

from __future__ import annotations

from aot.capture.backend import BackendFactory, BackendProperty, open_cv_backend
from aot.hardware_testing.types import UsbDeviceCandidate


def discover_usb_devices(
    *,
    max_index: int = 10,
    backend_factory: BackendFactory = open_cv_backend,
) -> tuple[UsbDeviceCandidate, ...]:
    if (
        isinstance(max_index, bool)
        or not isinstance(max_index, int)
        or max_index < 0
    ):
        raise ValueError("max_index must be a nonnegative integer")

    candidates: list[UsbDeviceCandidate] = []
    for index in range(max_index):
        backend = None
        try:
            backend = backend_factory(index)
            if backend.is_opened():
                candidates.append(
                    UsbDeviceCandidate(
                        device_index=index,
                        width=int(backend.get(BackendProperty.WIDTH)),
                        height=int(backend.get(BackendProperty.HEIGHT)),
                        fps=float(backend.get(BackendProperty.FPS)),
                    )
                )
        except Exception:
            continue
        finally:
            if backend is not None:
                backend.release()
    return tuple(candidates)
