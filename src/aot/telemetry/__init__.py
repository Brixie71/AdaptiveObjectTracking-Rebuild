"""Runtime diagnostics and structured telemetry."""

from .device_probe import ComputeDeviceInfo, probe_compute_device

__all__ = ["ComputeDeviceInfo", "probe_compute_device"]
