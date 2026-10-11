from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from aot.config import ComputeConfig, ComputePreference
from aot.telemetry import device_probe
from aot.telemetry.device_probe import probe_compute_device


class FakeCuda:
    def __init__(
        self,
        *,
        available: bool,
        count: int = 0,
        name: str = "Test GPU",
        name_error: Exception | None = None,
    ) -> None:
        self.available = available
        self.count = count
        self.name = name
        self.name_error = name_error

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        if not self.available:
            raise AssertionError("device_count called while CUDA unavailable")
        return self.count

    def get_device_name(self, index: int) -> str:
        if not self.available:
            raise AssertionError("get_device_name called while CUDA unavailable")
        if self.name_error is not None:
            raise self.name_error
        return f"{self.name} {index}"


class FakeTorch:
    __version__ = "test-torch"

    def __init__(self, cuda: FakeCuda) -> None:
        self.cuda = cuda


def _install_torch(monkeypatch, cuda: FakeCuda) -> None:
    monkeypatch.setattr(device_probe, "torch", FakeTorch(cuda))


def test_explicit_cpu_remains_on_cpu_when_cuda_is_present(monkeypatch) -> None:
    _install_torch(monkeypatch, FakeCuda(available=True, count=2))

    result = probe_compute_device(ComputeConfig(ComputePreference.CPU))

    assert result.requested is ComputePreference.CPU
    assert result.resolved == "cpu"
    assert result.cuda_available is True
    assert result.cuda_device_count == 2
    assert result.device_name is None
    assert result.fallback_reason is None


def test_auto_selects_requested_cuda_device(monkeypatch) -> None:
    _install_torch(monkeypatch, FakeCuda(available=True, count=2, name="NVIDIA Test"))

    result = probe_compute_device(
        ComputeConfig(preference=ComputePreference.AUTO, cuda_index=1)
    )

    assert result.resolved == "cuda:1"
    assert result.device_name == "NVIDIA Test 1"
    assert result.fallback_reason is None


def test_auto_falls_back_without_calling_cuda_only_queries(monkeypatch) -> None:
    _install_torch(monkeypatch, FakeCuda(available=False))

    result = probe_compute_device(ComputeConfig())

    assert result.resolved == "cpu"
    assert result.cuda_available is False
    assert result.cuda_device_count == 0
    assert result.fallback_reason


def test_requested_cuda_falls_back_when_cuda_is_unavailable(monkeypatch) -> None:
    _install_torch(monkeypatch, FakeCuda(available=False))

    result = probe_compute_device(ComputeConfig(ComputePreference.CUDA))

    assert result.resolved == "cpu"
    assert "unavailable" in result.fallback_reason.lower()


def test_out_of_range_cuda_index_falls_back_to_cpu(monkeypatch) -> None:
    _install_torch(monkeypatch, FakeCuda(available=True, count=1))

    result = probe_compute_device(
        ComputeConfig(preference=ComputePreference.CUDA, cuda_index=3)
    )

    assert result.resolved == "cpu"
    assert result.cuda_available is True
    assert result.cuda_device_count == 1
    assert result.fallback_reason


def test_device_name_failure_becomes_cpu_fallback(monkeypatch) -> None:
    _install_torch(
        monkeypatch,
        FakeCuda(available=True, count=1, name_error=RuntimeError("driver query failed")),
    )

    result = probe_compute_device(ComputeConfig())

    assert result.resolved == "cpu"
    assert result.device_name is None
    assert "driver query failed" in result.fallback_reason


def test_torch_import_failure_becomes_cpu_fallback(monkeypatch) -> None:
    monkeypatch.setattr(device_probe, "torch", None)

    def fail_import(_name: str):
        raise OSError("c10.dll failed")

    monkeypatch.setattr(device_probe, "import_module", fail_import)

    result = probe_compute_device(ComputeConfig())

    assert result.resolved == "cpu"
    assert result.cuda_available is False
    assert "PyTorch import failed" in result.fallback_reason


def test_device_report_is_immutable(monkeypatch) -> None:
    _install_torch(monkeypatch, FakeCuda(available=False))
    result = probe_compute_device(ComputeConfig())

    with pytest.raises(FrozenInstanceError):
        result.resolved = "cuda:0"
