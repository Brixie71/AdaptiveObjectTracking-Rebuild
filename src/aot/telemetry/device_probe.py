"""Safe compute-device capability probing."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module, metadata
from typing import Any

from aot.config import ComputeConfig, ComputePreference


torch: Any | None = None


@dataclass(frozen=True)
class ComputeDeviceInfo:
    """Resolved compute choice and the capability evidence behind it."""

    requested: ComputePreference
    resolved: str
    cuda_available: bool
    cuda_device_count: int
    device_name: str | None
    torch_version: str
    fallback_reason: str | None


def _reason(prefix: str, error: Exception | None = None) -> str:
    if error is None:
        return prefix
    detail = " ".join(str(error).splitlines()).strip()
    return f"{prefix}: {detail}" if detail else prefix


def _cpu_result(
    config: ComputeConfig,
    torch_module: Any | None,
    *,
    cuda_available: bool,
    cuda_device_count: int,
    fallback_reason: str | None,
) -> ComputeDeviceInfo:
    return ComputeDeviceInfo(
        requested=config.preference,
        resolved="cpu",
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        device_name=None,
        torch_version=_torch_version(torch_module),
        fallback_reason=fallback_reason,
    )


def _torch_version(torch_module: Any | None) -> str:
    if torch_module is not None:
        return str(torch_module.__version__)
    try:
        return metadata.version("torch")
    except metadata.PackageNotFoundError:
        return "unavailable"


def probe_compute_device(config: ComputeConfig) -> ComputeDeviceInfo:
    """Resolve a safe compute device without assuming CUDA is usable."""

    global torch
    try:
        torch_module = torch if torch is not None else import_module("torch")
    except Exception as error:
        return _cpu_result(
            config,
            None,
            cuda_available=False,
            cuda_device_count=0,
            fallback_reason=_reason("PyTorch import failed", error),
        )

    try:
        cuda_available = bool(torch_module.cuda.is_available())
    except Exception as error:
        return _cpu_result(
            config,
            torch_module,
            cuda_available=False,
            cuda_device_count=0,
            fallback_reason=_reason("CUDA availability probe failed", error),
        )

    if not cuda_available:
        fallback = None
        if config.preference is not ComputePreference.CPU:
            fallback = "CUDA is unavailable; using CPU"
        return _cpu_result(
            config,
            torch_module,
            cuda_available=False,
            cuda_device_count=0,
            fallback_reason=fallback,
        )

    try:
        device_count = int(torch_module.cuda.device_count())
    except Exception as error:
        return _cpu_result(
            config,
            torch_module,
            cuda_available=True,
            cuda_device_count=0,
            fallback_reason=_reason("CUDA device-count probe failed", error),
        )

    if config.preference is ComputePreference.CPU:
        return _cpu_result(
            config,
            torch_module,
            cuda_available=True,
            cuda_device_count=device_count,
            fallback_reason=None,
        )

    if config.cuda_index >= device_count:
        return _cpu_result(
            config,
            torch_module,
            cuda_available=True,
            cuda_device_count=device_count,
            fallback_reason=(
                f"CUDA device index {config.cuda_index} is unavailable; "
                f"detected {device_count} device(s)"
            ),
        )

    try:
        device_name = str(torch_module.cuda.get_device_name(config.cuda_index))
    except Exception as error:
        return _cpu_result(
            config,
            torch_module,
            cuda_available=True,
            cuda_device_count=device_count,
            fallback_reason=_reason("CUDA device-name probe failed", error),
        )

    return ComputeDeviceInfo(
        requested=config.preference,
        resolved=f"cuda:{config.cuda_index}",
        cuda_available=True,
        cuda_device_count=device_count,
        device_name=device_name,
        torch_version=str(torch_module.__version__),
        fallback_reason=None,
    )
