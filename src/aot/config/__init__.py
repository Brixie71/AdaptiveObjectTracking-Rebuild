"""Configuration types and loading boundaries.

Third-party-backed loaders are exposed lazily so environment diagnostics can
start and report missing bootstrap dependencies instead of crashing on import.
"""

from typing import Any

from .types import (
    AppSettings,
    CameraSourceConfig,
    ComputeConfig,
    ComputePreference,
    SourceType,
)

__all__ = [
    "AppSettings",
    "CameraSourceConfig",
    "ConfigError",
    "ComputeConfig",
    "ComputePreference",
    "ModelManifest",
    "ModelManifestError",
    "ModelSpec",
    "SourceType",
    "load_settings",
    "load_model_manifest",
    "resolve_rtsp_url",
]


def __getattr__(name: str) -> Any:
    if name in {"ConfigError", "load_settings", "resolve_rtsp_url"}:
        from . import loader

        value = getattr(loader, name)
    elif name in {"ModelManifest", "ModelManifestError", "ModelSpec", "load_model_manifest"}:
        from . import model_manifest

        value = getattr(model_manifest, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
