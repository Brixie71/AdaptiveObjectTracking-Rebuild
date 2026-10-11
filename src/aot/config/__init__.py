"""Configuration types and loading boundaries."""

from .loader import ConfigError, load_settings, resolve_rtsp_url
from .model_manifest import (
    ModelManifest,
    ModelManifestError,
    ModelSpec,
    load_model_manifest,
)
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
