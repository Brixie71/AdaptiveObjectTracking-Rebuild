"""Configuration types and loading boundaries."""

from .loader import ConfigError, load_settings, resolve_rtsp_url
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
    "SourceType",
    "load_settings",
    "resolve_rtsp_url",
]
