"""Immutable application configuration records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class SourceType(str, Enum):
    """Supported source categories."""

    USB = "usb"
    RTSP = "rtsp"
    OFFLINE = "offline"


class ComputePreference(str, Enum):
    """User-selectable compute preferences."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True)
class ComputeConfig:
    """Compute-device selection settings."""

    preference: ComputePreference = ComputePreference.AUTO
    cuda_index: int = 0


@dataclass(frozen=True)
class CameraSourceConfig:
    """Non-sensitive configuration for one camera or video source."""

    source_id: str
    name: str
    source_type: SourceType
    enabled: bool = False
    device_index: int | None = None
    rtsp_env_key: str | None = None
    offline_path: Path | None = None
    width: int = 640
    height: int = 480
    fps: float = 15.0
    rotation: int = 0


@dataclass(frozen=True)
class AppSettings:
    """Validated application configuration."""

    schema_version: int
    runtime_dir: Path
    log_level: str
    compute: ComputeConfig
    sources: tuple[CameraSourceConfig, ...]
