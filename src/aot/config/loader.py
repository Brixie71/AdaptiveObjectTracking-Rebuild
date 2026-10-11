"""Strict YAML configuration loading and environment-only secret resolution."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .types import (
    AppSettings,
    CameraSourceConfig,
    ComputeConfig,
    ComputePreference,
    SourceType,
)


class ConfigError(ValueError):
    """Raised when project configuration is invalid or incomplete."""


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"Cannot read configuration {path}: {error}") from error
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        location = (
            f" at line {mark.line + 1}, column {mark.column + 1}"
            if mark is not None
            else ""
        )
        raise ConfigError(f"Invalid YAML in {path}{location}") from error
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration {path} must contain a mapping")
    return value


def _reject_unknown(
    value: Mapping[str, Any],
    allowed: set[str],
    path: Path,
    context: str,
) -> None:
    non_string = next((key for key in value if not isinstance(key, str)), None)
    if non_string is not None:
        raise ConfigError(f"Keys in {context} must be strings at {path}")
    unknown = sorted(key for key in value if key not in allowed)
    if unknown:
        raise ConfigError(f"Unknown key {unknown[0]!r} in {context} at {path}")


def _schema_version(value: Mapping[str, Any], path: Path) -> int:
    version = value.get("schema_version")
    if version != 1:
        raise ConfigError(f"Unsupported schema_version {version!r} in {path}; expected 1")
    return 1


def _positive_number(value: Any, field: str, path: Path) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ConfigError(f"{field} must be a finite positive number in {path}")
    return float(value)


def _positive_integer(value: Any, field: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{field} must be a positive integer in {path}")
    return value


def _nonblank(value: Any, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a nonblank string in {path}")
    return value.strip()


def _load_compute(value: Any, path: Path) -> ComputeConfig:
    if value is None:
        return ComputeConfig()
    if not isinstance(value, dict):
        raise ConfigError(f"compute must be a mapping in {path}")
    _reject_unknown(value, {"preference", "cuda_index"}, path, "compute")
    try:
        preference = ComputePreference(value.get("preference", "auto"))
    except ValueError as error:
        raise ConfigError(f"Unsupported compute preference in {path}") from error
    cuda_index = value.get("cuda_index", 0)
    if isinstance(cuda_index, bool) or not isinstance(cuda_index, int) or cuda_index < 0:
        raise ConfigError(f"compute.cuda_index must be a nonnegative integer in {path}")
    return ComputeConfig(preference=preference, cuda_index=cuda_index)


def _load_source(value: Any, path: Path) -> CameraSourceConfig:
    if not isinstance(value, dict):
        raise ConfigError(f"Each source in {path} must be a mapping")
    allowed = {
        "source_id",
        "name",
        "type",
        "enabled",
        "device_index",
        "rtsp_env_key",
        "offline_path",
        "width",
        "height",
        "fps",
        "rotation",
    }
    _reject_unknown(value, allowed, path, "source")

    source_id = _nonblank(value.get("source_id"), "source_id", path)
    name = _nonblank(value.get("name"), "name", path)
    try:
        source_type = SourceType(value.get("type"))
    except ValueError as error:
        raise ConfigError(f"Unsupported source type for {source_id!r} in {path}") from error

    enabled = value.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError(f"enabled must be true or false for {source_id!r} in {path}")

    width = _positive_integer(value.get("width", 640), "width", path)
    height = _positive_integer(value.get("height", 480), "height", path)
    fps = _positive_number(value.get("fps", 15.0), "fps", path)
    rotation = value.get("rotation", 0)
    if (
        isinstance(rotation, bool)
        or not isinstance(rotation, int)
        or rotation not in {0, 90, 180, 270}
    ):
        raise ConfigError(f"rotation must be one of 0, 90, 180, 270 in {path}")

    device_index = value.get("device_index")
    rtsp_env_key = value.get("rtsp_env_key")
    offline_value = value.get("offline_path")
    populated = sum(item is not None for item in (device_index, rtsp_env_key, offline_value))
    if populated != 1:
        raise ConfigError(
            f"Source {source_id!r} must define exactly one source-specific field in {path}"
        )

    offline_path: Path | None = None
    if source_type is SourceType.USB:
        if (
            isinstance(device_index, bool)
            or not isinstance(device_index, int)
            or device_index < 0
            or rtsp_env_key is not None
            or offline_value is not None
        ):
            raise ConfigError(f"USB source {source_id!r} requires a nonnegative device_index in {path}")
    elif source_type is SourceType.RTSP:
        if device_index is not None or offline_value is not None:
            raise ConfigError(f"RTSP source {source_id!r} accepts only rtsp_env_key in {path}")
        rtsp_env_key = _nonblank(rtsp_env_key, "rtsp_env_key", path)
    else:
        if device_index is not None or rtsp_env_key is not None:
            raise ConfigError(f"Offline source {source_id!r} accepts only offline_path in {path}")
        offline_text = _nonblank(offline_value, "offline_path", path)
        offline_path = Path(offline_text)
        if not offline_path.is_absolute():
            offline_path = (path.parent / offline_path).resolve()

    return CameraSourceConfig(
        source_id=source_id,
        name=name,
        source_type=source_type,
        enabled=enabled,
        device_index=device_index,
        rtsp_env_key=rtsp_env_key,
        offline_path=offline_path,
        width=width,
        height=height,
        fps=fps,
        rotation=rotation,
    )


def load_settings(
    app_path: Path,
    cameras_path: Path,
    env_path: Path | None = None,
) -> AppSettings:
    """Load and validate non-sensitive application and source configuration."""

    app_path = Path(app_path).resolve()
    cameras_path = Path(cameras_path).resolve()
    if env_path is not None:
        load_dotenv(dotenv_path=Path(env_path), override=False)

    app = _load_mapping(app_path)
    cameras = _load_mapping(cameras_path)
    _reject_unknown(
        app,
        {"schema_version", "runtime_dir", "log_level", "compute"},
        app_path,
        "application configuration",
    )
    _reject_unknown(cameras, {"schema_version", "sources"}, cameras_path, "camera configuration")
    version = _schema_version(app, app_path)
    _schema_version(cameras, cameras_path)

    runtime_text = _nonblank(app.get("runtime_dir", "runtime"), "runtime_dir", app_path)
    runtime_dir = Path(runtime_text)
    if not runtime_dir.is_absolute():
        runtime_dir = (app_path.parent / runtime_dir).resolve()
    log_level = _nonblank(app.get("log_level", "INFO"), "log_level", app_path).upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError(f"Unsupported log_level {log_level!r} in {app_path}")

    raw_sources = cameras.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ConfigError(f"sources must be a list in {cameras_path}")
    sources = tuple(_load_source(item, cameras_path) for item in raw_sources)
    source_ids = [source.source_id for source in sources]
    if len(source_ids) != len(set(source_ids)):
        raise ConfigError(f"source_id values must be unique in {cameras_path}")

    return AppSettings(
        schema_version=version,
        runtime_dir=runtime_dir,
        log_level=log_level,
        compute=_load_compute(app.get("compute"), app_path),
        sources=sources,
    )


def resolve_rtsp_url(
    source: CameraSourceConfig,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Resolve one RTSP URL without storing it on the configuration object."""

    if source.source_type is not SourceType.RTSP or not source.rtsp_env_key:
        raise ConfigError(f"Source {source.source_id!r} is not configured as RTSP")
    values = os.environ if environment is None else environment
    resolved = values.get(source.rtsp_env_key)
    if not resolved or not resolved.strip():
        raise ConfigError(f"Required RTSP environment variable {source.rtsp_env_key!r} is not set")
    return resolved
