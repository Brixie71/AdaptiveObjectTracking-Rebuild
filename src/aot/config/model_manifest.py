"""Validated model declarations and provenance metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml


SUPPORTED_MODEL_ROLES = {
    "detector",
    "face_detector",
    "person_encoder",
    "generic_encoder",
}
SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


class ModelManifestError(ValueError):
    """Raised when the model manifest is unsafe or invalid."""


@dataclass(frozen=True)
class ModelSpec:
    """One model artifact and its declared provenance."""

    model_id: str
    role: str
    path: Path
    source: str
    sha256: str | None = None
    required: bool = False


@dataclass(frozen=True)
class ModelManifest:
    """Versioned collection of model declarations."""

    schema_version: int
    models: tuple[ModelSpec, ...]


def _nonblank(value: Any, field: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelManifestError(f"{field} must be a nonblank string in {path}")
    return value.strip()


def _reject_unknown(
    value: dict[str, Any],
    allowed: set[str],
    path: Path,
    context: str,
) -> None:
    non_string = next((key for key in value if not isinstance(key, str)), None)
    if non_string is not None:
        raise ModelManifestError(f"Keys in {context} must be strings at {path}")
    unknown = sorted(key for key in value if key not in allowed)
    if unknown:
        raise ModelManifestError(f"Unknown key {unknown[0]!r} in {context} at {path}")


def _safe_relative_path(value: Any, manifest_path: Path) -> Path:
    text = _nonblank(value, "path", manifest_path)
    model_path = Path(text)
    posix_path = PurePosixPath(text)
    windows_path = PureWindowsPath(text)
    is_unsafe = (
        posix_path.is_absolute()
        or bool(windows_path.drive)
        or bool(windows_path.root)
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    )
    if is_unsafe:
        raise ModelManifestError(f"Model path must remain relative to the model store in {manifest_path}")
    return model_path


def _parse_model(value: Any, manifest_path: Path) -> ModelSpec:
    if not isinstance(value, dict):
        raise ModelManifestError(f"Each model in {manifest_path} must be a mapping")
    _reject_unknown(
        value,
        {"model_id", "role", "path", "source", "sha256", "required"},
        manifest_path,
        "model declaration",
    )
    model_id = _nonblank(value.get("model_id"), "model_id", manifest_path)
    role = _nonblank(value.get("role"), "role", manifest_path)
    if role not in SUPPORTED_MODEL_ROLES:
        raise ModelManifestError(f"Unsupported model role {role!r} in {manifest_path}")
    source = _nonblank(value.get("source"), "source", manifest_path)
    digest = value.get("sha256")
    if digest is not None and (
        not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None
    ):
        raise ModelManifestError(f"sha256 must be 64 hexadecimal characters in {manifest_path}")
    required = value.get("required", False)
    if not isinstance(required, bool):
        raise ModelManifestError(f"required must be true or false in {manifest_path}")
    return ModelSpec(
        model_id=model_id,
        role=role,
        path=_safe_relative_path(value.get("path"), manifest_path),
        source=source,
        sha256=digest.lower() if digest is not None else None,
        required=required,
    )


def load_model_manifest(path: Path) -> ModelManifest:
    """Load a strict version-1 model manifest without loading model weights."""

    path = Path(path).resolve()
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ModelManifestError(f"Cannot read model manifest {path}: {error}") from error
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        location = (
            f" at line {mark.line + 1}, column {mark.column + 1}"
            if mark is not None
            else ""
        )
        raise ModelManifestError(f"Invalid YAML in model manifest {path}{location}") from error
    if not isinstance(value, dict):
        raise ModelManifestError(f"Model manifest {path} must contain a mapping")
    _reject_unknown(value, {"schema_version", "models"}, path, "model manifest")
    if value.get("schema_version") != 1:
        raise ModelManifestError(f"Unsupported schema_version in model manifest {path}; expected 1")
    raw_models = value.get("models", [])
    if not isinstance(raw_models, list):
        raise ModelManifestError(f"models must be a list in {path}")
    models = tuple(_parse_model(item, path) for item in raw_models)
    model_ids = [model.model_id.casefold() for model in models]
    if len(model_ids) != len(set(model_ids)):
        raise ModelManifestError(f"Duplicate model_id in {path}")
    return ModelManifest(schema_version=1, models=models)
