"""Resolve and verify model artifacts before third-party loaders see them."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from pathlib import Path

from .model_manifest import ModelManifest, SUPPORTED_MODEL_ROLES


class ModelArtifactError(ValueError):
    """Raised when a selected model artifact cannot be trusted."""


@dataclass(frozen=True)
class ResolvedModelArtifact:
    """A contained regular file whose declared digest has been verified."""

    model_id: str
    role: str
    path: Path
    source: str
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_artifact(
    manifest: ModelManifest,
    *,
    model_id: str,
    expected_role: str,
    model_store: Path,
) -> ResolvedModelArtifact:
    """Return a selected artifact only after containment and digest checks."""

    if expected_role not in SUPPORTED_MODEL_ROLES:
        raise ModelArtifactError("Unsupported expected model role")
    selected = next(
        (
            model
            for model in manifest.models
            if model.model_id.casefold() == model_id.casefold()
        ),
        None,
    )
    if selected is None:
        raise ModelArtifactError("Selected model is not declared")
    if selected.role != expected_role:
        raise ModelArtifactError("Selected model has an incompatible role")
    if selected.sha256 is None:
        raise ModelArtifactError("Selected model has no declared SHA-256")

    try:
        store = Path(model_store).resolve(strict=True)
        artifact = (store / selected.path).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ModelArtifactError("Selected model artifact is unavailable") from error
    if not artifact.is_relative_to(store):
        raise ModelArtifactError("Selected model artifact escapes the model store")
    if not artifact.is_file():
        raise ModelArtifactError("Selected model artifact is not a regular file")

    try:
        actual_digest = _sha256(artifact)
    except OSError as error:
        raise ModelArtifactError("Selected model artifact could not be read") from error
    if not hmac.compare_digest(actual_digest, selected.sha256.lower()):
        raise ModelArtifactError("Selected model artifact failed SHA-256 verification")
    return ResolvedModelArtifact(
        model_id=selected.model_id,
        role=selected.role,
        path=artifact,
        source=selected.source,
        sha256=selected.sha256.lower(),
    )
