from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from aot.config.model_manifest import ModelManifest, ModelSpec
from aot.config.model_store import ModelArtifactError, resolve_model_artifact


PAYLOAD = b"detector-bytes"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def _manifest(
    path: Path,
    *,
    model_id: str = "yolov8n-coco",
    role: str = "detector",
    sha256: str | None = DIGEST,
) -> ModelManifest:
    return ModelManifest(
        schema_version=1,
        models=(
            ModelSpec(
                model_id=model_id,
                role=role,
                path=path,
                source="owner-approved artifact",
                sha256=sha256,
                required=True,
            ),
        ),
    )


def test_resolver_returns_only_a_verified_contained_artifact(tmp_path) -> None:
    store = tmp_path / "models"
    artifact = store / "yolo" / "detector.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(PAYLOAD)

    resolved = resolve_model_artifact(
        _manifest(Path("yolo/detector.pt")),
        model_id="yolov8n-coco",
        expected_role="detector",
        model_store=store,
    )

    assert resolved.model_id == "yolov8n-coco"
    assert resolved.role == "detector"
    assert resolved.path == artifact.resolve()
    assert resolved.source == "owner-approved artifact"
    assert resolved.sha256 == DIGEST


@pytest.mark.parametrize(
    ("manifest", "model_id", "expected_role"),
    [
        (_manifest(Path("model.pt")), "missing", "detector"),
        (_manifest(Path("model.pt"), role="person_encoder"), "yolov8n-coco", "detector"),
        (_manifest(Path("model.pt"), sha256=None), "yolov8n-coco", "detector"),
        (_manifest(Path("model.pt")), "yolov8n-coco", "unsupported"),
    ],
)
def test_resolver_rejects_invalid_selection(
    tmp_path,
    manifest: ModelManifest,
    model_id: str,
    expected_role: str,
) -> None:
    (tmp_path / "model.pt").write_bytes(PAYLOAD)

    with pytest.raises(ModelArtifactError):
        resolve_model_artifact(
            manifest,
            model_id=model_id,
            expected_role=expected_role,
            model_store=tmp_path,
        )


def test_resolver_rejects_missing_directory_and_hash_mismatch(tmp_path) -> None:
    missing = _manifest(Path("missing.pt"))
    with pytest.raises(ModelArtifactError):
        resolve_model_artifact(
            missing,
            model_id="yolov8n-coco",
            expected_role="detector",
            model_store=tmp_path,
        )

    directory = tmp_path / "directory.pt"
    directory.mkdir()
    with pytest.raises(ModelArtifactError):
        resolve_model_artifact(
            _manifest(Path("directory.pt")),
            model_id="yolov8n-coco",
            expected_role="detector",
            model_store=tmp_path,
        )

    artifact = tmp_path / "mismatch.pt"
    artifact.write_bytes(b"wrong bytes")
    with pytest.raises(ModelArtifactError):
        resolve_model_artifact(
            _manifest(Path("mismatch.pt")),
            model_id="yolov8n-coco",
            expected_role="detector",
            model_store=tmp_path,
        )


def test_resolver_rejects_manifest_path_outside_model_store(tmp_path) -> None:
    store = tmp_path / "models"
    store.mkdir()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(PAYLOAD)

    with pytest.raises(ModelArtifactError):
        resolve_model_artifact(
            _manifest(Path("../outside.pt")),
            model_id="yolov8n-coco",
            expected_role="detector",
            model_store=store,
        )


def test_resolver_rejects_link_that_escapes_model_store(tmp_path) -> None:
    store = tmp_path / "models"
    store.mkdir()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(PAYLOAD)
    link = store / "linked.pt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"filesystem links unavailable: {error}")

    with pytest.raises(ModelArtifactError):
        resolve_model_artifact(
            _manifest(Path("linked.pt")),
            model_id="yolov8n-coco",
            expected_role="detector",
            model_store=store,
        )
