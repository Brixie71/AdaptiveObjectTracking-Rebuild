from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from aot.config.model_manifest import ModelManifestError, load_model_manifest


def _write_manifest(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_empty_version_one_manifest_is_valid(tmp_path) -> None:
    path = _write_manifest(tmp_path, "schema_version: 1\nmodels: []\n")

    manifest = load_model_manifest(path)

    assert manifest.schema_version == 1
    assert manifest.models == ()


def test_valid_model_declaration_preserves_relative_provenance(tmp_path) -> None:
    digest = "a" * 64
    path = _write_manifest(
        tmp_path,
        f"""\
schema_version: 1
models:
  - model_id: yolov8n
    role: detector
    path: yolo/yolov8n.pt
    source: https://example.invalid/yolov8n.pt
    sha256: {digest}
    required: true
""",
    )

    manifest = load_model_manifest(path)
    model = manifest.models[0]

    assert model.model_id == "yolov8n"
    assert model.role == "detector"
    assert model.path == Path("yolo/yolov8n.pt")
    assert model.source == "https://example.invalid/yolov8n.pt"
    assert model.sha256 == digest
    assert model.required is True
    with pytest.raises(FrozenInstanceError):
        model.required = False


@pytest.mark.parametrize(
    "text, unknown_key",
    [
        ("schema_version: 1\nmodels: []\nextra: true\n", "extra"),
        (
            """\
schema_version: 1
models:
  - model_id: model
    role: detector
    path: model.pt
    source: local
    unexpected: true
""",
            "unexpected",
        ),
    ],
)
def test_unknown_keys_are_rejected(tmp_path, text: str, unknown_key: str) -> None:
    path = _write_manifest(tmp_path, text)

    with pytest.raises(ModelManifestError) as error:
        load_model_manifest(path)

    assert unknown_key in str(error.value)
    assert str(path) in str(error.value)


def test_duplicate_model_ids_are_rejected(tmp_path) -> None:
    path = _write_manifest(
        tmp_path,
        """\
schema_version: 1
models:
  - model_id: duplicate
    role: detector
    path: first.pt
    source: local
  - model_id: duplicate
    role: detector
    path: second.pt
    source: local
""",
    )

    with pytest.raises(ModelManifestError) as error:
        load_model_manifest(path)

    assert "duplicate" in str(error.value)


def test_model_ids_are_unique_ignoring_case_and_whitespace(tmp_path) -> None:
    path = _write_manifest(
        tmp_path,
        """\
schema_version: 1
models:
  - model_id: Detector
    role: detector
    path: first.pt
    source: local
  - model_id: detector
    role: detector
    path: second.pt
    source: local
""",
    )

    with pytest.raises(ModelManifestError) as error:
        load_model_manifest(path)

    assert "duplicate" in str(error.value).lower()


@pytest.mark.parametrize(
    "unsafe_path",
    ["C:/models/model.pt", "/models/model.pt", "../historical/model.pt", "models/../../model.pt"],
)
def test_model_paths_must_stay_relative_to_model_store(tmp_path, unsafe_path: str) -> None:
    path = _write_manifest(
        tmp_path,
        f"""\
schema_version: 1
models:
  - model_id: unsafe
    role: detector
    path: {unsafe_path}
    source: local
""",
    )

    with pytest.raises(ModelManifestError) as error:
        load_model_manifest(path)

    assert "path" in str(error.value)


def test_unsupported_model_role_is_rejected(tmp_path) -> None:
    path = _write_manifest(
        tmp_path,
        """\
schema_version: 1
models:
  - model_id: unsupported
    role: classifier
    path: model.onnx
    source: local
""",
    )

    with pytest.raises(ModelManifestError) as error:
        load_model_manifest(path)

    assert "role" in str(error.value)


@pytest.mark.parametrize("digest", ["abc", "g" * 64, "a" * 63, "a" * 65])
def test_supplied_sha256_must_be_64_hexadecimal_characters(tmp_path, digest: str) -> None:
    path = _write_manifest(
        tmp_path,
        f"""\
schema_version: 1
models:
  - model_id: invalid-hash
    role: person_encoder
    path: reid/model.pt
    source: local
    sha256: {digest}
""",
    )

    with pytest.raises(ModelManifestError) as error:
        load_model_manifest(path)

    assert "sha256" in str(error.value)


@pytest.mark.parametrize(
    "role",
    ["detector", "face_detector", "person_encoder", "generic_encoder"],
)
def test_all_supported_roles_are_accepted(tmp_path, role: str) -> None:
    path = _write_manifest(
        tmp_path,
        f"""\
schema_version: 1
models:
  - model_id: model-{role}
    role: {role}
    path: {role}/model.onnx
    source: local
""",
    )

    manifest = load_model_manifest(path)

    assert manifest.models[0].role == role


@pytest.mark.parametrize("unsafe_path", [r"C:model.pt", r"\external\model.pt"])
def test_windows_drive_or_root_qualified_paths_are_rejected(tmp_path, unsafe_path: str) -> None:
    path = _write_manifest(
        tmp_path,
        f"""\
schema_version: 1
models:
  - model_id: unsafe
    role: detector
    path: '{unsafe_path}'
    source: local
""",
    )

    with pytest.raises(ModelManifestError) as error:
        load_model_manifest(path)

    assert "relative" in str(error.value)


def test_mixed_type_unknown_manifest_keys_raise_boundary_error(tmp_path) -> None:
    path = _write_manifest(
        tmp_path,
        "schema_version: 1\nmodels: []\nz_unknown: unsupported\n1: unsupported\n",
    )

    with pytest.raises(ModelManifestError) as error:
        load_model_manifest(path)

    assert "must be strings" in str(error.value)


def test_malformed_manifest_yaml_does_not_echo_credentials(tmp_path) -> None:
    secret = "admin:" + "model-pass"
    token = "manifest-" + "token"
    path = _write_manifest(
        tmp_path,
        f"schema_version: 1\nmodels: [rtsp://{secret}@host/model?token={token}\n",
    )

    with pytest.raises(ModelManifestError) as error:
        load_model_manifest(path)

    message = str(error.value)
    assert "Invalid YAML" in message
    assert secret not in message
    assert token not in message
