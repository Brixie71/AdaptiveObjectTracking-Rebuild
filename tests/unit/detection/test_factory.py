from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from aot.config.model_manifest import ModelManifest, ModelSpec
from aot.config.types import ComputePreference, DetectionConfig
from aot.detection.factory import create_detector
from aot.detection.types import DetectorLoadError
from aot.telemetry.device_probe import ComputeDeviceInfo

from .fakes import FakeYoloBackend


def _config(*, enabled: bool = True, classes: tuple[str, ...] = ("person",)) -> DetectionConfig:
    return DetectionConfig(
        enabled=enabled,
        model_id="yolov8n-coco" if enabled else None,
        allowed_classes=classes,
    )


def _compute(device: str = "cpu") -> ComputeDeviceInfo:
    return ComputeDeviceInfo(
        requested=ComputePreference.CPU,
        resolved=device,
        cuda_available=device.startswith("cuda"),
        cuda_device_count=1 if device.startswith("cuda") else 0,
        device_name="Test GPU" if device.startswith("cuda") else None,
        torch_version="test",
        fallback_reason=None,
    )


def _manifest(store: Path) -> tuple[ModelManifest, Path]:
    artifact = store / "yolo" / "yolov8n.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"verified model")
    digest = hashlib.sha256(b"verified model").hexdigest()
    return (
        ModelManifest(
            schema_version=1,
            models=(
                ModelSpec(
                    model_id="yolov8n-coco",
                    role="detector",
                    path=Path("yolo/yolov8n.pt"),
                    source="owner-approved",
                    sha256=digest,
                    required=True,
                ),
            ),
        ),
        artifact.resolve(),
    )


def test_factory_passes_only_verified_path_and_resolved_device(tmp_path) -> None:
    manifest, artifact = _manifest(tmp_path)
    received_paths: list[Path] = []

    def backend_factory(path: Path) -> FakeYoloBackend:
        received_paths.append(path)
        return FakeYoloBackend()

    detector = create_detector(
        _config(),
        manifest,
        model_store=tmp_path,
        compute=_compute("cuda:0"),
        backend_factory=backend_factory,
    )

    assert received_paths == [artifact]
    assert detector.model_id == "yolov8n-coco"
    assert detector.device == "cuda:0"


def test_factory_rejects_disabled_detection_before_loading(tmp_path) -> None:
    manifest, _ = _manifest(tmp_path)

    with pytest.raises(DetectorLoadError, match="disabled"):
        create_detector(
            _config(enabled=False),
            manifest,
            model_store=tmp_path,
            compute=_compute(),
        )


def test_factory_closes_backend_when_allowlist_validation_fails(tmp_path) -> None:
    manifest, _ = _manifest(tmp_path)
    backend = FakeYoloBackend()

    with pytest.raises(DetectorLoadError):
        create_detector(
            _config(classes=("unknown",)),
            manifest,
            model_store=tmp_path,
            compute=_compute(),
            backend_factory=lambda _path: backend,
        )

    assert backend.close_count == 1


def test_factory_hides_resolver_and_backend_details(tmp_path) -> None:
    secret = "rtsp://admin:model-pass@example.invalid/model.pt"
    manifest, _ = _manifest(tmp_path)

    with pytest.raises(DetectorLoadError) as missing_error:
        create_detector(
            DetectionConfig(enabled=True, model_id="missing"),
            manifest,
            model_store=tmp_path,
            compute=_compute(),
        )
    with pytest.raises(DetectorLoadError) as backend_error:
        create_detector(
            _config(),
            manifest,
            model_store=tmp_path,
            compute=_compute(),
            backend_factory=lambda _path: (_ for _ in ()).throw(RuntimeError(secret)),
        )

    assert secret not in str(missing_error.value)
    assert secret not in str(backend_error.value)
