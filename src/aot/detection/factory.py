"""Construct detectors only from verified configuration and artifacts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from aot.config.model_manifest import ModelManifest
from aot.config.model_store import resolve_model_artifact
from aot.config.types import DetectionConfig
from aot.telemetry.device_probe import ComputeDeviceInfo

from .backend import YoloBackend, isolated_backend_output, open_ultralytics_backend
from .types import Detector, DetectorLoadError
from .yolo import YoloDetector


def create_detector(
    config: DetectionConfig,
    manifest: ModelManifest,
    *,
    model_store: Path,
    compute: ComputeDeviceInfo,
    backend_factory: Callable[[Path], YoloBackend] = open_ultralytics_backend,
) -> Detector:
    """Create one detector without exposing resolver or backend failures."""

    if not config.enabled:
        raise DetectorLoadError("Detection is disabled")
    if config.model_id is None:
        raise DetectorLoadError("Detector model is not configured")
    try:
        artifact = resolve_model_artifact(
            manifest,
            model_id=config.model_id,
            expected_role="detector",
            model_store=model_store,
        )
    except Exception:
        raise DetectorLoadError("Detector model could not be resolved") from None

    backend: YoloBackend | None = None
    try:
        with isolated_backend_output():
            backend = backend_factory(artifact.path)
            return YoloDetector(
                artifact.model_id,
                config,
                backend,
                compute.resolved,
            )
    except Exception:
        if backend is not None:
            try:
                with isolated_backend_output():
                    backend.close()
            except Exception:
                pass
        raise DetectorLoadError("Detector could not be loaded") from None
