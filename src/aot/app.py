"""Composition root and native application entry point."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import QApplication

from aot.capture.manager import CaptureManager
from aot.capture.types import FramePacket
from aot.config.loader import load_settings
from aot.config.model_manifest import load_model_manifest
from aot.config.types import AppSettings
from aot.detection.backend import isolated_backend_output
from aot.detection.factory import create_detector
from aot.hardware_testing.detector_probe import probe_detector
from aot.hardware_testing.source_probe import probe_sources
from aot.hardware_testing.types import DetectorTestResult, HardwareTestStatus
from aot.hardware_testing.usb_discovery import discover_usb_devices
from aot.runtime.live_runtime import LiveRuntime
from aot.selection.target_selector import TargetSelector
from aot.telemetry.device_probe import probe_compute_device
from aot.ui.main_window import MainWindow
from aot.ui.settings_dialog import SettingsDialog


class AppCompositionError(RuntimeError):
    """Fixed-boundary composition failure."""


def _validate_settings(settings: AppSettings) -> None:
    enabled = tuple(source for source in settings.sources if source.enabled)
    if not 1 <= len(enabled) <= 3:
        raise AppCompositionError("Invalid enabled source count")
    if settings.detection.enabled and settings.detection.model_id is None:
        raise AppCompositionError("Detector model is not configured")


def _prepare_detector_factory(
    settings: AppSettings,
    model_manifest_path: Path,
    models_dir: Path,
    *,
    manifest_loader,
    compute_probe,
    detector_builder,
):
    if not settings.detection.enabled:
        return (lambda: None), None
    manifest = manifest_loader(model_manifest_path)
    compute = compute_probe(settings.compute)

    def detector_factory():
        return detector_builder(
            settings.detection,
            manifest,
            model_store=models_dir,
            compute=compute,
        )

    return detector_factory, compute


def _runtime_from_factory(
    settings: AppSettings,
    detector_factory,
    capture_manager_factory,
    processing_worker_factory=None,
) -> LiveRuntime:
    kwargs = {}
    if processing_worker_factory is not None:
        kwargs["processing_worker_factory"] = processing_worker_factory
    return LiveRuntime(
        settings,
        capture_manager_factory,
        detector_factory,
        **kwargs,
    )


def build_runtime(
    settings: AppSettings,
    model_manifest_path: Path,
    models_dir: Path,
    *,
    manifest_loader=load_model_manifest,
    compute_probe=probe_compute_device,
    detector_builder=create_detector,
    capture_manager_factory=CaptureManager,
    processing_worker_factory=None,
) -> LiveRuntime:
    """Validate composition and build a fresh-session live runtime."""
    _validate_settings(settings)
    detector_factory, _compute = _prepare_detector_factory(
        settings,
        Path(model_manifest_path),
        Path(models_dir),
        manifest_loader=manifest_loader,
        compute_probe=compute_probe,
        detector_builder=detector_builder,
    )
    return _runtime_from_factory(
        settings,
        detector_factory,
        capture_manager_factory,
        processing_worker_factory,
    )


def _unsupported_detector_result() -> DetectorTestResult:
    now = datetime.now(timezone.utc)
    return DetectorTestResult(
        "unconfigured",
        HardwareTestStatus.UNSUPPORTED,
        "unresolved",
        0,
        0,
        0,
        None,
        None,
        0,
        now,
        now,
        "Detection is disabled",
    )


def _detector_probe_service(detector_factory, compute):
    if compute is None:
        return _unsupported_detector_result

    def run_probe():
        detector = detector_factory()
        packet = FramePacket(
            "hardware-test",
            0,
            time.monotonic_ns(),
            np.zeros((480, 640, 3), dtype=np.uint8),
        )
        return probe_detector(detector, packet, device=compute.resolved)

    return run_probe


def create_main_window(
    *,
    runtime: LiveRuntime,
    settings: AppSettings,
    detector_probe_fn: Callable,
    probe_sources_fn=probe_sources,
    discover_usb_fn=discover_usb_devices,
    target_selector_factory: Callable[[], TargetSelector] = TargetSelector,
) -> MainWindow:
    """Compose widgets with probe services while keeping widgets service-free."""

    def settings_factory(parent):
        return SettingsDialog(
            settings.sources,
            probe_sources_fn,
            discover_usb_fn,
            detector_probe_fn,
            parent,
        )

    target_selector = target_selector_factory()
    return MainWindow(
        runtime,
        target_selector=target_selector,
        settings_factory=settings_factory,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive Object Tracking")
    parser.add_argument("--app-config", type=Path, default=Path("config/app.example.yaml"))
    parser.add_argument(
        "--camera-config", type=Path, default=Path("config/cameras.example.yaml")
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--model-manifest", type=Path, default=Path("models/manifest.yaml")
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    settings_loader=load_settings,
    manifest_loader=load_model_manifest,
    compute_probe=probe_compute_device,
    detector_builder=create_detector,
    capture_manager_factory=CaptureManager,
    application_factory=QApplication,
    window_factory=create_main_window,
    target_selector_factory=TargetSelector,
) -> int:
    """Run the native Qt application with sanitized process-boundary errors."""
    args = _parser().parse_args(argv)
    try:
        with isolated_backend_output():
            settings = settings_loader(
                args.app_config, args.camera_config, args.env_file
            )
    except Exception:
        print("Application configuration could not be loaded.", file=sys.stderr)
        return 1

    try:
        _validate_settings(settings)
        with isolated_backend_output():
            detector_factory, compute = _prepare_detector_factory(
                settings,
                args.model_manifest,
                args.models_dir,
                manifest_loader=manifest_loader,
                compute_probe=compute_probe,
                detector_builder=detector_builder,
            )
        runtime = _runtime_from_factory(
            settings, detector_factory, capture_manager_factory
        )
    except Exception:
        print("Application could not be initialized.", file=sys.stderr)
        return 1

    application = application_factory(["adaptive-object-tracking"])
    try:
        window = window_factory(
            runtime=runtime,
            settings=settings,
            detector_probe_fn=_detector_probe_service(detector_factory, compute),
            target_selector_factory=target_selector_factory,
        )
        window.show()
    except Exception:
        try:
            runtime.stop()
        except Exception:
            pass
        print("Application could not be started.", file=sys.stderr)
        return 1

    qt_result = application.exec()
    if not window.wait_for_shutdown():
        print("Application shutdown did not complete.", file=sys.stderr)
        return 1
    return 0 if qt_result == 0 else 1
