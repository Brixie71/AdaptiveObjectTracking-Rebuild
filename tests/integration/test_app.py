from __future__ import annotations

from pathlib import Path

import pytest

from aot.app import build_runtime, main
from aot.config.model_manifest import ModelManifest
from aot.config.types import (
    AppSettings,
    CameraSourceConfig,
    ComputeConfig,
    DetectionConfig,
    SourceType,
)
from aot.telemetry.device_probe import ComputeDeviceInfo


def source(source_id: str) -> CameraSourceConfig:
    return CameraSourceConfig(
        source_id, source_id, SourceType.USB, True, device_index=0
    )


def settings(*sources, detection=False, model_id="detector") -> AppSettings:
    return AppSettings(
        1,
        Path("runtime"),
        "INFO",
        ComputeConfig(),
        tuple(sources),
        detection=DetectionConfig(
            enabled=detection,
            model_id=model_id if detection else None,
            allowed_classes=("person", "forklift"),
        ),
    )


class FakeCapture:
    def start(self, _sources):
        pass

    def latest(self, _source_id):
        return None

    def snapshot(self):
        return {}

    def stop(self):
        pass


class FakeProcessing:
    def start(self):
        pass

    def request_stop(self):
        pass

    def join(self, _timeout=None):
        return True


def test_build_runtime_skips_all_detector_dependencies_when_disabled() -> None:
    calls = []
    runtime = build_runtime(
        settings(source("one")),
        Path("manifest.yaml"),
        Path("models"),
        manifest_loader=lambda _path: calls.append("manifest"),
        compute_probe=lambda _config: calls.append("compute"),
        detector_builder=lambda *_args, **_kwargs: calls.append("detector"),
        capture_manager_factory=lambda _config: FakeCapture(),
        processing_worker_factory=lambda *_args: FakeProcessing(),
    )
    runtime.start()
    runtime.stop()

    assert calls == []


def test_enabled_runtime_passes_exact_detector_composition_values() -> None:
    app_settings = settings(source("one"), detection=True)
    manifest = ModelManifest(schema_version=1, models=())
    compute = ComputeDeviceInfo(
        app_settings.compute.preference, "cpu", False, 0, None, "test", None
    )
    captured = {}

    class Detector:
        model_id = "detector"
        class_names = ()

        def close(self):
            captured["closed"] = True

    def detector_builder(config, loaded_manifest, *, model_store, compute):
        captured.update(
            config=config,
            manifest=loaded_manifest,
            model_store=model_store,
            compute=compute,
        )
        return Detector()

    runtime = build_runtime(
        app_settings,
        Path("manifest.yaml"),
        Path("models-root"),
        manifest_loader=lambda _path: manifest,
        compute_probe=lambda _config: compute,
        detector_builder=detector_builder,
        capture_manager_factory=lambda _config: FakeCapture(),
        processing_worker_factory=lambda *_args: FakeProcessing(),
    )
    runtime.start()
    runtime.stop()

    assert captured["config"] is app_settings.detection
    assert captured["manifest"] is manifest
    assert captured["model_store"] == Path("models-root")
    assert captured["compute"] is compute
    assert captured["closed"] is True


class FakeApplication:
    def __init__(self, argv, events, exit_code=0):
        self.events = events
        self.events.append(("qt", tuple(argv)))
        self.exit_code = exit_code

    def exec(self):
        self.events.append("exec")
        return self.exit_code


class FakeWindow:
    def __init__(self, events, shutdown_ok=True):
        self.events = events
        self.shutdown_ok = shutdown_ok

    def show(self):
        self.events.append("show")

    def wait_for_shutdown(self):
        self.events.append("shutdown")
        return self.shutdown_ok


def test_main_parses_paths_and_loads_configuration_before_qt(capsys) -> None:
    events = []

    def loader(app_path, camera_path, env_path):
        events.append(("load", app_path, camera_path, env_path))
        return settings(source("one"))

    exit_code = main(
        [
            "--app-config", "private-app.yaml",
            "--camera-config", "private-cameras.yaml",
            "--env-file", "private.env",
            "--model-manifest", "private-manifest.yaml",
            "--models-dir", "private-models",
        ],
        settings_loader=loader,
        application_factory=lambda argv: FakeApplication(argv, events),
        window_factory=lambda **_kwargs: FakeWindow(events),
        capture_manager_factory=lambda _config: FakeCapture(),
    )

    assert exit_code == 0
    assert events[0] == (
        "load",
        Path("private-app.yaml"),
        Path("private-cameras.yaml"),
        Path("private.env"),
    )
    assert events[1][0] == "qt"
    assert events[-2:] == ["exec", "shutdown"]
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("source_count", [0, 4])
def test_main_rejects_invalid_source_count_without_constructing_qt(
    source_count, capsys
) -> None:
    qt_calls = []
    exit_code = main(
        [],
        settings_loader=lambda *_args: settings(
            *(source(f"s{index}") for index in range(source_count))
        ),
        application_factory=lambda _argv: qt_calls.append("qt"),
    )

    assert exit_code == 1
    assert qt_calls == []
    error = capsys.readouterr().err
    assert error == "Application could not be initialized.\n"


def test_main_sanitizes_configuration_and_model_failures(capsys) -> None:
    def bad_loader(*_args):
        raise RuntimeError("PRIVATE-PATH rtsp://user:password@camera")

    assert main([], settings_loader=bad_loader) == 1
    assert capsys.readouterr().err == "Application configuration could not be loaded.\n"

    assert main(
        [],
        settings_loader=lambda *_args: settings(source("one"), detection=True),
        manifest_loader=lambda _path: (_ for _ in ()).throw(
            RuntimeError("PRIVATE-MODEL-PATH")
        ),
    ) == 1
    assert capsys.readouterr().err == "Application could not be initialized.\n"


def test_main_rejects_missing_model_declaration(capsys) -> None:
    broken = settings(source("one"))
    object.__setattr__(broken, "detection", DetectionConfig(enabled=True, model_id=None))

    assert main([], settings_loader=lambda *_args: broken) == 1
    assert capsys.readouterr().err == "Application could not be initialized.\n"


def test_main_returns_nonzero_when_window_composition_hits_detector_failure(capsys) -> None:
    def window_factory(**kwargs):
        kwargs["runtime"].start()
        raise AssertionError("unreachable")

    assert main(
        [],
        settings_loader=lambda *_args: settings(source("one"), detection=True),
        manifest_loader=lambda _path: ModelManifest(1, ()),
        compute_probe=lambda config: ComputeDeviceInfo(
            config.preference, "cpu", False, 0, None, "test", None
        ),
        detector_builder=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("PRIVATE-BACKEND")
        ),
        capture_manager_factory=lambda _config: FakeCapture(),
        application_factory=lambda argv: FakeApplication(argv, []),
        window_factory=window_factory,
    ) == 1
    assert capsys.readouterr().err == "Application could not be started.\n"


@pytest.mark.parametrize(("shutdown_ok", "expected"), [(True, 0), (False, 1)])
def test_main_waits_for_window_shutdown_and_reports_safe_failure(
    shutdown_ok, expected, capsys
) -> None:
    events = []
    result = main(
        [],
        settings_loader=lambda *_args: settings(source("one")),
        application_factory=lambda argv: FakeApplication(argv, events),
        window_factory=lambda **_kwargs: FakeWindow(events, shutdown_ok),
        capture_manager_factory=lambda _config: FakeCapture(),
    )

    assert result == expected
    assert events[-1] == "shutdown"
    error = capsys.readouterr().err
    assert error == ("" if shutdown_ok else "Application shutdown did not complete.\n")
