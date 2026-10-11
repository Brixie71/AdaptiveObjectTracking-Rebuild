from __future__ import annotations

import json
from importlib import metadata
from pathlib import Path

import pytest

from aot.config import ComputePreference
from aot.telemetry import environment
from aot.telemetry.device_probe import ComputeDeviceInfo
from aot.telemetry.environment import CheckStatus, check_environment
from aot.telemetry.logging_setup import LoggingSetupError


REQUIRED_DISTRIBUTIONS = {
    "torch",
    "torchvision",
    "ultralytics",
    "opencv-python",
    "numpy",
    "Pillow",
    "PyQt5",
    "PyYAML",
    "python-dotenv",
    "psutil",
    "onnx",
    "onnxruntime",
    "pytest",
    "pytest-qt",
}


def _write_valid_inputs(
    root: Path,
    *,
    cameras: str = "schema_version: 1\nsources: []\n",
) -> tuple[Path, Path, Path]:
    app = root / "app.yaml"
    camera_path = root / "cameras.yaml"
    manifest = root / "manifest.yaml"
    app.write_text(
        "schema_version: 1\nruntime_dir: runtime\nlog_level: INFO\ncompute:\n  preference: auto\n  cuda_index: 0\n",
        encoding="utf-8",
    )
    camera_path.write_text(cameras, encoding="utf-8")
    manifest.write_text("schema_version: 1\nmodels: []\n", encoding="utf-8")
    return app, camera_path, manifest


def _cpu_fallback() -> ComputeDeviceInfo:
    return ComputeDeviceInfo(
        requested=ComputePreference.AUTO,
        resolved="cpu",
        cuda_available=False,
        cuda_device_count=0,
        device_name=None,
        torch_version="2.test",
        fallback_reason="CUDA is unavailable; using CPU",
    )


@pytest.fixture
def stable_environment(monkeypatch):
    monkeypatch.setattr(environment.platform, "system", lambda: "Windows")
    monkeypatch.setattr(environment.sys, "version_info", (3, 11, 8))
    monkeypatch.setattr(environment.struct, "calcsize", lambda _format: 8)
    monkeypatch.setattr(environment.metadata, "version", lambda _name: "1.0")
    monkeypatch.setattr(environment, "probe_compute_device", lambda _config: _cpu_fallback())


def _checks_by_name(report) -> dict[str, object]:
    return {check.name: check for check in report.checks}


def test_python_311_and_64_bit_are_separate_checks(tmp_path, stable_environment) -> None:
    app, cameras, manifest = _write_valid_inputs(tmp_path)

    report = check_environment(app, cameras, manifest)
    checks = _checks_by_name(report)

    assert checks["python"].status is CheckStatus.PASS
    assert checks["architecture"].status is CheckStatus.PASS


def test_wrong_python_does_not_hide_valid_architecture(tmp_path, stable_environment, monkeypatch) -> None:
    app, cameras, manifest = _write_valid_inputs(tmp_path)
    monkeypatch.setattr(environment.sys, "version_info", (3, 10, 12))

    checks = _checks_by_name(check_environment(app, cameras, manifest))

    assert checks["python"].status is CheckStatus.FAIL
    assert checks["architecture"].status is CheckStatus.PASS


def test_non_64_bit_does_not_hide_valid_python(tmp_path, stable_environment, monkeypatch) -> None:
    app, cameras, manifest = _write_valid_inputs(tmp_path)
    monkeypatch.setattr(environment.struct, "calcsize", lambda _format: 4)

    checks = _checks_by_name(check_environment(app, cameras, manifest))

    assert checks["python"].status is CheckStatus.PASS
    assert checks["architecture"].status is CheckStatus.FAIL


def test_required_distribution_versions_are_reported(tmp_path, stable_environment) -> None:
    app, cameras, manifest = _write_valid_inputs(tmp_path)

    check = _checks_by_name(check_environment(app, cameras, manifest))["dependencies"]

    assert check.status is CheckStatus.PASS
    assert set(check.details["versions"]) == REQUIRED_DISTRIBUTIONS
    assert set(check.details["versions"].values()) == {"1.0"}


def test_valid_configuration_empty_manifest_and_cpu_fallback_are_accepted(
    tmp_path,
    stable_environment,
) -> None:
    app, cameras, manifest = _write_valid_inputs(tmp_path)

    checks = _checks_by_name(check_environment(app, cameras, manifest))

    assert checks["configuration"].status is CheckStatus.PASS
    assert checks["model_manifest"].status is CheckStatus.PASS
    assert checks["compute"].status is CheckStatus.WARNING
    assert checks["logging"].status is CheckStatus.PASS


def test_malformed_configuration_becomes_named_failed_check(tmp_path, stable_environment) -> None:
    app, cameras, manifest = _write_valid_inputs(tmp_path)
    cameras.write_text("schema_version: 2\nsources: []\n", encoding="utf-8")

    report = check_environment(app, cameras, manifest)
    checks = _checks_by_name(report)

    assert report.ok is False
    assert checks["configuration"].status is CheckStatus.FAIL


def test_missing_distribution_becomes_named_failed_check(
    tmp_path,
    stable_environment,
    monkeypatch,
) -> None:
    app, cameras, manifest = _write_valid_inputs(tmp_path)

    def installed_version(name: str) -> str:
        if name == "onnxruntime":
            raise metadata.PackageNotFoundError(name)
        return "1.0"

    monkeypatch.setattr(environment.metadata, "version", installed_version)

    check = _checks_by_name(check_environment(app, cameras, manifest))["dependencies"]

    assert check.status is CheckStatus.FAIL
    assert check.details["missing"] == ["onnxruntime"]


def test_logging_setup_failure_becomes_named_failed_check(
    tmp_path,
    stable_environment,
    monkeypatch,
) -> None:
    app, cameras, manifest = _write_valid_inputs(tmp_path)

    def fail_logging(*_args, **_kwargs):
        raise LoggingSetupError("cannot open safe log path")

    monkeypatch.setattr(environment, "configure_logging", fail_logging)

    check = _checks_by_name(check_environment(app, cameras, manifest))["logging"]

    assert check.status is CheckStatus.FAIL
    assert "cannot open safe log path" in check.summary


def test_pytorch_import_failure_is_a_failed_compute_check(
    tmp_path,
    stable_environment,
    monkeypatch,
) -> None:
    app, cameras, manifest = _write_valid_inputs(tmp_path)
    broken = ComputeDeviceInfo(
        requested=ComputePreference.AUTO,
        resolved="cpu",
        cuda_available=False,
        cuda_device_count=0,
        device_name=None,
        torch_version="2.test",
        fallback_reason="PyTorch import failed: c10.dll failed",
    )
    monkeypatch.setattr(environment, "probe_compute_device", lambda _config: broken)

    check = _checks_by_name(check_environment(app, cameras, manifest))["compute"]

    assert check.status is CheckStatus.FAIL


def test_report_serialization_does_not_contain_resolved_rtsp_secret(
    tmp_path,
    stable_environment,
) -> None:
    cameras = """\
schema_version: 1
sources:
  - source_id: cctv-1
    name: CCTV 1
    type: rtsp
    enabled: true
    rtsp_env_key: CAMERA_1_URL
"""
    app, camera_path, manifest = _write_valid_inputs(tmp_path, cameras=cameras)
    secret_url = "rtsp://admin:camera-secret@example.test/live"
    env_file = tmp_path / ".env"
    env_file.write_text(f"CAMERA_1_URL={secret_url}\n", encoding="utf-8")

    serialized = json.dumps(
        check_environment(app, camera_path, manifest, env_file).to_dict()
    )

    assert secret_url not in serialized
    assert "admin:camera-secret" not in serialized
    assert "CAMERA_1_URL" in serialized
