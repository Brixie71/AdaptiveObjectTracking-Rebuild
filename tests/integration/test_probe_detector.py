from __future__ import annotations

import json
from pathlib import Path

import pytest

from aot.config.model_manifest import ModelManifest
from aot.config.types import AppSettings, ComputeConfig, ComputePreference, DetectionConfig
from aot.hardware_testing.detector_probe import main
from aot.hardware_testing.types import DetectorTestResult, HardwareTestStatus
from aot.telemetry.device_probe import ComputeDeviceInfo


def _settings(*, enabled: bool = True) -> AppSettings:
    return AppSettings(
        schema_version=1,
        runtime_dir=Path("runtime"),
        log_level="INFO",
        compute=ComputeConfig(preference=ComputePreference.CPU),
        sources=(),
        detection=DetectionConfig(
            enabled=enabled,
            model_id="detector-test" if enabled else None,
            allowed_classes=("person", "backpack"),
        ),
    )


def _compute() -> ComputeDeviceInfo:
    return ComputeDeviceInfo(
        requested=ComputePreference.CPU,
        resolved="cpu",
        cuda_available=False,
        cuda_device_count=0,
        device_name=None,
        torch_version="test",
        fallback_reason=None,
    )


def _result(status: HardwareTestStatus) -> DetectorTestResult:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    return DetectorTestResult(
        model_id="detector-test",
        status=status,
        device="cpu",
        warmup_runs=2,
        measured_runs=10,
        successful_runs=10 if status is HardwareTestStatus.SUCCESS else 1,
        mean_inference_ms=5.0,
        p95_inference_ms=8.0,
        detection_count=0,
        started_at_utc=now,
        ended_at_utc=now,
        summary="Detector probe completed successfully",
    )


def _run(
    *,
    status: HardwareTestStatus = HardwareTestStatus.SUCCESS,
    enabled: bool = True,
    detector_factory=lambda *args, **kwargs: object(),
) -> int:
    return main(
        ["--json"],
        settings_loader=lambda *_args: _settings(enabled=enabled),
        manifest_loader=lambda _path: ModelManifest(schema_version=1, models=()),
        compute_probe=lambda _config: _compute(),
        detector_factory=detector_factory,
        detector_probe=lambda detector, packet, **kwargs: _result(status),
    )


def test_detector_probe_cli_prints_successful_sanitized_json(capsys) -> None:
    observed = {}

    def detector_probe(detector, packet, **kwargs):
        observed["shape"] = packet.frame.shape
        observed["dtype"] = str(packet.frame.dtype)
        observed.update(kwargs)
        return _result(HardwareTestStatus.SUCCESS)

    exit_code = main(
        ["--json", "--warmup-runs", "2", "--measured-runs", "10"],
        settings_loader=lambda *_args: _settings(),
        manifest_loader=lambda _path: ModelManifest(schema_version=1, models=()),
        compute_probe=lambda _config: _compute(),
        detector_factory=lambda *args, **kwargs: object(),
        detector_probe=detector_probe,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["detector"]["status"] == "success"
    assert observed == {
        "shape": (480, 640, 3),
        "dtype": "uint8",
        "device": "cpu",
        "warmup_runs": 2,
        "measured_runs": 10,
    }


@pytest.mark.parametrize("status", [HardwareTestStatus.DEGRADED, HardwareTestStatus.FAILED])
def test_detector_probe_cli_returns_one_for_non_success(status, capsys) -> None:
    exit_code = _run(status=status)

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["detector"]["status"] == status.value


def test_detector_probe_cli_reports_disabled_configuration_as_unsupported(capsys) -> None:
    exit_code = _run(enabled=False)

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["detector"]["status"] == "unsupported"


def test_detector_probe_cli_sanitizes_factory_failures(capsys) -> None:
    secret = "rtsp://admin:model-pass@example.invalid/model?token=private"

    def noisy_factory(*args, **kwargs):
        print(secret)
        raise RuntimeError(secret)

    exit_code = _run(
        detector_factory=noisy_factory
    )

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert exit_code == 1
    assert payload["detector"]["status"] == "failed"
    assert secret not in output
    assert "Traceback" not in output


def test_detector_probe_cli_contains_noisy_success_output(capsys) -> None:
    secret = "successful-private-output"

    def noisy_factory(*args, **kwargs):
        print(secret)
        return object()

    def noisy_probe(detector, packet, **kwargs):
        print(secret)
        return _result(HardwareTestStatus.SUCCESS)

    exit_code = main(
        ["--json"],
        settings_loader=lambda *_args: _settings(),
        manifest_loader=lambda _path: ModelManifest(schema_version=1, models=()),
        compute_probe=lambda _config: _compute(),
        detector_factory=noisy_factory,
        detector_probe=noisy_probe,
    )

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert exit_code == 0
    assert payload["ok"] is True
    assert secret not in output


def test_detector_probe_cli_rejects_invalid_numeric_arguments(capsys) -> None:
    exit_code = main(
        ["--json", "--warmup-runs", "-1", "--measured-runs", "0"],
        settings_loader=lambda *_args: _settings(),
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["detector"]["status"] == "failed"


def test_detector_probe_cli_human_output_is_concise(capsys) -> None:
    exit_code = main(
        [],
        settings_loader=lambda *_args: _settings(enabled=False),
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert output.startswith("[UNSUPPORTED]")
    assert "Traceback" not in output
