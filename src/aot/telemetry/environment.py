"""Gate 1 environment diagnostics and command-line reporting."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import struct
import sys
from dataclasses import dataclass, field
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

from aot.config.types import (
    AppSettings,
    ComputeConfig,
    SourceType,
)

from .device_probe import ComputeDeviceInfo, probe_compute_device
from .logging_setup import LoggingSetupError, configure_logging, redact_text


REQUIRED_DISTRIBUTIONS = (
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
)


class CheckStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass(frozen=True)
class EnvironmentCheck:
    name: str
    status: CheckStatus
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentReport:
    checks: tuple[EnvironmentCheck, ...]
    _secrets: tuple[str, ...] = field(default=(), repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return all(check.status is not CheckStatus.FAIL for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        raw = {
            "ok": self.ok,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "summary": check.summary,
                    "details": check.details,
                }
                for check in self.checks
            ],
        }
        serialized = json.dumps(raw, ensure_ascii=False)
        return json.loads(redact_text(serialized, self._secrets))


def _runtime_checks() -> list[EnvironmentCheck]:
    system = platform.system()
    python_version = tuple(sys.version_info[:3])
    pointer_bits = struct.calcsize("P") * 8
    return [
        EnvironmentCheck(
            name="platform",
            status=CheckStatus.PASS if system == "Windows" else CheckStatus.FAIL,
            summary=f"{system} detected; Windows 10/11 x64 is required",
            details={"system": system},
        ),
        EnvironmentCheck(
            name="python",
            status=(
                CheckStatus.PASS
                if python_version[:2] == (3, 11)
                else CheckStatus.FAIL
            ),
            summary=f"Python {'.'.join(str(part) for part in python_version)}",
            details={"version": ".".join(str(part) for part in python_version)},
        ),
        EnvironmentCheck(
            name="architecture",
            status=CheckStatus.PASS if pointer_bits == 64 else CheckStatus.FAIL,
            summary=f"{pointer_bits}-bit interpreter",
            details={"pointer_bits": pointer_bits},
        ),
    ]


def _dependency_check() -> EnvironmentCheck:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for distribution in REQUIRED_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            missing.append(distribution)
    if missing:
        return EnvironmentCheck(
            name="dependencies",
            status=CheckStatus.FAIL,
            summary=f"Missing required distributions: {', '.join(missing)}",
            details={"versions": versions, "missing": missing},
        )
    return EnvironmentCheck(
        name="dependencies",
        status=CheckStatus.PASS,
        summary=f"{len(versions)} required distributions found",
        details={"versions": versions, "missing": []},
    )


def _configuration_check(
    app_path: Path,
    cameras_path: Path,
    env_path: Path | None,
) -> tuple[EnvironmentCheck, AppSettings | None, tuple[str, ...]]:
    try:
        from aot.config.loader import ConfigError, load_settings, resolve_rtsp_url

        settings = load_settings(app_path, cameras_path, env_path)
        secrets: list[str] = []
        rtsp_keys: list[str] = []
        for source in settings.sources:
            if source.source_type is SourceType.RTSP and source.rtsp_env_key:
                rtsp_keys.append(source.rtsp_env_key)
                if source.enabled:
                    secrets.append(resolve_rtsp_url(source))
    except ModuleNotFoundError as error:
        return (
            EnvironmentCheck(
                name="configuration",
                status=CheckStatus.FAIL,
                summary=(
                    "Configuration checks are unavailable because "
                    f"{error.name!r} cannot be imported"
                ),
                details={"app_path": str(app_path), "cameras_path": str(cameras_path)},
            ),
            None,
            (),
        )
    except ConfigError as error:
        return (
            EnvironmentCheck(
                name="configuration",
                status=CheckStatus.FAIL,
                summary=str(error),
                details={"app_path": str(app_path), "cameras_path": str(cameras_path)},
            ),
            None,
            (),
        )
    return (
        EnvironmentCheck(
            name="configuration",
            status=CheckStatus.PASS,
            summary=f"Loaded {len(settings.sources)} source configuration(s)",
            details={
                "app_path": str(app_path),
                "cameras_path": str(cameras_path),
                "source_ids": [source.source_id for source in settings.sources],
                "rtsp_environment_keys": rtsp_keys,
            },
        ),
        settings,
        tuple(secrets),
    )


def _manifest_check(manifest_path: Path) -> EnvironmentCheck:
    try:
        from aot.config.model_manifest import ModelManifestError, load_model_manifest

        manifest = load_model_manifest(manifest_path)
    except ModuleNotFoundError as error:
        return EnvironmentCheck(
            name="model_manifest",
            status=CheckStatus.FAIL,
            summary=(
                "Model manifest checks are unavailable because "
                f"{error.name!r} cannot be imported"
            ),
            details={"path": str(manifest_path)},
        )
    except ModelManifestError as error:
        return EnvironmentCheck(
            name="model_manifest",
            status=CheckStatus.FAIL,
            summary=str(error),
            details={"path": str(manifest_path)},
        )
    return EnvironmentCheck(
        name="model_manifest",
        status=CheckStatus.PASS,
        summary=f"Validated {len(manifest.models)} model declaration(s)",
        details={
            "path": str(manifest_path),
            "model_ids": [model.model_id for model in manifest.models],
        },
    )


def _compute_check(config: ComputeConfig) -> EnvironmentCheck:
    info: ComputeDeviceInfo = probe_compute_device(config)
    status = CheckStatus.PASS
    if info.fallback_reason:
        status = (
            CheckStatus.FAIL
            if info.fallback_reason.startswith("PyTorch import failed")
            else CheckStatus.WARNING
        )
    return EnvironmentCheck(
        name="compute",
        status=status,
        summary=info.fallback_reason or f"Using {info.resolved}",
        details={
            "requested": info.requested.value,
            "resolved": info.resolved,
            "cuda_available": info.cuda_available,
            "cuda_device_count": info.cuda_device_count,
            "device_name": info.device_name,
            "torch_version": info.torch_version,
            "fallback_reason": info.fallback_reason,
        },
    )


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _logging_check(settings: AppSettings | None, secrets: tuple[str, ...]) -> EnvironmentCheck:
    if settings is None:
        return EnvironmentCheck(
            name="logging",
            status=CheckStatus.FAIL,
            summary="Logging was not checked because configuration is invalid",
        )
    log_dir = settings.runtime_dir / "logs"
    try:
        logger = configure_logging(settings.log_level, log_dir, secrets)
        logger.info("Gate 1 logging check")
        _close_logger(logger)
    except LoggingSetupError as error:
        return EnvironmentCheck(
            name="logging",
            status=CheckStatus.FAIL,
            summary=redact_text(str(error), secrets),
            details={"log_dir": str(log_dir)},
        )
    return EnvironmentCheck(
        name="logging",
        status=CheckStatus.PASS,
        summary="Structured log write succeeded",
        details={"log_dir": str(log_dir)},
    )


def check_environment(
    app_path: Path,
    cameras_path: Path,
    manifest_path: Path,
    env_path: Path | None = None,
) -> EnvironmentReport:
    """Run all non-hardware Gate 1 environment checks."""

    app_path = Path(app_path).resolve()
    cameras_path = Path(cameras_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    env_path = Path(env_path).resolve() if env_path is not None else None

    checks = _runtime_checks()
    checks.append(_dependency_check())
    config_check, settings, secrets = _configuration_check(
        app_path,
        cameras_path,
        env_path,
    )
    checks.append(config_check)
    checks.append(_manifest_check(manifest_path))
    checks.append(_compute_check(settings.compute if settings else ComputeConfig()))
    checks.append(_logging_check(settings, secrets))
    return EnvironmentReport(checks=tuple(checks), _secrets=secrets)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the Gate 1 development environment")
    parser.add_argument("--app-config", type=Path, default=Path("config/app.example.yaml"))
    parser.add_argument("--camera-config", type=Path, default=Path("config/cameras.example.yaml"))
    parser.add_argument("--model-manifest", type=Path, default=Path("models/manifest.yaml"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the environment diagnostic and return a process-compatible status."""

    arguments = _parser().parse_args(argv)
    report = check_environment(
        arguments.app_config,
        arguments.camera_config,
        arguments.model_manifest,
        arguments.env_file,
    )
    if arguments.as_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True, ensure_ascii=False))
    else:
        for check in report.checks:
            print(f"[{check.status.value.upper()}] {check.name}: {check.summary}")
    return 0 if report.ok else 1
