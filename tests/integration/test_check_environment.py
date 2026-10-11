from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from aot.config import ComputePreference
from aot.telemetry import environment
from aot.telemetry.device_probe import ComputeDeviceInfo
from aot.telemetry.environment import main


def _write_cli_inputs(root: Path, *, valid: bool = True) -> tuple[Path, Path, Path]:
    app = root / "app.yaml"
    cameras = root / "cameras.yaml"
    manifest = root / "manifest.yaml"
    app.write_text(
        "schema_version: 1\nruntime_dir: runtime\nlog_level: INFO\ncompute:\n  preference: cpu\n  cuda_index: 0\n",
        encoding="utf-8",
    )
    cameras.write_text(
        f"schema_version: {1 if valid else 2}\nsources: []\n",
        encoding="utf-8",
    )
    manifest.write_text("schema_version: 1\nmodels: []\n", encoding="utf-8")
    return app, cameras, manifest


@pytest.fixture
def passing_machine(monkeypatch):
    monkeypatch.setattr(environment.platform, "system", lambda: "Windows")
    monkeypatch.setattr(environment.sys, "version_info", (3, 11, 8))
    monkeypatch.setattr(environment.struct, "calcsize", lambda _format: 8)
    monkeypatch.setattr(environment.metadata, "version", lambda _name: "1.0")
    monkeypatch.setattr(
        environment,
        "probe_compute_device",
        lambda _config: ComputeDeviceInfo(
            requested=ComputePreference.CPU,
            resolved="cpu",
            cuda_available=False,
            cuda_device_count=0,
            device_name=None,
            torch_version="2.test",
            fallback_reason=None,
        ),
    )


def _arguments(app: Path, cameras: Path, manifest: Path, *extra: str) -> list[str]:
    return [
        "--app-config",
        str(app),
        "--camera-config",
        str(cameras),
        "--model-manifest",
        str(manifest),
        *extra,
    ]


def test_cli_returns_zero_and_prints_human_report_for_passing_checks(
    tmp_path,
    passing_machine,
    capsys,
) -> None:
    app, cameras, manifest = _write_cli_inputs(tmp_path)

    result = main(_arguments(app, cameras, manifest))

    output = capsys.readouterr().out
    assert result == 0
    assert "[PASS] python" in output
    assert "[PASS] logging" in output


def test_cli_returns_one_when_a_check_fails(tmp_path, passing_machine, capsys) -> None:
    app, cameras, manifest = _write_cli_inputs(tmp_path, valid=False)

    result = main(_arguments(app, cameras, manifest))

    output = capsys.readouterr().out
    assert result == 1
    assert "[FAIL] configuration" in output


def test_cli_json_is_parseable_and_sanitized(tmp_path, passing_machine, capsys) -> None:
    app, cameras, manifest = _write_cli_inputs(tmp_path)

    result = main(_arguments(app, cameras, manifest, "--json"))

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert result == 0
    assert payload["ok"] is True
    assert isinstance(payload["checks"], list)
    assert "rtsp://" not in output


def test_cli_human_error_does_not_echo_malformed_yaml_credentials(
    tmp_path,
    passing_machine,
    capsys,
) -> None:
    app, cameras, manifest = _write_cli_inputs(tmp_path)
    secret = "admin:" + "camera-pass"
    token = "review-" + "token"
    cameras.write_text(
        f"schema_version: 1\nsources: [rtsp://{secret}@camera.test/live?token={token}\n",
        encoding="utf-8",
    )

    result = main(_arguments(app, cameras, manifest))

    output = capsys.readouterr().out
    assert result == 1
    assert "[FAIL] configuration" in output
    assert secret not in output
    assert token not in output


@pytest.mark.parametrize("blocked_module", ["yaml", "dotenv"])
def test_cli_bootstraps_when_configuration_dependency_cannot_import(
    tmp_path,
    blocked_module: str,
) -> None:
    app, cameras, manifest = _write_cli_inputs(tmp_path)
    source_root = Path(__file__).resolve().parents[2] / "src"
    code = f"""
import builtins
import json
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] == {blocked_module!r}:
        raise ModuleNotFoundError("blocked test dependency", name={blocked_module!r})
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
from aot.telemetry.environment import main
raise SystemExit(main([
    '--json',
    '--app-config', {str(app)!r},
    '--camera-config', {str(cameras)!r},
    '--model-manifest', {str(manifest)!r},
]))
"""
    process_environment = os.environ.copy()
    process_environment["PYTHONPATH"] = str(source_root)

    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=process_environment,
        check=False,
    )

    assert completed.returncode == 1
    assert "Traceback" not in completed.stderr
    payload = json.loads(completed.stdout)
    names = [check["name"] for check in payload["checks"]]
    assert names.index("dependencies") < names.index("configuration")
    configuration = next(check for check in payload["checks"] if check["name"] == "configuration")
    assert configuration["status"] == "fail"
    assert "unavailable" in configuration["summary"].lower()
