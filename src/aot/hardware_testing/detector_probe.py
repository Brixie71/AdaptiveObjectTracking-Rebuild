"""Repeatable, sanitized detector compatibility and timing probe."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from aot.capture.types import FramePacket
from aot.config.loader import load_settings
from aot.config.model_manifest import load_model_manifest
from aot.detection.factory import create_detector
from aot.detection.types import Detector
from aot.telemetry.device_probe import probe_compute_device

from .types import DetectorTestResult, HardwareTestStatus


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _validate_counts(warmup_runs: int, measured_runs: int) -> None:
    if (
        isinstance(warmup_runs, bool)
        or not isinstance(warmup_runs, int)
        or warmup_runs < 0
    ):
        raise ValueError("warmup_runs must be a nonnegative integer")
    if (
        isinstance(measured_runs, bool)
        or not isinstance(measured_runs, int)
        or measured_runs <= 0
    ):
        raise ValueError("measured_runs must be a positive integer")


def probe_detector(
    detector: Detector,
    packet: FramePacket,
    *,
    device: str,
    warmup_runs: int = 2,
    measured_runs: int = 10,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> DetectorTestResult:
    """Benchmark one constructed detector without exporting visual data."""

    _validate_counts(warmup_runs, measured_runs)
    started = _utc_now()
    successful_runs = 0
    detection_count = 0
    latencies_ms: list[float] = []
    status = HardwareTestStatus.FAILED
    summary = "Detector probe failed before completing inference"
    cleanup_failed = False

    try:
        try:
            for _ in range(warmup_runs):
                detector.detect(packet)
        except Exception:
            summary = "Detector warm-up failed"
        else:
            for _ in range(measured_runs):
                run_started = clock_ns()
                try:
                    batch = detector.detect(packet)
                except Exception:
                    if successful_runs:
                        status = HardwareTestStatus.DEGRADED
                        summary = "Detector probe ended before all measured runs completed"
                    else:
                        status = HardwareTestStatus.FAILED
                        summary = "Detector inference failed"
                    break
                run_ended = clock_ns()
                latencies_ms.append((run_ended - run_started) / 1_000_000)
                successful_runs += 1
                detection_count += len(batch.detections)
            else:
                status = HardwareTestStatus.SUCCESS
                summary = "Detector probe completed successfully"
    finally:
        try:
            detector.close()
        except Exception:
            cleanup_failed = True

    if cleanup_failed:
        status = HardwareTestStatus.FAILED
        summary = "Detector cleanup failed"

    mean_latency = None
    p95_latency = None
    if latencies_ms:
        mean_latency = sum(latencies_ms) / len(latencies_ms)
        ordered = sorted(latencies_ms)
        rank = math.ceil(0.95 * len(ordered))
        p95_latency = ordered[rank - 1]

    return DetectorTestResult(
        model_id=detector.model_id,
        status=status,
        device=device,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
        successful_runs=successful_runs,
        mean_inference_ms=mean_latency,
        p95_inference_ms=p95_latency,
        detection_count=detection_count,
        started_at_utc=started,
        ended_at_utc=_utc_now(),
        summary=summary,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe a configured detector model")
    parser.add_argument("--app-config", type=Path, default=Path("config/app.example.yaml"))
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=Path("models/manifest.yaml"),
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--measured-runs", type=int, default=10)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def _failure_payload(
    status: HardwareTestStatus,
    summary: str,
    *,
    model_id: str = "unconfigured",
    device: str = "unresolved",
) -> dict[str, object]:
    return {
        "ok": False,
        "detector": {
            "model_id": model_id,
            "status": status.value,
            "device": device,
            "summary": summary,
        },
    }


def _print_payload(payload: dict[str, object], json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, sort_keys=True))
        return
    detector = payload["detector"]
    assert isinstance(detector, dict)
    print(
        f"[{str(detector['status']).upper()}] "
        f"{detector['model_id']}: {detector['summary']}"
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    settings_loader=load_settings,
    manifest_loader=load_model_manifest,
    compute_probe=probe_compute_device,
    detector_factory=create_detector,
    detector_probe=probe_detector,
) -> int:
    """Run the detector compatibility probe and return a CLI exit code."""

    args = _parser().parse_args(argv)
    try:
        _validate_counts(args.warmup_runs, args.measured_runs)
    except ValueError:
        payload = _failure_payload(HardwareTestStatus.FAILED, "Invalid detector probe counts")
        _print_payload(payload, args.json_output)
        return 1

    try:
        settings = settings_loader(
            args.app_config,
            Path("config/cameras.example.yaml"),
            None,
        )
    except Exception:
        payload = _failure_payload(HardwareTestStatus.FAILED, "Configuration could not be loaded")
        _print_payload(payload, args.json_output)
        return 1

    model_id = settings.detection.model_id or "unconfigured"
    if not settings.detection.enabled:
        payload = _failure_payload(
            HardwareTestStatus.UNSUPPORTED,
            "Detection is disabled",
            model_id=model_id,
        )
        _print_payload(payload, args.json_output)
        return 1

    try:
        manifest = manifest_loader(args.model_manifest)
        compute = compute_probe(settings.compute)
        detector = detector_factory(
            settings.detection,
            manifest,
            model_store=args.models_dir,
            compute=compute,
        )
        packet = FramePacket(
            source_id="hardware-test",
            sequence=0,
            captured_at_monotonic_ns=time.monotonic_ns(),
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
        )
        result = detector_probe(
            detector,
            packet,
            device=compute.resolved,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
        )
    except Exception:
        payload = _failure_payload(
            HardwareTestStatus.FAILED,
            "Detector probe could not be started",
            model_id=model_id,
        )
        _print_payload(payload, args.json_output)
        return 1

    ok = result.status is HardwareTestStatus.SUCCESS
    payload = {"ok": ok, "detector": result.to_dict()}
    _print_payload(payload, args.json_output)
    return 0 if ok else 1
