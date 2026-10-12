"""Device-independent source probing for Settings and diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path

from aot.capture.factory import create_frame_source
from aot.capture.types import FrameSource
from aot.config.loader import load_settings
from aot.config.types import CameraSourceConfig
from aot.hardware_testing.types import HardwareTestStatus, SourceTestResult
from aot.hardware_testing.usb_discovery import discover_usb_devices


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def probe_source(
    source: FrameSource,
    *,
    frame_count: int = 30,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> SourceTestResult:
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
    ):
        raise ValueError("frame_count must be a positive integer")

    started = _utc_now()
    properties = None
    latencies_ms: list[float] = []
    frames_received = 0
    dropped_frames = 0
    previous_sequence: int | None = None
    status = HardwareTestStatus.FAILED
    summary = "Source could not be opened"

    try:
        properties = source.open()
        for _ in range(frame_count):
            read_started = clock_ns()
            try:
                packet = source.read()
            except Exception:
                if frames_received:
                    status = HardwareTestStatus.DEGRADED
                    summary = "Source probe ended before all frames were captured"
                else:
                    status = HardwareTestStatus.FAILED
                    summary = "Source probe captured no frames"
                break
            read_ended = clock_ns()
            latencies_ms.append((read_ended - read_started) / 1_000_000)
            if previous_sequence is None:
                dropped_frames += max(0, packet.sequence)
            else:
                dropped_frames += max(0, packet.sequence - previous_sequence - 1)
            previous_sequence = packet.sequence
            frames_received += 1
        else:
            status = HardwareTestStatus.SUCCESS
            summary = "Source probe completed successfully"
    except Exception:
        status = HardwareTestStatus.FAILED
        summary = "Source could not be opened"
    finally:
        try:
            source.close()
        except Exception:
            pass

    mean_latency = None
    p95_latency = None
    if latencies_ms:
        mean_latency = sum(latencies_ms) / len(latencies_ms)
        ordered = sorted(latencies_ms)
        rank = math.ceil(0.95 * len(ordered))
        p95_latency = ordered[rank - 1]

    return SourceTestResult(
        source_id=source.source_id,
        source_type=source.source_type,
        status=status,
        properties=properties,
        frames_requested=frame_count,
        frames_received=frames_received,
        mean_capture_latency_ms=mean_latency,
        p95_capture_latency_ms=p95_latency,
        dropped_frames=dropped_frames,
        started_at_utc=started,
        ended_at_utc=_utc_now(),
        summary=summary,
    )


def probe_sources(
    sources: Iterable[CameraSourceConfig],
    *,
    source_factory: Callable[[CameraSourceConfig], FrameSource] = create_frame_source,
    frame_count: int = 30,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> tuple[SourceTestResult, ...]:
    results: list[SourceTestResult] = []
    for config in sources:
        try:
            source = source_factory(config)
        except Exception:
            now = _utc_now()
            results.append(
                SourceTestResult(
                    source_id=config.source_id,
                    source_type=config.source_type,
                    status=HardwareTestStatus.UNSUPPORTED,
                    properties=None,
                    frames_requested=frame_count,
                    frames_received=0,
                    mean_capture_latency_ms=None,
                    p95_capture_latency_ms=None,
                    dropped_frames=0,
                    started_at_utc=now,
                    ended_at_utc=now,
                    summary="Source type is unsupported",
                )
            )
            continue
        results.append(probe_source(source, frame_count=frame_count, clock_ns=clock_ns))
    return tuple(results)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe configured camera sources")
    parser.add_argument("--app-config", type=Path, default=Path("config/app.example.yaml"))
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=Path("config/cameras.example.yaml"),
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--discover-usb", action="store_true")
    parser.add_argument("--max-usb-index", type=int, default=10)
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    settings_loader=load_settings,
    source_factory=create_frame_source,
    discover_usb_fn=discover_usb_devices,
    probe_sources_fn=probe_sources,
) -> int:
    args = _parser().parse_args(argv)
    try:
        settings = settings_loader(
            args.app_config,
            args.camera_config,
            args.env_file,
        )
    except Exception:
        return _print_cli_error("Configuration could not be loaded", args.json_output)

    configured = {source.source_id: source for source in settings.sources}
    if args.source_id:
        unknown = next(
            (source_id for source_id in args.source_id if source_id not in configured),
            None,
        )
        if unknown is not None:
            return _print_cli_error(f"Unknown source ID: {unknown}", args.json_output)
        selected = tuple(configured[source_id] for source_id in args.source_id)
    else:
        selected = tuple(source for source in settings.sources if source.enabled)
        if not selected and not args.discover_usb:
            return _print_cli_error("No enabled camera sources", args.json_output)

    try:
        devices = (
            discover_usb_fn(max_index=args.max_usb_index)
            if args.discover_usb
            else ()
        )
        results = probe_sources_fn(
            selected,
            source_factory=source_factory,
            frame_count=args.frames,
        )
    except Exception:
        return _print_cli_error("Camera probe could not be completed", args.json_output)

    ok = all(result.status is HardwareTestStatus.SUCCESS for result in results)
    payload = {
        "ok": ok,
        "sources": [result.to_dict() for result in results],
        "usb_devices": [device.to_dict() for device in devices],
    }
    if args.json_output:
        print(json.dumps(payload, sort_keys=True))
    else:
        for result in results:
            print(f"[{result.status.value.upper()}] {result.source_id}: {result.summary}")
        for device in devices:
            print(
                f"[USB] index {device.device_index}: "
                f"{device.width}x{device.height} at {device.fps:g} FPS"
            )
    return 0 if ok else 1


def _print_cli_error(summary: str, json_output: bool) -> int:
    if json_output:
        print(json.dumps({"ok": False, "error": summary, "sources": [], "usb_devices": []}))
    else:
        print(f"[FAILED] {summary}")
    return 1
