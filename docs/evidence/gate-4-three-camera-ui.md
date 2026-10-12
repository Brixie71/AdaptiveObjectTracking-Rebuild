# Gate 4 Native Three-Camera UI Evidence

**Date:** 2026-10-12

**Machine label:** `LAPTOP-PAJLTCUH`

**Runtime:** Windows 10/11 x64 family, CPython 3.11.8, PyQt5 5.15.11,
`QT_QPA_PLATFORM=offscreen`, CPU-selected fallback

**Implementation commit before acceptance documentation:** `8469667`

**Final review fix commit:** `f717abb`

**Scope:** Native three-slot PyQt5 presentation, immutable runtime snapshots,
single-worker detector scheduling, asynchronous Start/Stop, source health,
Settings Hardware Testing, sanitized composition, and deterministic offline
responsiveness/failure-isolation/shutdown acceptance.

## Deterministic fixture

| Item | Value |
| --- | --- |
| Path | `tests/fixtures/media/gate2-three-frame.avi` |
| Dimensions | 640 x 480 |
| Frames | 3 |
| Codec | Motion JPEG (`MJPG`) |
| License | CC0-1.0 |
| SHA-256 | `dd607536bed2fb9493bedfea68600e6a10af88c6073a5f94b2c4711e38bd2080` |

Gate 4 decodes this fixture into three independent offline source streams and
uses an injected deterministic detector. No camera, network access, model
weight, or retained output image is required.

## Automated results

All commands used the repository Python 3.11 virtual environment with the
worktree `src` directory first on `PYTHONPATH`.

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest tests/offline/test_three_source_capture.py tests/offline/test_gate3_detection_acceptance.py tests/offline/test_gate4_three_camera_ui.py -q
# exit 0; 3 passed

python -m pytest -q
# exit 0; 371 passed, 1 skipped

python -m pip check
# exit 0; No broken requirements found.

python scripts/check_environment.py --json
# exit 0; sanitized report returned "ok": true

git diff --check
# exit 0; no whitespace errors
```

The Gate 4 acceptance launched the real `MainWindow` offscreen and drove its
Start and Stop controls against a real `LiveRuntime`. A blocked fake detector
left the GUI heartbeat active; after release, all three panels advanced with
matching source and frame-sequence provenance and an allowed `person` label.
One source then entered `FAILED`, one reached `EOF`, and the third continued to
advance. Visible status remained sanitized, Stop remained usable, detector and
processing ownership closed, and repeated shutdown was safe.

Unit coverage additionally proves owned BGR-to-RGB image memory, clipped
overlays, fixed three-slot mapping, raw-preview rendering when detection is
disabled, asynchronous lifecycle/probe calls, duplicate-job rejection,
stale/late callback rejection, one-to-three source validation, fresh restart
ownership, and sanitized application-boundary failures.

The final whole-branch review found no Critical issues. Its six Important
findings were corrected with regressions covering process-safe blocked-task
teardown, retained and retryable partial-start/failed-stop cleanup, deferred
Stop during startup, one application-owned Settings/probe surface, serialized
process-global detector output isolation, and visible sanitized lifecycle
failures. The focused review-fix suite passed 24 tests before the complete
suite was rerun.

## Hardware and model proof boundary

**Physical three-camera native UI:** `NOT RUN`

**Physical USB/RTSP Hardware Testing actions:** `NOT RUN`

**Real detector model load/inference and native overlay:** `NOT RUN`

**CUDA inference:** `NOT RUN` (the environment diagnostic resolved CPU because
CUDA was unavailable)

No owner-approved physical sources or manifest-declared detector artifact were
provided for this gate. These results must not be interpreted as physical
throughput, real-model accuracy, or end-to-end performance evidence.

## Scope exclusions

Gate 4 does not add target selection, local tracking, dual encoders, ReID,
cross-camera/global identity, database persistence, retained media, retention
jobs, uploads, or named-profile saving. Hardware Testing may report a
successful configuration, but persistence of a named profile remains Gate 10.

## Gate result

The native UI and offline application boundary satisfy the automated Gate 4
acceptance criteria. AOT-002 and AOT-009 remain `PARTIAL` because physical
three-camera and real-model display checks are unrun. AOT-023 is
`REBUILD_VERIFIED` for the tested offscreen concurrency and lifecycle boundary;
physical-load responsiveness remains explicitly unverified.
