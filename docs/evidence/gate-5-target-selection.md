# Gate 5 Target Selection Evidence

**Date:** 2026-10-12

**Machine label:** `LAPTOP-PAJLTCUH`

**Runtime:** Windows 10/11 x64 family, CPython 3.11.8, PyQt5 5.15.11,
`QT_QPA_PLATFORM=offscreen`, CPU-selected fallback

**Implementation baseline before acceptance documentation:** `8364d2a`

**Final reviewed commit:** Pending whole-branch review

**Scope:** Exact displayed-frame hit testing, deterministic overlap precedence,
validated owned reference crops, one session-only selected-target state,
native selected-target presentation, cross-panel replacement, explicit
disengagement, and deterministic three-source offline acceptance.

## Deterministic fixture

| Item | Value |
| --- | --- |
| Path | `tests/fixtures/media/gate2-three-frame.avi` |
| Dimensions | 640 x 480 |
| Frames | 3 |
| Codec | Motion JPEG (`MJPG`) |
| License | CC0-1.0 |
| SHA-256 | `dd607536bed2fb9493bedfea68600e6a10af88c6073a5f94b2c4711e38bd2080` |

The acceptance test decodes the existing CC0 fixture into three independent
offline sources and uses one injected deterministic detector. No physical
camera, network access, model weight, database, or retained crop is required.

## Automated results

All commands used the repository Python 3.11 virtual environment with the
worktree `src` directory first on `PYTHONPATH`.

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest tests/offline/test_three_source_capture.py tests/offline/test_gate3_detection_acceptance.py tests/offline/test_gate4_three_camera_ui.py tests/offline/test_gate5_target_selection.py -q
# exit 0; 4 passed

python -m pytest -q
# exit 0; 442 passed, 1 skipped

python -m pip check
# exit 0; No broken requirements found.

python scripts/check_environment.py --json
# exit 0; sanitized report returned "ok": true; CPU fallback warning only

git diff --check
# exit 0; no whitespace errors
```

The Gate 5 acceptance launches the real `MainWindow`, `LiveRuntime`, processing
worker, selector, three camera panels, and selected-target panel offscreen. It
clicks Camera A, B, and C through real Qt widgets and proves exact source,
frame-sequence, monotonic timestamp, class, confidence, clipped half-open box,
and pixel-for-pixel crop provenance from each panel's retained displayed
`ProcessedFrame`.

The overlapping Camera A click selects the highest-confidence candidate and,
for equal confidence, the smaller clipped box. Camera B proves edge clipping.
Letterbox-margin and in-frame empty clicks preserve the current target. A valid
click on another camera replaces it. The native disengage button clears both
selector state and crop presentation. A Qt heartbeat continues, then Stop and
window shutdown release processing and detector ownership.

## Hardware and model proof boundary

**Physical camera selection:** `NOT RUN`

**Real-model selection:** `NOT RUN`

**CUDA inference:** `NOT RUN` (the environment diagnostic resolved CPU because
CUDA was unavailable)

No separately authorized physical camera run or verified model artifact was
provided. These results do not establish physical-camera behavior, real-model
accuracy, throughput, or end-to-end performance.

## Scope exclusions

Gate 5 does not implement local tracking, encoders, ReID matching,
cross-camera/global identity, persistence, retained media, retention jobs,
uploads, target naming, save/load, SQLite, or named profiles. The selected crop
and metadata exist only in memory for the current application session.

## Gate result

AOT-010, AOT-011, and AOT-024 are `REBUILD_VERIFIED` for the automated Gate 5
boundary described above. The manuscript-reported 28/30 selection results are
retained only as historical claims; they were not reproduced or converted into
rebuild results.
