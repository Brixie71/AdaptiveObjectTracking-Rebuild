# Gate 3 Detection and Class Allowlist Evidence

**Date:** 2026-10-12

**Machine label:** `LAPTOP-PAJLTCUH`

**Runtime:** Windows, CPython 3.11.8, CPU-selected fallback

**Implementation commit before evidence-only documentation:** `55aa3ed0d425e005ca2f65e5f59d30f5c00a2dce`

**Scope:** Strict detector configuration, verified detector-artifact resolution,
immutable detection contracts, Ultralytics YOLOv8 result adaptation,
people/generic-object allowlist enforcement, sanitized Hardware Testing model
probe, and deterministic offline capture-to-detection acceptance.

## Configuration under test

| Item | Value |
| --- | --- |
| Checked-in detector state | Disabled |
| Model ID | Not configured |
| Confidence threshold | `0.25` |
| IoU threshold | `0.70` |
| Maximum detections | `300` |
| Demonstration allowlist | `person`, `backpack`, `bottle` |
| Resolved compute device | CPU |

The checked-in manifest declares zero models and model weights remain excluded
from Git. Runtime loading requires a role-`detector` manifest entry, a contained
regular file, and a matching declared SHA-256.

## Deterministic fixture

| Item | Value |
| --- | --- |
| Path | `tests/fixtures/media/gate2-three-frame.avi` |
| Dimensions | 640 x 480 |
| Frames | 3 |
| Codec | Motion JPEG (`MJPG`) |
| License | CC0-1.0 |
| SHA-256 | `dd607536bed2fb9493bedfea68600e6a10af88c6073a5f94b2c4711e38bd2080` |

Gate 3 reuses this fixture for frame provenance and three-source isolation. An
injected deterministic backend supplies detector rows; therefore this fixture
does not prove real YOLO accuracy.

## Automated results

All commands used the repository Python 3.11 virtual environment. Because that
environment's editable installation points at the main checkout while
verification ran in an isolated worktree, standalone scripts were run with the
worktree `src` directory first on `PYTHONPATH`.

```powershell
python -m pytest -q
# exit 0; 292 passed, 1 skipped

python -m pytest tests/offline/test_gate3_detection_acceptance.py -v
# exit 0; 1 passed

python -m pip check
# exit 0; no broken requirements found

$env:PYTHONPATH = (Resolve-Path 'src').Path
python scripts/check_environment.py --json
# exit 0; sanitized JSON reported ok: true
```

Automated coverage includes strict configuration boundaries, case-insensitive
allowlist resolution, unknown and ambiguous classes, path/link containment,
artifact-role and SHA-256 verification, malformed/non-finite detector rows,
box clipping, maximum-result bounds, compute-device propagation, adapter
cleanup, sanitized failures, nearest-rank timing, capture provenance, and
three-source detector failure isolation.

## Credential scan

A focused scan covered `src`, `scripts`, `config`, `models`, and tests. No live
credential or production connection string was found. Matches were redaction
implementation identifiers or synthetic test-only values using reserved
`.invalid`/`.test` hosts and assertions that the values do not escape.

## Opt-in acceptance not run

**Real detector model load/inference probe:** `NOT RUN`

No owner-approved, manifest-declared detector artifact was present. The
checked-in application configuration intentionally keeps detection disabled.
No model was downloaded implicitly, no digest was fabricated, and no accuracy
claim is made.

**Physical USB/RTSP detector pipeline:** `NOT RUN`

**CUDA detector probe:** `NOT RUN` (the environment diagnostic resolved CPU
because CUDA was unavailable)

## Gate result

The project-owned detection layer and configurable people/generic-object
allowlist pass automated and offline-first acceptance. AOT-007 remains
`PARTIAL` rather than `REBUILD_VERIFIED` until an approved model artifact passes
the opt-in probe; labeled accuracy evaluation and physical-camera inference
also remain unverified.

Target selection, local tracking, person/generic encoders, cross-camera global
identity, persistence, named hardware profiles, retention, and the PyQt UI are
outside Gate 3.
