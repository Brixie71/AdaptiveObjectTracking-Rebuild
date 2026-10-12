# Gate 2 Camera Layer Evidence

**Date:** 2026-10-12

**Machine label:** `LAPTOP-PAJLTCUH`

**Runtime:** Windows, CPython 3.11.8, CPU-selected fallback

**Scope:** USB, RTSP, and offline capture boundaries; canonical normalization; bounded independent workers; reconnection and shutdown; sanitized Hardware Testing probe; deterministic three-source offline acceptance.

## Deterministic fixture

| Item | Value |
| --- | --- |
| Path | `tests/fixtures/media/gate2-three-frame.avi` |
| Generator | `python scripts/generate_gate2_fixture.py` |
| Dimensions | 640 x 480 |
| Frames | 3 |
| Frame rate | 5 FPS |
| Codec | Motion JPEG (`MJPG`) |
| License | CC0-1.0 |
| SHA-256 | `dd607536bed2fb9493bedfea68600e6a10af88c6073a5f94b2c4711e38bd2080` |

The generator reopened the artifact and verified exactly three decodable frames.

## Automated results

```powershell
python -m pytest -q
# exit 0; 198 passed, 1 skipped

python -m pytest tests/offline/test_three_source_capture.py -q
# exit 0; 1 passed

python -m pip check
# exit 0; no broken requirements found

python scripts/check_environment.py --json
# exit 0; sanitized JSON reported ok: true

python scripts/probe_cameras.py --app-config config/app.example.yaml --camera-config config/cameras.example.yaml --json
# exit 1 as designed; sanitized JSON reported No enabled camera sources
```

The implementation credential scan produced no live credential match outside excluded historical/reference, plan, prerequisite, and test paths. Automated tests cover secret-safe resolution and errors, normalized output, bounded drops, reconnect timing, per-source isolation, repeated shutdown, source probing, and USB discovery through injected backends.

## Hardware acceptance

**USB/RTSP hardware tests:** `NOT RUN`

No authorized physical-device configuration was supplied during this verification. The default suite safely skipped the module. Therefore live USB/RTSP connectivity, negotiated device properties, and physical three-camera concurrency remain `PARTIAL`; this evidence does not claim them.

## Gate result

Automated and offline-first Gate 2 evidence passed. Physical USB/RTSP acceptance remains pending the documented opt-in run on named hardware. Detection, tracking, ReID, identity, persistence, and UI behavior are not part of this evidence.
