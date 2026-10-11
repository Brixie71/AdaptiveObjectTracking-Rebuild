# Gate 1 Environment Evidence

**Date:** 2026-10-11

**Machine label:** `LAPTOP-PAJLTCUH`

**Scope:** Reproducible environment, validated configuration, model manifest, compute-device selection, and structured logging. Camera and model-inference hardware checks begin in later gates.

## Runtime

| Item | Observed value |
| --- | --- |
| Platform | Windows |
| Interpreter | CPython 3.11.8, 64-bit |
| Requested compute | `auto` |
| Resolved compute | `cpu` |
| CUDA | Unavailable; controlled warning and CPU fallback |
| PyTorch runtime | `2.14.1+cpu` |
| Declared models | 0; empty version-1 manifest validated |
| Example sources | 3 disabled source definitions validated |

## Installed distributions

| Distribution | Version |
| --- | --- |
| torch | 2.14.1 |
| torchvision | 0.29.1 |
| ultralytics | 8.4.158 |
| opencv-python | 4.14.0.94 |
| numpy | 2.4.6 |
| Pillow | 12.3.0 |
| PyQt5 | 5.15.11 |
| PyYAML | 6.0.3 |
| python-dotenv | 1.2.4 |
| psutil | 7.2.2 |
| onnx | 1.23.2 |
| onnxruntime | 1.30.0 |
| pytest | 8.4.2 |
| pytest-qt | 4.5.0 |

## Commands and results

```powershell
python -m pip install -e . --no-deps
# exit 0; adaptive-object-tracking 0.1.0 installed editable

python -m pytest tests/unit/config tests/unit/telemetry tests/integration/test_check_environment.py -v
# exit 0; 67 passed

python -m pip check
# exit 0; no broken requirements found

python scripts/check_environment.py
# exit 0; seven PASS checks and one CPU-fallback WARNING

python scripts/check_environment.py --json
# exit 0; sanitized JSON reported ok: true
```

The diagnostic validated Windows, Python 3.11, 64-bit architecture, all required installed distributions, non-sensitive example configuration, the empty model manifest, CPU-safe compute resolution, and a structured log write. The JSON output contained environment-variable key names but no resolved RTSP credentials.

## Gate result

Gate 1 acceptance passed on this machine. This evidence does not claim camera connectivity, CUDA inference, model loading, detection, tracking, ReID, persistence, or UI behavior.
