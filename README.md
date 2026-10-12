# Adaptive Object Tracking Rebuild

This folder is the active rebuild workspace for the thesis system.

`ThesisV2` remains outside this folder and is treated as read-only historical/source-of-truth material.

Start with `REBUILD_PREREQUISITES.md` before writing implementation code.

## Gate 1 development environment

The initial acceptance platform is Windows 10 or 11 x64 with Python 3.11 x64. Use a project virtual environment rather than the system Python.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Select and install the correct `torch` and `torchvision` pair for the actual machine using the [official PyTorch installer](https://pytorch.org/get-started/locally/). Do not guess a CUDA wheel. Install that pair before the remaining project requirements, then install the package in editable mode:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The committed examples contain no credentials. Copy `.env.example` to `.env` only when local RTSP sources need values; camera configuration stores environment-variable names, not URLs or passwords.

## Gate 1 verification

Run the focused automated suite and dependency consistency check:

```powershell
python -m pytest tests/unit/config tests/unit/telemetry tests/integration/test_check_environment.py -v
python -m pip check
```

Then run the machine diagnostic from the repository root:

```powershell
python scripts/check_environment.py
python scripts/check_environment.py --json
```

The diagnostic validates Windows/Python architecture, installed distributions, configuration, the model manifest, compute capability, and structured-log writability. A machine without usable CUDA may report a CPU-fallback warning, but startup must not crash. A missing or unloadable PyTorch installation is a failure because neither CPU nor GPU inference would be usable.

Gate 1 is accepted only when the test suite and `pip check` pass and both diagnostic commands exit successfully on the intended machine. Sanitized JSON may then be recorded as Gate 1 evidence.

## Gate 2 camera layer

Gate 2 adds project-owned USB, RTSP, and finite offline-video adapters; canonical BGR frame normalization; bounded per-source buffers; independent capture workers with capped reconnection; and sanitized Hardware Testing services. Detection, tracking, ReID, identity, persistence, and the desktop UI remain outside this gate.

Run the automated suite and deterministic three-source offline acceptance from the repository root:

```powershell
python -m pytest -q
python -m pytest tests/offline/test_three_source_capture.py -v
python -m pip check
```

The committed three-frame MJPG fixture is generated entirely from NumPy arrays and can be reproduced with:

```powershell
python scripts/generate_gate2_fixture.py
```

Probe configured sources through the same service intended for the Settings > Hardware Testing module:

```powershell
python scripts/probe_cameras.py --app-config config/app.example.yaml --camera-config config/cameras.example.yaml --json
python scripts/probe_cameras.py --camera-config config/cameras.local.yaml --source-id usb-1 --frames 30
```

Committed examples are disabled, so the first probe command intentionally exits `1` with a sanitized, actionable message. RTSP configuration contains only an environment-variable key; the resolved URL is used immediately before opening and is never included in results or logs.

Physical-device checks are opt-in and require an ignored local camera configuration:

```powershell
$env:AOT_RUN_HARDWARE_TESTS = "1"
$env:AOT_HARDWARE_CAMERA_CONFIG = "config/cameras.local.yaml"
$env:AOT_HARDWARE_SOURCE_IDS = "usb-1,cctv-1,cctv-2"
python -m pytest tests/hardware/test_camera_sources.py -v
```

Optional `AOT_HARDWARE_FRAME_COUNT` and `AOT_HARDWARE_TIMEOUT_SECONDS` values control the live sample size and concurrent wait. Do not put URLs, usernames, passwords, or query tokens in command arguments, source IDs, or committed YAML. Gate 2 does not claim live USB/RTSP acceptance until these checks pass on named hardware.

## Gate 3 detection and class allowlist

Gate 3 adds immutable project-owned detection records, verified model-artifact
resolution, a YOLOv8 adapter, and class-name allowlist enforcement. The
allowlist supports both people and generic objects and is applied inside the
detector before any result can reach tracking, ReID, persistence, or the UI.

The checked-in application example is intentionally safe for a clean checkout:

```yaml
detection:
  enabled: false
  model_id: null
  confidence_threshold: 0.25
  iou_threshold: 0.70
  max_detections: 300
  allowed_classes:
    - person
    - backpack
    - bottle
```

Class names are matched case-insensitively against the selected model and must
be unique. Enabling detection requires a nonblank `model_id`, a non-empty
allowlist, and a matching `detector` declaration in `models/manifest.yaml`.
Before Ultralytics receives a path, the application verifies that the artifact
is a regular file contained under `models/` and that its bytes match the
manifest's SHA-256. The compute device is the CPU or CUDA choice resolved by
the Gate 1 diagnostic; the detector does not independently change devices.

Run automated and offline acceptance from the repository root:

```powershell
python -m pytest -q
python -m pytest tests/offline/test_gate3_detection_acceptance.py -v
python -m pip check
python scripts/check_environment.py --json
```

After deliberately provisioning and declaring an approved detector artifact as
described in `models/README.md`, enable its model ID in a local application
configuration and run the reusable Hardware Testing probe:

```powershell
python scripts/probe_detector.py --app-config config/app.local.yaml --model-manifest models/manifest.yaml --models-dir models --json
```

The probe uses a synthetic 640x480 BGR frame and reports aggregate load and
inference compatibility. A successful empty detection result is valid for that
check and does not establish accuracy. The default suite uses injected
deterministic backends and requires no model weights, camera, network, GUI, or
CUDA device. Gate 3 does not implement target selection, local tracking,
encoders, cross-camera identity, persistence, saved profiles, retention, or the
PyQt UI.

## Gate 4 native three-camera UI

Gate 4 adds a native PyQt5 desktop window with three stable camera slots,
asynchronous Start and Stop controls, source-local health states, detection
overlays, and a Settings > Hardware Testing section. One to three enabled
sources are supported in configured order; unused slots remain visibly
unconfigured, and more than three enabled sources fail before capture starts.

Launch the application from the repository root:

```powershell
python scripts/run_app.py
```

Use `--app-config`, `--camera-config`, `--env-file`, `--model-manifest`, and
`--models-dir` to select ignored local configuration where needed. The
checked-in application example keeps detection disabled, which provides raw
preview without loading a model. Enabling detection requires a verified
role-`detector` manifest entry and artifact as described in the Gate 3 section.

Hardware Testing runs source probes, bounded USB discovery, and the detector
probe outside the GUI thread. Its tables show project-owned sanitized fields;
they never display resolved RTSP URLs or raw exception details. Gate 4 does not
save successful configurations. Named profile saving remains Gate 10.

Run the deterministic native UI acceptance without cameras or model weights:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest tests/offline/test_three_source_capture.py tests/offline/test_gate3_detection_acceptance.py tests/offline/test_gate4_three_camera_ui.py -v
python -m pytest -q
```

The automated acceptance covers three source-local displays, blocked fake
inference while the Qt heartbeat remains responsive, detection provenance,
FAILED/EOF isolation, Stop usability, and bounded repeated shutdown. It does
not establish physical three-camera operation, real-model accuracy, or
end-to-end performance. Target selection, local tracking, dual encoders, ReID,
cross-camera identity, persistence, retention, and saved profiles remain later
gates.

## Gate 5 target selection

Gate 5 lets the user click a detection box in any live camera panel to create
one application-level selected target. The target holds an owned in-memory BGR
crop plus its source ID, exact displayed-frame sequence and timestamp, class,
confidence, clipped half-open pixel box, target ID, and UTC selection time.
The selected-target panel shows the crop and safe metadata. A later valid click
replaces the current target; **Disable / Disengage Target** clears it.

Clicks in preview letterbox/pillarbox margins do nothing. A frame click with no
candidate reports that no selectable target is present and preserves the
current target. Where boxes overlap, selection prefers highest confidence,
then the smaller clipped area, then original detector order. Selection is
enabled only while the live runtime is running, but stopping live view does not
discard the current in-memory target.

Run Gate 2-5 deterministic offline acceptance and the complete suite with:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest tests/offline/test_three_source_capture.py tests/offline/test_gate3_detection_acceptance.py tests/offline/test_gate4_three_camera_ui.py tests/offline/test_gate5_target_selection.py -v
python -m pytest -q
```

Gate 5 creates no retained crops or other files and adds no configuration or
database schema. Targets exist only for the application session. Physical
camera selection and real-model selection remain unrun. Local tracking, dual
encoders, ReID, cross-camera identity, persistence/retention, uploads, target
naming, save/load, and named profiles remain later gates.
