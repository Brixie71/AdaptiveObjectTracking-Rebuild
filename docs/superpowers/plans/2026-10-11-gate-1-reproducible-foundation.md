# Gate 1 Reproducible Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a reproducible, secret-safe Windows/Python foundation whose configuration, compute-device selection, model manifest, logging, and environment checks pass before camera or inference work begins.

**Architecture:** Use a small installable `src/aot` package with immutable configuration data classes, strict YAML boundary validation, environment-only secret resolution, a capability-first PyTorch device probe, and standard-library structured logging. A thin script composes those services into the Gate 1 diagnostic; no capture, model inference, tracking, persistence, or UI behavior enters this gate.

**Tech Stack:** Python 3.11 x64, PyYAML 6.x, python-dotenv 1.x, PyTorch/torchvision hardware-selected pair, standard-library `logging`, pytest 8.x.

**Spec:** `docs/superpowers/specs/2026-10-11-adaptive-object-tracking-architecture-design.md`

## Global Constraints

- Primary acceptance platform: Windows 10 or 11 x64 with Python 3.11 x64 in `.venv`; do not develop against system Python.
- Historical `ThesisV2` material and `docs/reference/` are read-only evidence and never runtime imports or model paths.
- Install the official hardware-compatible `torch` and `torchvision` pair together before the remaining requirements; do not guess a CUDA wheel.
- CPU startup must not crash when CUDA is absent or unusable.
- Secrets come from environment variables and must not appear in committed configuration, object representations, logs, errors, or diagnostic exports.
- RTSP configuration stores an environment-variable key, never a resolved URL.
- Gate 1 contains environment, dependencies, configuration, device probing, model-manifest validation, and structured logging only.
- Gate 1 exit command is `python scripts/check_environment.py` on the intended Windows machine.
- All runtime output belongs under ignored `runtime/`; large weights remain outside Git and are declared through `models/manifest.yaml`.
- Add behavior test-first, run the focused test before the full Gate 1 suite, and commit each task independently.

## Review Focus

- A malformed or unknown YAML key must produce a path-aware `ConfigError`, not a raw parser exception or silently ignored setting; Task 2 pins this behavior.
- A missing RTSP environment variable or a URL containing credentials must never leak its resolved value through configuration representations or errors; Tasks 2 and 5 pin this behavior.
- CUDA reported as available with an invalid device index or a failing device query must resolve safely to CPU with a reason; Task 4 pins this behavior.
- An unwritable log directory must become a sanitized failed environment check and nonzero CLI result, not an uncaught traceback; Tasks 5 and 6 pin this behavior.
- The wrong Python version, non-64-bit interpreter, or missing required package must be listed as a failed check with actionable details; Task 6 pins this behavior.

---

## Gate Roadmap

This document is executable only for Gate 1. Later gates receive separate plans after the preceding gate passes so their interfaces are based on verified code rather than guesses.

| Gate | Planned deliverable | Exit evidence |
| --- | --- | --- |
| 0 | Historical freeze, reference manifest, and traceability baseline | Existing Gate 0 documents plus outstanding owner confirmations recorded honestly |
| 1 | Reproducible foundation detailed below | Environment CLI, focused tests, `pip check`, and sanitized machine report |
| 2 | Common USB, RTSP, and offline-video capture contract; initial Hardware Testing | Adapter contract tests, offline source test, authorized hardware checks |
| 3 | Detector contracts, YOLO/YuNet adapters, configurable allowlist | Hashed offline fixtures and bounded-detection tests |
| 4 | Responsive three-source PyQt5 view and bounded frame workers | `pytest-qt` evidence and three-source offline run |
| 5 | Target selection and reference capture | Hit-test, crop, and uploaded-image validation |
| 6 | Independent per-camera tracking | Recorded continuity and isolation tests |
| 7 | Person ReID and generic-object encoders | Separate embedding/quality/threshold distributions |
| 8 | Cross-camera identity, topology, and review states | Labeled handoff, ambiguity, merge, and split scenarios |
| 9 | Lost-target state and recovery | Timeout, recovery, and false-reassociation tests |
| 10 | SQLite records, retention, persistent targets, and saved profiles | Migration/CRUD/retention/profile round trips |
| 11 | Full performance and hardware acceptance | Three-source GPU/CPU report and reproducible thesis evaluation |

## Planned Gate 1 File Structure

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | Installable `src` package metadata and pytest discovery |
| `src/aot/__init__.py` | Package version only |
| `src/aot/config/types.py` | Immutable configuration domain types |
| `src/aot/config/loader.py` | Strict application/source YAML loading and RTSP secret resolution |
| `src/aot/config/model_manifest.py` | Strict model declaration and provenance validation |
| `src/aot/telemetry/device_probe.py` | Capability-first CPU/CUDA selection |
| `src/aot/telemetry/logging_setup.py` | JSON-line logging and sensitive-value redaction |
| `src/aot/telemetry/environment.py` | Composed environment checks and sanitized report types |
| `scripts/check_environment.py` | Thin Gate 1 CLI entry point |
| `config/app.example.yaml` | Non-sensitive application/compute defaults |
| `config/cameras.example.yaml` | Disabled USB, RTSP-key, and offline source examples |
| `models/manifest.yaml` | Versioned, initially empty model registry |
| `tests/unit/...` | Focused configuration, manifest, device, and logging tests |
| `tests/integration/test_check_environment.py` | Gate 1 composition and exit-code tests |
| `README.md` | Exact setup and Gate 1 verification commands |
| `docs/traceability/MANUSCRIPT_TO_SOURCE_MATRIX.md` | Evidence links and truthful Gate 1 status updates |

### Task 1: Package Bootstrap and Configuration Types

**Files:**
- Create: `pyproject.toml`
- Create: `src/aot/__init__.py`
- Create: `src/aot/config/__init__.py`
- Create: `src/aot/config/types.py`
- Create: `tests/unit/config/test_types.py`

**Interfaces:**
- Consumes: no project code.
- Produces: `SourceType`, `ComputePreference`, `ComputeConfig`, `CameraSourceConfig`, and `AppSettings` from `aot.config.types`.

- [ ] **Step 1: Add package/test metadata and write the failing type tests**

Create `pyproject.toml` with project name `adaptive-object-tracking`, version `0.1.0`, `setuptools.build_meta`, package discovery under `src`, `requires-python = "==3.11.*"`, and pytest paths `tests` plus Python path `src`. Set `aot.__version__` to the same `0.1.0` value. In `test_types.py`, assert:

```python
def test_camera_source_defaults_are_canonical():
    source = CameraSourceConfig(
        source_id="camera-1",
        name="Camera 1",
        source_type=SourceType.USB,
        device_index=0,
    )
    assert (source.width, source.height, source.fps, source.rotation) == (640, 480, 15.0, 0)
    assert source.enabled is False
```

Also assert the data classes are frozen, `SourceType` accepts only `usb`, `rtsp`, or `offline`, and `ComputePreference` accepts only `auto`, `cpu`, or `cuda`.

- [ ] **Step 2: Run the focused tests and verify the import failure**

Run: `python -m pytest tests/unit/config/test_types.py -v`

Expected: FAIL because `aot.config.types` does not exist.

- [ ] **Step 3: Implement the immutable configuration types**

Implement these exact frozen data classes in `src/aot/config/types.py`:

```python
class SourceType(str, Enum): ...
class ComputePreference(str, Enum): ...

@dataclass(frozen=True)
class ComputeConfig:
    preference: ComputePreference = ComputePreference.AUTO
    cuda_index: int = 0

@dataclass(frozen=True)
class CameraSourceConfig:
    source_id: str
    name: str
    source_type: SourceType
    enabled: bool = False
    device_index: int | None = None
    rtsp_env_key: str | None = None
    offline_path: Path | None = None
    width: int = 640
    height: int = 480
    fps: float = 15.0
    rotation: int = 0

@dataclass(frozen=True)
class AppSettings:
    schema_version: int
    runtime_dir: Path
    log_level: str
    compute: ComputeConfig
    sources: tuple[CameraSourceConfig, ...]
```

Expose these names from `aot.config`. Keep validation at the loader boundary in Task 2.

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest tests/unit/config/test_types.py -v`

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```powershell
git add pyproject.toml src/aot/__init__.py src/aot/config tests/unit/config/test_types.py
git commit -m "build: establish typed project foundation"
```

### Task 2: Strict Configuration and Secret Resolution

**Files:**
- Create: `src/aot/config/loader.py`
- Create: `config/app.example.yaml`
- Create: `config/cameras.example.yaml`
- Modify: `.env.example`
- Create: `tests/unit/config/test_loader.py`

**Interfaces:**
- Consumes: Task 1 configuration types.
- Produces: `ConfigError`, `load_settings(app_path: Path, cameras_path: Path, env_path: Path | None = None) -> AppSettings`, and `resolve_rtsp_url(source: CameraSourceConfig, environment: Mapping[str, str] | None = None) -> str`.

- [ ] **Step 1: Write failing happy-path and strict-validation tests**

Tests must assert that `load_settings` parses all three source kinds, resolves relative paths against the configuration file directory, and rejects:

- unknown keys with `ConfigError` containing the file path and key name;
- duplicate or blank source IDs;
- nonpositive width, height, or FPS;
- rotation outside `0`, `90`, `180`, `270`;
- missing or conflicting source-specific fields (`device_index`, `rtsp_env_key`, `offline_path`);
- unsupported `schema_version` values.

Add the review-focus secret test:

```python
def test_missing_rtsp_environment_value_does_not_leak_other_values(monkeypatch):
    monkeypatch.setenv("UNRELATED_SECRET", "do-not-print")
    source = CameraSourceConfig(
        source_id="cctv-1",
        name="CCTV 1",
        source_type=SourceType.RTSP,
        rtsp_env_key="CAMERA_1_URL",
    )
    with pytest.raises(ConfigError) as error:
        resolve_rtsp_url(source, {})
    assert "CAMERA_1_URL" in str(error.value)
    assert "do-not-print" not in str(error.value)
```

- [ ] **Step 2: Run the loader tests and verify failure**

Run: `python -m pytest tests/unit/config/test_loader.py -v`

Expected: FAIL because `aot.config.loader` does not exist.

- [ ] **Step 3: Implement strict loading and resolution**

Use `yaml.safe_load`. Reject unknown keys at every parsed mapping. Load `.env` with `override=False` only when `env_path` is supplied. Never place resolved RTSP values into `CameraSourceConfig` or `AppSettings`; `resolve_rtsp_url` returns the value only to an authorized runtime caller. `ConfigError` messages may name an environment key but never its value.

Create non-sensitive example files with `schema_version: 1`. Include disabled examples for USB index `0`, RTSP key `CAMERA_1_URL`, and an offline path. Expand `.env.example` with key names only and explanatory comments.

- [ ] **Step 4: Run the focused configuration tests**

Run: `python -m pytest tests/unit/config/test_loader.py -v`

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```powershell
git add .env.example config src/aot/config tests/unit/config/test_loader.py
git commit -m "feat: load validated secret-safe configuration"
```

### Task 3: Model Manifest Boundary

**Files:**
- Create: `src/aot/config/model_manifest.py`
- Create: `models/README.md`
- Create: `models/manifest.yaml`
- Create: `tests/unit/config/test_model_manifest.py`

**Interfaces:**
- Consumes: filesystem paths only.
- Produces: immutable `ModelSpec`, immutable `ModelManifest`, and `load_model_manifest(path: Path) -> ModelManifest`.

- [ ] **Step 1: Write failing manifest tests**

Test an empty version-1 manifest and a valid declaration. Assert rejection of unknown keys, duplicate model IDs, absolute or parent-traversing model paths, unsupported roles, and a supplied SHA-256 that is not exactly 64 hexadecimal characters. Supported roles at this boundary are `detector`, `face_detector`, `person_encoder`, and `generic_encoder`.

- [ ] **Step 2: Run the manifest tests and verify failure**

Run: `python -m pytest tests/unit/config/test_model_manifest.py -v`

Expected: FAIL because `aot.config.model_manifest` does not exist.

- [ ] **Step 3: Implement manifest types and strict loader**

Use these signatures:

```python
@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    role: str
    path: Path
    source: str
    sha256: str | None = None
    required: bool = False

@dataclass(frozen=True)
class ModelManifest:
    schema_version: int
    models: tuple[ModelSpec, ...]

def load_model_manifest(path: Path) -> ModelManifest: ...
```

Keep the committed manifest at `schema_version: 1` with an empty model list until a later gate selects and fingerprints actual weights. Document the deliberate copy/provenance/hash/license process in `models/README.md`.

- [ ] **Step 4: Run the focused manifest tests**

Run: `python -m pytest tests/unit/config/test_model_manifest.py -v`

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```powershell
git add models src/aot/config/model_manifest.py tests/unit/config/test_model_manifest.py
git commit -m "feat: validate model provenance manifest"
```

### Task 4: Capability-first CPU/CUDA Probe

**Files:**
- Create: `src/aot/telemetry/__init__.py`
- Create: `src/aot/telemetry/device_probe.py`
- Create: `tests/unit/telemetry/test_device_probe.py`

**Interfaces:**
- Consumes: Task 1 `ComputeConfig` and PyTorch capability APIs.
- Produces: `ComputeDeviceInfo` and `probe_compute_device(config: ComputeConfig) -> ComputeDeviceInfo`.

- [ ] **Step 1: Write failing device-resolution tests**

Test explicit CPU, automatic CUDA, automatic CPU fallback, requested CUDA with no CUDA, out-of-range CUDA index, and an exception from `torch.cuda.get_device_name`. Assert no CUDA-only query occurs after availability is false. The invalid-index test must assert `resolved == "cpu"` and a nonempty `fallback_reason`.

- [ ] **Step 2: Run device tests and verify failure**

Run: `python -m pytest tests/unit/telemetry/test_device_probe.py -v`

Expected: FAIL because `aot.telemetry.device_probe` does not exist.

- [ ] **Step 3: Implement the device report and safe probe**

```python
@dataclass(frozen=True)
class ComputeDeviceInfo:
    requested: ComputePreference
    resolved: str
    cuda_available: bool
    cuda_device_count: int
    device_name: str | None
    torch_version: str
    fallback_reason: str | None

def probe_compute_device(config: ComputeConfig) -> ComputeDeviceInfo: ...
```

Check `torch.cuda.is_available()` before count, index, name, or other CUDA-only operations. Treat probe exceptions as a CPU fallback and preserve a sanitized reason; do not crash startup.

- [ ] **Step 4: Run focused device tests**

Run: `python -m pytest tests/unit/telemetry/test_device_probe.py -v`

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```powershell
git add src/aot/telemetry tests/unit/telemetry/test_device_probe.py
git commit -m "feat: add safe compute device probe"
```

### Task 5: Structured Logging and Redaction

**Files:**
- Create: `src/aot/telemetry/logging_setup.py`
- Create: `tests/unit/telemetry/test_logging_setup.py`

**Interfaces:**
- Consumes: a log level, runtime log directory, and caller-supplied sensitive values.
- Produces: `LoggingSetupError`, `redact_text(value: str, secrets: Iterable[str] = ()) -> str`, and `configure_logging(level: str, log_dir: Path, secrets: Iterable[str] = ()) -> logging.Logger`.

- [ ] **Step 1: Write failing JSON and redaction tests**

Assert each emitted line parses as JSON and includes UTC timestamp, level, logger, message, and supplied `source_id`, `session_id`, `track_id`, or `identity_id` context. Assert literal secrets and URL user-info are replaced with `[REDACTED]` in messages, arguments, and formatted exception text. Mock directory creation failure and assert `LoggingSetupError` contains the directory but no secret.

- [ ] **Step 2: Run logging tests and verify failure**

Run: `python -m pytest tests/unit/telemetry/test_logging_setup.py -v`

Expected: FAIL because `aot.telemetry.logging_setup` does not exist.

- [ ] **Step 3: Implement JSON-line logging and final-output redaction**

Use standard-library logging with stderr and rotating-file handlers. Redact the final formatted JSON string so secrets cannot bypass filtering through arguments or exception rendering. Reconfiguring the `aot` logger must replace project-owned handlers instead of duplicating output.

- [ ] **Step 4: Run focused logging tests**

Run: `python -m pytest tests/unit/telemetry/test_logging_setup.py -v`

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

```powershell
git add src/aot/telemetry/logging_setup.py tests/unit/telemetry/test_logging_setup.py
git commit -m "feat: add structured redacted logging"
```

### Task 6: Gate 1 Environment Report and CLI

**Files:**
- Create: `src/aot/telemetry/environment.py`
- Create: `scripts/check_environment.py`
- Create: `tests/unit/telemetry/test_environment.py`
- Create: `tests/integration/test_check_environment.py`

**Interfaces:**
- Consumes: Tasks 2-5 loaders, probe, and logging setup.
- Produces: `CheckStatus`, `EnvironmentCheck`, `EnvironmentReport`, `check_environment(...)`, and CLI `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write failing report and composition tests**

Use these signatures:

```python
def check_environment(
    app_path: Path,
    cameras_path: Path,
    manifest_path: Path,
    env_path: Path | None = None,
) -> EnvironmentReport: ...

def main(argv: Sequence[str] | None = None) -> int: ...
```

Tests must assert:

- Python `3.11` and 64-bit are separate checks;
- required distributions are reported with installed versions;
- valid configuration and the empty manifest pass;
- CPU fallback is a warning, not a failure;
- malformed configuration, missing packages, and logging setup failure produce named failed checks;
- report serialization contains no supplied RTSP secret;
- CLI returns `0` when there are no failed checks and `1` otherwise;
- `--json` emits parseable sanitized JSON.

- [ ] **Step 2: Run the environment tests and verify failure**

Run: `python -m pytest tests/unit/telemetry/test_environment.py tests/integration/test_check_environment.py -v`

Expected: FAIL because the environment module and script do not exist.

- [ ] **Step 3: Implement report types and composed checks**

Define `CheckStatus` values `pass`, `warning`, and `fail`. `EnvironmentReport.ok` is true only when no check failed. Report platform, architecture, Python, configured paths, required distribution versions, configuration/manifest validation, compute result, and logging writability. Catch expected boundary errors and convert them into sanitized checks; preserve unexpected programming errors for tests and development rather than swallowing everything.

The required distribution names are `torch`, `torchvision`, `ultralytics`, `opencv-python`, `numpy`, `Pillow`, `PyQt5`, `PyYAML`, `python-dotenv`, `psutil`, `onnx`, `onnxruntime`, `pytest`, and `pytest-qt`. Query their installed versions through `importlib.metadata`; a missing distribution is a failed check.

The script must be a thin import-and-exit wrapper with defaults:

```text
--app-config config/app.example.yaml
--camera-config config/cameras.example.yaml
--model-manifest models/manifest.yaml
--env-file .env
```

An absent default `.env` is allowed because the committed example sources are disabled. Human-readable output is the default; `--json` is deterministic and safe to save as evidence.

- [ ] **Step 4: Run focused environment tests**

Run: `python -m pytest tests/unit/telemetry/test_environment.py tests/integration/test_check_environment.py -v`

Expected: PASS.

- [ ] **Step 5: Commit Task 6**

```powershell
git add src/aot/telemetry/environment.py scripts/check_environment.py tests/unit/telemetry/test_environment.py tests/integration/test_check_environment.py
git commit -m "feat: add Gate 1 environment diagnostic"
```

### Task 7: Documentation, Traceability, and Gate 1 Acceptance

**Files:**
- Modify: `README.md`
- Modify: `docs/traceability/MANUSCRIPT_TO_SOURCE_MATRIX.md`
- Create only after a successful intended-machine run: `docs/evidence/gate-1-environment.md`

**Interfaces:**
- Consumes: all Gate 1 tasks.
- Produces: reproducible operator commands and evidence links; no new runtime API.

- [ ] **Step 1: Document exact setup and verification commands**

Add PowerShell instructions to create/activate `.venv`, install the official PyTorch pair, install `requirements.txt`, install this package editable with `python -m pip install -e . --no-deps`, run tests, run `pip check`, and invoke the Gate 1 script. State that CPU fallback may warn but must not fail startup.

- [ ] **Step 2: Run the complete automated Gate 1 suite**

Run:

```powershell
python -m pytest tests/unit/config tests/unit/telemetry tests/integration/test_check_environment.py -v
```

Expected: all collected tests PASS with zero failures.

- [ ] **Step 3: Validate the installed environment**

Run:

```powershell
python -m pip check
python scripts/check_environment.py
python scripts/check_environment.py --json
```

Expected: `pip check` reports no broken requirements; both environment commands exit `0`; JSON contains no credential values. A CPU-only machine may report a warning.

- [ ] **Step 4: Record evidence without overstating it**

If Step 3 ran on the intended Windows machine, create `docs/evidence/gate-1-environment.md` with the date, machine label, Python/architecture, package versions, resolved compute device, commands, exit codes, and sanitized output summary. Update AOT-005, AOT-028, and AOT-029 with implementation and test links. Mark a row `REBUILD_VERIFIED` only when its listed acceptance evidence actually passed; otherwise retain its prior status and record the missing evidence.

- [ ] **Step 5: Confirm Gate 1 scope and repository hygiene**

Run:

```powershell
git status --short
git grep -n -I -E "rtsp://[^[:space:]]+@|password[[:space:]]*=|token[[:space:]]*=" -- . ":(exclude)docs/reference/**"
git diff --check
```

Expected: only intended documentation/evidence changes remain, the credential scan returns no live secret, and the diff check reports no errors. Confirm no capture, detection, tracking, ReID, persistence, or UI implementation was added.

- [ ] **Step 6: Commit Gate 1 documentation and evidence**

```powershell
git add README.md docs/traceability/MANUSCRIPT_TO_SOURCE_MATRIX.md docs/evidence/gate-1-environment.md
git commit -m "docs: record Gate 1 verification"
```

If intended-machine evidence was not produced, omit `docs/evidence/gate-1-environment.md` from the command and state the remaining acceptance blocker in the commit message/body.

## Gate 1 Completion Boundary

Gate 1 is complete only when the automated suite passes, `pip check` is clean, `python scripts/check_environment.py` exits successfully on the intended Windows environment, secrets remain absent from output and Git, and traceability records the evidence. Passing unit tests alone does not authorize Gate 2.

After Gate 1 acceptance, write and review the separate Gate 2 capture-and-hardware-testing implementation plan before adding camera code.
