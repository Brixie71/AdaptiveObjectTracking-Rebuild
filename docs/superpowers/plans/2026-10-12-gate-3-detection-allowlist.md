# Gate 3 Detection and Class Allowlist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a project-owned YOLOv8 detection layer that converts captured frames into validated detections and enforces a configurable people-and-generic-object class allowlist before any downstream consumer can observe results.

**Architecture:** Immutable detection contracts isolate the application from Ultralytics result objects. Strict application configuration selects one declared detector model and names the enabled classes; a verified model-store resolver checks role, path containment, existence, and SHA-256 before the adapter loads weights. The synchronous detector remains UI-independent and receives the Gate 1 resolved compute device explicitly; later gates may schedule it off the GUI thread without changing its contract.

**Tech Stack:** Python 3.11, Ultralytics 8.4.158, PyTorch 2.14.1, OpenCV 4, NumPy, immutable dataclasses, runtime-checkable protocols, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-10-11-adaptive-object-tracking-architecture-design.md`

## Global Constraints

- Windows 10 or 11 x64 and Python `3.11.*` remain the first acceptance platform.
- Detection accepts only canonical Gate 2 `FramePacket` inputs: contiguous NumPy `uint8` BGR frames with source, sequence, and monotonic capture metadata.
- The application owns all public detection records; Ultralytics/PyTorch objects never cross the adapter boundary.
- The configured allowlist uses detector class names, supports people and generic-object classes, and is enforced inside the detector before results reach tracking, UI, persistence, or ReID.
- Class matching is case-insensitive after trimming, while emitted `class_name` values use the detector model's canonical label.
- Unknown or ambiguous configured class names block detector creation with an actionable sanitized error; an enabled detector requires a non-empty allowlist.
- Bounding boxes are clipped to the source frame and must remain finite, non-degenerate, and within `0 <= x1 < x2 <= width` and `0 <= y1 < y2 <= height`.
- Confidence values must be finite and within `[0.0, 1.0]`; malformed backend rows are dropped and counted, not propagated.
- A detector model is loaded only from a manifest declaration with role `detector`, a safe path contained by the model store, an existing regular file, and a matching declared SHA-256 digest.
- Model weights remain excluded from Git. Automated tests use injected fakes and temporary artifacts; a real model probe is opt-in and requires a locally provisioned, manifest-declared artifact.
- Compute selection comes from the Gate 1 `ComputeDeviceInfo.resolved` value. The detector does not independently select CUDA or silently change devices.
- Backend load and inference exceptions are translated into project-owned errors that do not expose third-party tracebacks, frame contents, environment values, or credential-bearing strings.
- Gate 3 adds no PyQt UI, target selection, tracking, ReID/embedding, global identity, persistence, saved profiles, or retention behavior.
- Default automated tests require no camera, network, GUI, CUDA device, or real model weights.

## Configuration Decisions

The optional `detection` block has these exact defaults when omitted:

```yaml
detection:
  enabled: false
  model_id: null
  confidence_threshold: 0.25
  iou_threshold: 0.70
  max_detections: 300
  allowed_classes:
    - person
```

When `enabled: true`, `model_id` must be a nonblank string and `allowed_classes` must contain at least one unique nonblank class name after case-insensitive normalization. `confidence_threshold` and `iou_threshold` accept finite numeric values in `(0.0, 1.0]`; `max_detections` is a positive integer and rejects booleans. The checked-in example remains disabled because model weights are deliberately not committed, but it demonstrates both people and generic objects with `allowed_classes: [person, backpack, bottle]`.

## Public Detection Contracts

- `BoundingBox(x1: float, y1: float, x2: float, y2: float)` contains clipped pixel coordinates.
- `Detection(class_id: int, class_name: str, confidence: float, box: BoundingBox)` represents one allow-listed observation.
- `DetectionBatch(source_id: str, frame_sequence: int, captured_at_monotonic_ns: int, frame_width: int, frame_height: int, detections: tuple[Detection, ...], rejected_count: int)` preserves capture provenance and reports malformed/disallowed backend rows without exposing them.
- `Detector` is a runtime-checkable protocol with `model_id: str`, `class_names: tuple[str, ...]`, `detect(packet: FramePacket) -> DetectionBatch`, and `close() -> None`.
- `DetectionError` is the sanitized base exception; `DetectorLoadError` and `DetectorInferenceError` distinguish startup from per-frame failure.

## Review Focus

- Duplicate allowlist entries that differ only by case or whitespace must be rejected so user intent is never silently collapsed; Task 1 adds this regression.
- A manifest path that resolves outside the model store through `..`, an absolute path, or a filesystem link must never be loaded; Task 2 adds containment regressions.
- A detector artifact whose bytes do not match the declared SHA-256 must fail before Ultralytics receives the path; Task 2 adds this regression.
- NaN/Infinity confidence or coordinates, reversed boxes, boxes outside the frame, and invalid class IDs must not escape as detections; Task 3 adds these regressions.
- A backend exception containing sensitive text must become a generic project-owned load or inference error, and CLI JSON must remain sanitized; Tasks 4 and 5 add these regressions.

---

### Task 1: Detection Configuration and Allowlist Policy

**Files:**
- Modify: `src/aot/config/types.py`
- Modify: `src/aot/config/loader.py`
- Modify: `src/aot/config/__init__.py`
- Modify: `config/app.example.yaml`
- Test: `tests/unit/config/test_loader.py`
- Test: `tests/unit/config/test_types.py`

**Interfaces:**
- Consumes: the strict Gate 1 application-configuration loader.
- Produces: frozen `DetectionConfig(enabled: bool = False, model_id: str | None = None, confidence_threshold: float = 0.25, iou_threshold: float = 0.70, max_detections: int = 300, allowed_classes: tuple[str, ...] = ("person",))` and `AppSettings.detection`.
- Produces internal loader `_load_detection(value: Any, path: Path) -> DetectionConfig`.

- [ ] **Step 1: Write failing tests for detection defaults and valid explicit configuration**

Add tests asserting the exact defaults above when `detection` is absent. Add one valid enabled example with model ID `yolov8n-coco` and `allowed_classes: [person, backpack, bottle]`; assert order is preserved.

- [ ] **Step 2: Run the focused tests to verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/config/test_loader.py tests/unit/config/test_types.py -k detection -v`

Expected: FAIL because `DetectionConfig` and `AppSettings.detection` do not exist.

- [ ] **Step 3: Add strict invalid-input tests**

Cover an unknown key; non-boolean `enabled`; missing, blank, or non-string `model_id` when enabled; any non-null `model_id` with the wrong type; empty/non-list `allowed_classes`; blank/non-string entries; case-insensitive duplicates; booleans, zero, negative, values above `1.0`, NaN, Infinity, and wrong types for thresholds; and boolean, zero, negative, or non-integer `max_detections`.

- [ ] **Step 4: Implement `DetectionConfig` and `_load_detection`**

Add `detection` to the application key boundary and to `AppSettings`. Preserve trimmed user class labels in their original order; use `casefold()` only to detect duplicates. When disabled, allow `model_id: null`; when enabled, require the nonblank model ID and non-empty allowlist.

- [ ] **Step 5: Document the disabled example configuration**

Add the exact block from Configuration Decisions to `config/app.example.yaml`, changing its demonstration allowlist to `[person, backpack, bottle]`. Keep `enabled: false` and `model_id: null` so a clean checkout still passes environment validation without untracked model weights.

- [ ] **Step 6: Run Task 1 tests and the full suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/config -q`

Then: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 7: Commit Task 1**

```powershell
git add config/app.example.yaml src/aot/config tests/unit/config
git commit -m "feat: configure detector and class allowlist"
```

---

### Task 2: Verified Detector Artifact Resolution

**Files:**
- Create: `src/aot/config/model_store.py`
- Modify: `src/aot/config/__init__.py`
- Modify: `src/aot/config/model_manifest.py`
- Test: `tests/unit/config/test_model_store.py`
- Test: `tests/unit/config/test_model_manifest.py`

**Interfaces:**
- Consumes: `ModelManifest`, `ModelSpec`, a selected `model_id`, and a model-store root.
- Produces: frozen `ResolvedModelArtifact(model_id: str, role: str, path: Path, source: str, sha256: str)`.
- Produces: `resolve_model_artifact(manifest: ModelManifest, *, model_id: str, expected_role: str, model_store: Path) -> ResolvedModelArtifact`.
- Produces: `ModelArtifactError(ValueError)` with sanitized summaries.

- [ ] **Step 1: Write failing resolver tests**

Using only temporary directories and small byte files, assert successful resolution returns an absolute contained path and the lowercase verified digest. Assert failure for an unknown model ID, wrong role, missing digest, missing path, directory instead of file, mismatched digest, an unsupported `expected_role`, and a resolved path outside the model store.

- [ ] **Step 2: Add link-containment coverage**

Where the host permits creating a symbolic link or junction, point a manifest path inside the store to a file outside it and assert resolution fails. Skip only that single test when the host denies link creation.

- [ ] **Step 3: Run resolver tests to verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/config/test_model_store.py -v`

Expected: FAIL because `aot.config.model_store` does not exist.

- [ ] **Step 4: Implement deterministic artifact verification**

Resolve the model-store root and candidate with `Path.resolve(strict=True)`, confirm the candidate is a regular file and remains under the resolved store with `Path.is_relative_to`, stream SHA-256 in fixed-size chunks, and compare with `hmac.compare_digest`. Require a declared digest at runtime even though the general manifest schema remains capable of describing not-yet-provisioned later-gate artifacts.

- [ ] **Step 5: Tighten duplicate manifest semantics**

Extend manifest tests so `model_id` uniqueness is case-insensitive after trimming, then update `load_model_manifest` to reject `Detector` and `detector` as duplicates. Do not change the version-1 manifest fields.

- [ ] **Step 6: Run Task 2 tests and the full suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/config/test_model_manifest.py tests/unit/config/test_model_store.py -q`

Then: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 7: Commit Task 2**

```powershell
git add src/aot/config tests/unit/config
git commit -m "feat: verify detector model artifacts"
```

---

### Task 3: Detection Domain Contracts and Result Normalization

**Files:**
- Create: `src/aot/detection/__init__.py`
- Create: `src/aot/detection/types.py`
- Create: `src/aot/detection/normalize.py`
- Create: `tests/unit/detection/test_types.py`
- Create: `tests/unit/detection/test_normalize.py`

**Interfaces:**
- Consumes: one `FramePacket`, backend rows shaped as `(x1, y1, x2, y2, confidence, class_id)`, a backend class-name mapping, and configured allowed class names.
- Produces: the Public Detection Contracts above.
- Produces: `resolve_allowed_class_ids(class_names: Mapping[int, str], allowed_classes: tuple[str, ...]) -> frozenset[int]`.
- Produces: `normalize_detections(packet: FramePacket, rows: Iterable[Sequence[object]], *, class_names: Mapping[int, str], allowed_class_ids: AbstractSet[int], max_detections: int) -> DetectionBatch`.

- [ ] **Step 1: Write failing immutable-contract tests**

Assert records are frozen, `Detector` is runtime-checkable, errors retain only the supplied sanitized summary, and `DetectionBatch` preserves the exact source ID, frame sequence, timestamp, and frame dimensions from its packet.

- [ ] **Step 2: Run contract tests to verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/detection/test_types.py -v`

Expected: FAIL because `aot.detection` does not exist.

- [ ] **Step 3: Implement the domain records, protocol, and errors**

Use frozen dataclasses and a `@runtime_checkable Protocol`. Keep NumPy, Torch, Ultralytics, and raw backend rows out of every public record.

- [ ] **Step 4: Write failing class-resolution and normalization tests**

Cover case-insensitive resolution to canonical model labels; unknown labels; ambiguous model labels that collide under `casefold()`; allowlisted and disallowed rows; stable backend order; `max_detections`; frame-edge clipping; and `rejected_count`. Parameterize malformed rows for wrong length/type, booleans, NaN/Infinity, confidence outside `[0, 1]`, unknown/negative/non-integral class IDs, and boxes that remain reversed or degenerate after clipping.

- [ ] **Step 5: Run normalization tests to verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/detection/test_normalize.py -v`

Expected: FAIL because normalization functions do not exist.

- [ ] **Step 6: Implement allowlist resolution and result normalization**

Validate the packet frame shape defensively, resolve configured names once during detector construction, filter disallowed classes before allocating public `Detection` records, clip coordinates to the frame, and count every disallowed or malformed row in `rejected_count`. Stop after `max_detections` accepted results while preserving backend confidence order.

- [ ] **Step 7: Run Task 3 tests and the full suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/detection -q`

Then: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 8: Commit Task 3**

```powershell
git add src/aot/detection tests/unit/detection
git commit -m "feat: define normalized detection contracts"
```

---

### Task 4: Ultralytics YOLOv8 Adapter and Detector Factory

**Files:**
- Create: `src/aot/detection/backend.py`
- Create: `src/aot/detection/yolo.py`
- Create: `src/aot/detection/factory.py`
- Create: `tests/unit/detection/fakes.py`
- Create: `tests/unit/detection/test_yolo.py`
- Create: `tests/unit/detection/test_factory.py`

**Interfaces:**
- Consumes: `DetectionConfig`, a verified `ResolvedModelArtifact`, `ComputeDeviceInfo.resolved`, and canonical `FramePacket` values.
- Produces internal `YoloBackend` with `class_names: Mapping[int, str]`, `predict(frame: np.ndarray, *, confidence_threshold: float, iou_threshold: float, max_detections: int, device: str) -> Iterable[Sequence[object]]`, and `close() -> None`.
- Produces `open_ultralytics_backend(path: Path) -> YoloBackend`.
- Produces `YoloDetector(model_id: str, config: DetectionConfig, backend: YoloBackend, device: str)` implementing `Detector`.
- Produces `create_detector(config: DetectionConfig, manifest: ModelManifest, *, model_store: Path, compute: ComputeDeviceInfo, backend_factory: Callable[[Path], YoloBackend] = open_ultralytics_backend) -> Detector`.

- [ ] **Step 1: Write failing YOLO-adapter tests with an injected backend**

Assert constructor-time allowlist resolution, exact propagation of confidence/IoU/max-detection/device values, correct use of the packet frame, normalized project-owned output, repeated `close()` safety, detection after close failure, and sanitized translation of backend inference and cleanup exceptions.

- [ ] **Step 2: Run adapter tests to verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/detection/test_yolo.py -v`

Expected: FAIL because the adapter does not exist.

- [ ] **Step 3: Implement the backend protocol and `YoloDetector`**

The Ultralytics wrapper calls `YOLO(str(path))` only during backend creation. Prediction uses `model.predict(source=frame, conf=..., iou=..., max_det=..., device=..., verbose=False)` and converts only `boxes.xyxy`, `boxes.conf`, and `boxes.cls` to plain CPU values inside the wrapper. Require exactly one result for one input frame; translate missing/malformed result structures into `DetectorInferenceError`.

- [ ] **Step 4: Write failing factory tests**

Assert disabled detection is rejected as unsupported, the configured model ID is resolved with expected role `detector`, the verified absolute path is the only path given to the backend factory, the Gate 1 resolved device string is passed unchanged, an unknown allowlisted class fails before first inference, and raw resolver/backend exception text is not exposed.

- [ ] **Step 5: Run factory tests to verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/detection/test_factory.py -v`

Expected: FAIL because `create_detector` does not exist.

- [ ] **Step 6: Implement `create_detector` and the lazy Ultralytics import**

Keep `ultralytics` imports inside `open_ultralytics_backend` so project contracts and diagnostics can import when the optional runtime backend is broken. Resolve and verify the artifact before invoking the backend factory. If construction fails at any point after backend creation, close the backend before raising a generic `DetectorLoadError`.

- [ ] **Step 7: Run Task 4 tests and the full suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/detection -q`

Then: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 8: Commit Task 4**

```powershell
git add src/aot/detection tests/unit/detection
git commit -m "feat: adapt YOLOv8 detection results"
```

---

### Task 5: Hardware Testing Detector Probe and CLI

**Files:**
- Modify: `src/aot/hardware_testing/types.py`
- Create: `src/aot/hardware_testing/detector_probe.py`
- Modify: `src/aot/hardware_testing/__init__.py`
- Create: `scripts/probe_detector.py`
- Create: `tests/unit/hardware_testing/test_detector_probe.py`
- Create: `tests/integration/test_probe_detector.py`

**Interfaces:**
- Consumes: a constructed `Detector`, a canonical `FramePacket`, warm-up count, measured inference count, and injectable monotonic clock.
- Produces frozen `DetectorTestResult(model_id: str, status: HardwareTestStatus, device: str, warmup_runs: int, measured_runs: int, successful_runs: int, mean_inference_ms: float | None, p95_inference_ms: float | None, detection_count: int, started_at_utc: datetime, ended_at_utc: datetime, summary: str)` with `to_dict()`.
- Produces `probe_detector(detector: Detector, packet: FramePacket, *, device: str, warmup_runs: int = 2, measured_runs: int = 10, clock_ns: Callable[[], int] = time.perf_counter_ns) -> DetectorTestResult`.
- Produces CLI arguments `--app-config`, `--model-manifest`, `--models-dir`, `--warmup-runs`, `--measured-runs`, and `--json`.

- [ ] **Step 1: Write failing probe-service tests**

Use fake detectors and a fixed 640x480 black `FramePacket`. Cover successful timing, nearest-rank p95, zero detections as a successful inference, partial inference failure as `DEGRADED`, load/first-inference failure as `FAILED`, invalid counts, repeated cleanup, cleanup failure, and generic summaries that never include fake sensitive exception text.

- [ ] **Step 2: Run service tests to verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/hardware_testing/test_detector_probe.py -v`

Expected: FAIL because detector probing does not exist.

- [ ] **Step 3: Implement `DetectorTestResult` and `probe_detector`**

Warm-up runs do not contribute to measured timing. Generate aggregate measurements only; never serialize the input frame, boxes, raw backend results, artifact path, source URL, or exception detail. Always call `close()` exactly once from a `finally` block.

- [ ] **Step 4: Write failing CLI integration tests**

Inject settings, manifest, compute probe, detector factory, and detector probe. Assert a successful JSON schema and exit `0`; disabled configuration, absent model, invalid numeric arguments, factory failure, and failed/degraded probe exit `1`; human output is concise; JSON is parseable; and neither output channel contains injected credential-like text or tracebacks.

- [ ] **Step 5: Run CLI tests to verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/integration/test_probe_detector.py -v`

Expected: FAIL because the CLI entry point does not exist.

- [ ] **Step 6: Implement the detector-probe CLI**

Load the app settings and manifest, resolve compute once through `probe_compute_device`, create one contiguous black BGR frame at 640x480, build the detector through `create_detector`, and invoke the probe. A successful empty result proves load/inference compatibility but does not claim detection accuracy. Keep all CLI errors at the project boundary generic and sanitized.

- [ ] **Step 7: Run Task 5 tests and the full suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/unit/hardware_testing tests/integration/test_probe_detector.py -q`

Then: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 8: Commit Task 5**

```powershell
git add scripts/probe_detector.py src/aot/hardware_testing tests/unit/hardware_testing tests/integration/test_probe_detector.py
git commit -m "feat: add detector hardware testing probe"
```

---

### Task 6: Offline Capture-to-Detection Acceptance

**Files:**
- Create: `tests/integration/test_detection_pipeline.py`
- Create: `tests/offline/test_gate3_detection_acceptance.py`
- Modify: `tests/fixtures/media/README.md`

**Interfaces:**
- Consumes: Gate 2 `OfflineFrameSource`, three packets from `tests/fixtures/media/gate2-three-frame.avi`, and `YoloDetector` with an injected deterministic backend.
- Produces: deterministic proof that source/frame provenance survives capture-to-detection conversion and that disabled generic classes never reach the returned batch.

- [ ] **Step 1: Write the failing capture-to-detection integration test**

Open the Gate 2 fixture through `OfflineFrameSource`. For each packet, make the fake backend return ordered rows for `person`, `backpack`, and `car`, plus one malformed row. Configure `[person, backpack]` and assert each batch preserves source ID, increasing sequence, capture timestamp, and frame dimensions; contains canonical `person` and `backpack` detections only; excludes `car`; and reports both disallowed and malformed rows in `rejected_count`.

- [ ] **Step 2: Run the integration test to verify RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests/integration/test_detection_pipeline.py -v`

Expected: FAIL until all Gate 3 public interfaces are connected correctly.

- [ ] **Step 3: Make only the minimal integration corrections required**

Adjust exports or adapter boundaries, not the Gate 2 source implementation. Do not add tracking, identity state, persistence, or GUI scheduling.

- [ ] **Step 4: Add the offline acceptance test**

Run the three-source Gate 2 acceptance first, then exercise three independent offline sources through separate fake-backed `YoloDetector` instances. Assert one detector failure does not alter the other two results and every successful batch remains source-local and allowlist-filtered.

- [ ] **Step 5: Run Gate 2 and Gate 3 offline acceptance**

Run: `.\.venv\Scripts\python.exe -m pytest tests/offline/test_three_source_capture.py tests/offline/test_gate3_detection_acceptance.py -v`

Expected: PASS with no camera, network, GUI, CUDA, or real model artifact.

- [ ] **Step 6: Document fixture reuse**

Update `tests/fixtures/media/README.md` to state that Gate 3 reuses the deterministic three-frame AVI for pipeline provenance and isolation while an injected detector backend supplies deterministic rows. Explicitly state that this does not prove real-model accuracy.

- [ ] **Step 7: Run Task 6 tests and the full suite**

Run: `.\.venv\Scripts\python.exe -m pytest tests/integration/test_detection_pipeline.py tests/offline -q`

Then: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 8: Commit Task 6**

```powershell
git add tests/integration/test_detection_pipeline.py tests/offline/test_gate3_detection_acceptance.py tests/fixtures/media/README.md
git commit -m "test: verify Gate 3 detection pipeline"
```

---

### Task 7: Gate 3 Documentation, Traceability, and Acceptance Evidence

**Files:**
- Modify: `README.md`
- Modify: `models/README.md`
- Modify: `docs/traceability/MANUSCRIPT_TO_SOURCE_MATRIX.md`
- Create: `docs/evidence/gate-3-detection-allowlist.md`

**Interfaces:**
- Consumes: completed Gate 3 contracts, configuration, tests, and probe command.
- Produces: reproducible setup instructions, explicit proof boundaries, and Gate 3 evidence without overstating unrun model or hardware checks.

- [ ] **Step 1: Update README Gate 3 usage**

Document the detection configuration fields, class-name allowlist semantics, required manifest role/hash, default disabled state, CPU/GPU selection inherited from Gate 1, the detector-probe command, and the automated acceptance commands. State that target selection, tracking, ReID, cross-camera identity, persistence, and the UI remain outside Gate 3.

- [ ] **Step 2: Document safe model provisioning**

In `models/README.md`, add a concrete workflow that copies an owner-approved YOLOv8-compatible artifact into `models/yolo/`, computes SHA-256 with `Get-FileHash -Algorithm SHA256`, declares it with role `detector`, and enables its matching `model_id` in application configuration. Do not prescribe an unverified URL, redistribute weights, auto-download during tests, or record a fabricated digest.

- [ ] **Step 3: Update manuscript traceability**

Change AOT-007 only after implementation evidence exists. Cite the project-owned detection contract, YOLO adapter, configurable allowlist tests, offline acceptance, and opt-in detector probe. Keep real-model accuracy, physical-camera inference, tracking, and cross-camera identity explicitly unverified unless separate evidence was actually recorded.

- [ ] **Step 4: Run final automated acceptance**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest tests/offline/test_gate3_detection_acceptance.py -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/check_environment.py --json
```

Expected: all tests PASS, `pip check` reports no broken requirements, and the environment JSON has `"ok": true` with the checked-in disabled detector example.

- [ ] **Step 5: Run the opt-in real-model probe only when an approved artifact exists**

Run:

```powershell
.\.venv\Scripts\python.exe scripts/probe_detector.py --json
```

Expected when a valid declared artifact is configured: exit `0`, status `success`, finite nonnegative timing, and sanitized JSON. If no approved artifact is present, record `NOT RUN`; do not download weights implicitly or claim real-model acceptance.

- [ ] **Step 6: Scan implementation paths for credential-like material**

Run a focused scan over `src`, `scripts`, `config`, `models`, and non-reference tests. Exclude `docs/reference`, approved prerequisite/design/plan prose, and credential-redaction fixtures. Inspect every match before recording the result.

- [ ] **Step 7: Write Gate 3 evidence**

Record the exact commit, model ID and SHA-256 if a real probe ran, fixture hash, commands, exit codes, test totals, compute resolution, proof boundaries, and `NOT RUN` items in `docs/evidence/gate-3-detection-allowlist.md`. Never copy a source URL containing credentials or raw exception output.

- [ ] **Step 8: Commit Task 7**

```powershell
git add README.md models/README.md docs/traceability/MANUSCRIPT_TO_SOURCE_MATRIX.md docs/evidence/gate-3-detection-allowlist.md
git commit -m "docs: record Gate 3 detection evidence"
```

---

## Gate 3 Acceptance Boundary

Gate 3 is ready for owner review only when:

- the default automated suite passes without cameras, network, GUI, CUDA, or model weights;
- the detector configuration is strict and the checked-in example remains safe for a clean checkout;
- model role, path containment, regular-file status, and SHA-256 are verified before backend loading;
- all public results use immutable project-owned detection records;
- allowlist filtering occurs before results leave the detector;
- people and enabled generic-object classes are demonstrably supported through deterministic tests;
- malformed backend output cannot escape as a valid detection;
- one source or detector failure remains isolated in offline three-source acceptance;
- Hardware Testing can distinguish model-probe success, degraded behavior, failure, and unsupported configuration;
- documentation and traceability distinguish fake-backed deterministic proof from an opt-in real-model probe;
- target selection, tracking, encoders, cross-camera identity, persistence, profiles, retention, and PyQt UI remain unimplemented.

Do not begin Gate 4 until this plan's automated acceptance is green, evidence is current, any real-model probe is labeled accurately, final review has no unresolved Critical or Important findings, and the owner approves the Gate 3 result.
