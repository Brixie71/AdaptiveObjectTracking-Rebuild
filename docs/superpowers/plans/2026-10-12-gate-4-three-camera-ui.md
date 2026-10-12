# Gate 4 Native Three-Camera UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a responsive native PyQt5 desktop shell that displays three independent camera streams, renders Gate 3 detections, exposes start/stop and source health, and hosts the existing camera and detector probes in Settings.

**Architecture:** Add a project-owned runtime layer between the Gate 2 capture manager, the Gate 3 detector, and PyQt. A single background processing worker consumes only newest source-local frames and invokes one detector instance serially; immutable, thread-safe snapshots cross into the GUI, where a timer performs rendering only. Hardware Testing runs through the existing probe services on background Qt tasks and returns sanitized records to a dedicated Settings tab.

**Tech Stack:** Python 3.11, PyQt5 5.15.11, NumPy, OpenCV-backed Gate 2 capture adapters, Gate 3 detector contracts, pytest, and pytest-qt.

**Spec:** `docs/superpowers/specs/2026-10-11-adaptive-object-tracking-architecture-design.md`

## Global Constraints

- Windows 10/11 x64 and Python `==3.11.*` remain the first supported application platform.
- Keep PyQt5 at `5.15.11`; do not introduce PySide, QML, a browser shell, or another GUI framework.
- The UI consumes application state and emits commands. It must not open cameras, invoke detector backends, resolve secrets, or implement computer-vision logic.
- Capture and inference never run on the GUI thread. All inter-thread outputs are project-owned immutable records or copies safe for Qt ownership.
- Preserve bounded newest-frame behavior. No unbounded frame, result, event, or background-task queue may be added.
- Use one detector instance from one processing thread. Do not load one model per camera or assume an Ultralytics backend is thread-safe.
- The main grid has three stable camera slots. One to three enabled sources are supported; unused slots show an explicit unconfigured state, and more than three enabled sources fail before capture starts.
- A source failure, end-of-file condition, or detector error is visible and source-scoped; it must not stop healthy panels or crash the Qt event loop.
- Detection-disabled configuration remains a supported raw-preview mode. Detection-enabled startup must use the verified manifest/model-store and Gate 1 compute selection.
- Hardware Testing uses the existing `probe_sources`, `probe_detector`, and USB discovery boundaries. Raw RTSP URLs, credentials, exception chains, and third-party output never reach widgets, logs, or exported text.
- Gate 4 adds no click selection, tracking, ReID, cross-camera identity, persistence, retained media, profile storage, or upload flow. Successful named-profile persistence remains Gate 10.
- Automated UI acceptance is offline-first and runs with `QT_QPA_PLATFORM=offscreen`. Physical USB/RTSP and real-model checks remain `NOT RUN` unless separately authorized and recorded.

## Review Focus

- Stop or window-close while capture startup, inference, or a hardware probe is blocked must leave the GUI responsive, complete within configured bounds, and never update a destroyed widget; Tasks 2, 3, 5, and 6 pin this behavior.
- Repeated, stale, or out-of-order source sequences must not overwrite newer display state or pair boxes with the wrong frame; Tasks 1 and 2 pin this behavior.
- One failed source or detector call must produce a sanitized degraded state while other sources continue updating; Tasks 2, 3, and 8 pin this behavior.
- NumPy-to-Qt image conversion must preserve BGR-to-RGB color, aspect ratio, and backing-memory lifetime after the source array changes; Task 4 pins this behavior.
- Empty, disabled, over-capacity, and partially configured source sets must produce deterministic controls and status instead of starting an ambiguous session; Tasks 3, 5, and 7 pin this behavior.

---

## Planned Gate 4 File Structure

| Path | Responsibility |
| --- | --- |
| `src/aot/runtime/types.py` | Immutable processing and display snapshots crossing worker/UI boundaries |
| `src/aot/runtime/result_store.py` | Thread-safe newest result per source with monotonic sequence enforcement |
| `src/aot/runtime/processing_worker.py` | Single background round-robin frame/detection worker |
| `src/aot/runtime/live_runtime.py` | Capture, detector, processing, snapshot, and shutdown orchestration |
| `src/aot/ui/rendering.py` | BGR-to-QImage conversion and detection overlay painting |
| `src/aot/ui/camera_panel.py` | One source tile with frame, source name, and sanitized health state |
| `src/aot/ui/background_task.py` | Reusable bounded Qt task wrapper for start/stop and hardware checks |
| `src/aot/ui/controller.py` | Qt signal bridge between `LiveRuntime` and widgets |
| `src/aot/ui/hardware_testing_panel.py` | Settings UI for source discovery/probes and detector probes |
| `src/aot/ui/settings_dialog.py` | Settings dialog and Hardware Testing tab ownership |
| `src/aot/ui/main_window.py` | Three-panel grid, application controls, polling, and close behavior |
| `src/aot/app.py` | Validated application composition and Qt entry point |
| `scripts/run_app.py` | Thin executable wrapper around `aot.app.main` |
| `tests/unit/runtime/` | Store, worker, and runtime lifecycle tests |
| `tests/unit/ui/` | Renderer, panel, controller, window, and Settings tests |
| `tests/integration/test_app.py` | Configuration/composition and safe startup failures |
| `tests/offline/test_gate4_three_camera_ui.py` | Three-source responsive UI acceptance |

### Task 1: Runtime Display Contracts and Newest-Result Store

**Files:**
- Create: `src/aot/runtime/__init__.py`
- Create: `src/aot/runtime/types.py`
- Create: `src/aot/runtime/result_store.py`
- Create: `tests/unit/runtime/__init__.py`
- Create: `tests/unit/runtime/test_types.py`
- Create: `tests/unit/runtime/test_result_store.py`

**Interfaces:**
- Consumes: `FramePacket`, `DetectionBatch`, `SourceState`, and `WorkerSnapshot` from Gates 2 and 3.
- Produces: `ProcessedFrame`, `SourceRuntimeSnapshot`, `LiveRuntimeSnapshot`, and `LatestResultStore` for the processing worker and UI controller.

- [ ] **Step 1: Write the failing immutable-contract tests**

Define tests that construct:

```python
ProcessedFrame(packet: FramePacket, detections: DetectionBatch | None, processing_error: str | None)
SourceRuntimeSnapshot(source_id: str, display_name: str, source_type: SourceType, capture: WorkerSnapshot, result: ProcessedFrame | None)
LiveRuntimeSnapshot(running: bool, sources: tuple[SourceRuntimeSnapshot, ...])
```

Assert frozen assignment fails, source/sequence provenance is preserved, and a detection batch with a different source ID or frame sequence is rejected.

- [ ] **Step 2: Write the failing newest-result tests**

Test `LatestResultStore(source_ids: Iterable[str])`, `put(result: ProcessedFrame) -> bool`, `get(source_id: str) -> ProcessedFrame | None`, and `snapshot() -> dict[str, ProcessedFrame | None]`. Assert unknown sources raise `KeyError`, duplicate/older sequences return `False`, newer sequences replace atomically, and callers cannot mutate the store's mapping.

- [ ] **Step 3: Run the focused tests and verify failure**

Run: `python -m pytest tests/unit/runtime/test_types.py tests/unit/runtime/test_result_store.py -q`

Expected: FAIL because `aot.runtime` does not exist.

- [ ] **Step 4: Implement the contracts and store**

Use frozen dataclasses and one `threading.Lock`. Keep NumPy image ownership inside the existing `FramePacket`; do not clone frames in the store or introduce a queue.

- [ ] **Step 5: Run focused and regression tests**

Run: `python -m pytest tests/unit/runtime/test_types.py tests/unit/runtime/test_result_store.py tests/unit/capture/test_capture_types.py tests/unit/detection/test_types.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/aot/runtime tests/unit/runtime
git commit -m "feat: define Gate 4 runtime display state"
```

### Task 2: Background Frame Processing Worker

**Files:**
- Create: `src/aot/runtime/processing_worker.py`
- Create: `tests/unit/runtime/test_processing_worker.py`
- Modify: `src/aot/runtime/__init__.py`

**Interfaces:**
- Consumes: `packet_supplier(source_id: str) -> FramePacket | None`, optional Gate 3 `Detector`, ordered source IDs, `LatestResultStore`, and an interruptible wait function.
- Produces: `ProcessingWorker.start()`, `request_stop()`, `join(timeout: float | None) -> bool`, `is_alive`, and `snapshot_error(source_id: str) -> str | None`.

- [ ] **Step 1: Write failure-first scheduling tests**

Test that one daemon worker polls sources in stable round-robin order, ignores duplicate sequences, publishes raw `ProcessedFrame` records when the detector is `None`, and never creates an unbounded collection.

- [ ] **Step 2: Write detection and isolation tests**

With a fake detector, assert `detect(packet)` runs on a non-test thread, matching batches are published, and only the newest sequence wins. Make one source inference raise a private-marker exception; assert that source receives only `"Detector inference failed"`, the marker is absent from snapshots, and later sources still update.

- [ ] **Step 3: Write bounded shutdown tests**

Assert `request_stop()` interrupts idle waiting, `join()` is idempotent, and a deliberately blocked detector makes `join(short_timeout)` return `False` without hanging the caller or closing the detector concurrently.

- [ ] **Step 4: Run the focused tests and verify failure**

Run: `python -m pytest tests/unit/runtime/test_processing_worker.py -q`

Expected: FAIL because `ProcessingWorker` is not defined.

- [ ] **Step 5: Implement `ProcessingWorker`**

Use one Python thread and one detector instance. Catch project-owned and unexpected detector failures at the worker boundary, retain only a fixed safe summary, and continue the round-robin loop. Preserve detector cleanup for the owning runtime after the worker has stopped.

- [ ] **Step 6: Run focused and Gate 3 regressions**

Run: `python -m pytest tests/unit/runtime/test_processing_worker.py tests/unit/detection tests/offline/test_gate3_detection_acceptance.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/aot/runtime tests/unit/runtime/test_processing_worker.py
git commit -m "feat: process newest camera frames off the UI thread"
```

### Task 3: Live Runtime Orchestration

**Files:**
- Create: `src/aot/runtime/live_runtime.py`
- Create: `tests/unit/runtime/test_live_runtime.py`
- Modify: `src/aot/runtime/__init__.py`

**Interfaces:**
- Consumes: `AppSettings`, `CaptureManager`, optional `Detector`, and factories injected as `capture_manager_factory(config: CaptureRuntimeConfig) -> CaptureManager` and `detector_factory() -> Detector | None`.
- Produces: `LiveRuntime.start() -> None`, `snapshot() -> LiveRuntimeSnapshot`, `stop() -> None`, `running`, `LiveRuntimeConfigError`, and `LiveRuntimeShutdownError`.

- [ ] **Step 1: Write configuration-boundary tests**

Assert zero enabled sources fails safely, one to three enabled sources preserve configured order and names, four enabled sources fail before either factory is called, and detection-disabled startup never invokes `detector_factory`.

- [ ] **Step 2: Write lifecycle-composition tests**

Assert `start()` constructs a fresh capture manager and optional detector, starts capture before processing, rejects a second start, and `snapshot()` combines every `WorkerSnapshot` with the source-local newest `ProcessedFrame` without exposing internal dictionaries.

- [ ] **Step 3: Write shutdown and restart tests**

Assert `stop()` requests processing shutdown first, always attempts capture shutdown even when processing is stuck, and closes the detector only after its worker has stopped. Repeated stop is safe; a completed stop permits a fresh start with newly constructed dependencies; and a stuck processor or capture worker raises a sanitized `LiveRuntimeShutdownError` naming only affected source IDs after every safe cleanup attempt has run.

- [ ] **Step 4: Run the focused tests and verify failure**

Run: `python -m pytest tests/unit/runtime/test_live_runtime.py -q`

Expected: FAIL because `LiveRuntime` is not defined.

- [ ] **Step 5: Implement `LiveRuntime`**

Treat each start/stop cycle as new ownership. Do not reuse the one-shot Gate 2 `CaptureManager`. On partial startup failure, unwind only successfully created resources in reverse order and suppress third-party exception chains from the public error.

- [ ] **Step 6: Run focused capture/detection regressions**

Run: `python -m pytest tests/unit/runtime tests/integration/test_capture_manager.py tests/integration/test_detection_pipeline.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/aot/runtime tests/unit/runtime/test_live_runtime.py
git commit -m "feat: orchestrate Gate 4 live runtime lifecycle"
```

### Task 4: Frame Rendering and Camera Panel

**Files:**
- Create: `src/aot/ui/__init__.py`
- Create: `src/aot/ui/rendering.py`
- Create: `src/aot/ui/camera_panel.py`
- Create: `tests/unit/ui/__init__.py`
- Create: `tests/unit/ui/test_rendering.py`
- Create: `tests/unit/ui/test_camera_panel.py`

**Interfaces:**
- Consumes: `ProcessedFrame`, `SourceRuntimeSnapshot`, and project-owned detection boxes.
- Produces: `frame_to_qimage(frame: np.ndarray) -> QImage`, `render_processed_frame(result: ProcessedFrame) -> QImage`, and `CameraPanel.apply_snapshot(snapshot: SourceRuntimeSnapshot | None) -> None`.

- [ ] **Step 1: Write failing QImage conversion tests**

Under `QT_QPA_PLATFORM=offscreen`, verify BGR blue/green/red pixels become correct RGB pixels, grayscale or malformed arrays are rejected with a UI-owned error, and the returned `QImage` remains valid after the NumPy source is changed or released.

- [ ] **Step 2: Write failing overlay tests**

Render deterministic detections and assert boxes are clipped to frame bounds, allowed class/confidence labels are present through a testable overlay description, and a batch is never painted over a mismatched source or sequence.

- [ ] **Step 3: Write failing camera-panel tests**

Assert a panel preserves aspect ratio, displays configured name and source ID, maps `OPEN`, `RECONNECTING`, `EOF`, `FAILED`, and unconfigured states to stable text/styles, replaces rather than accumulates pixmaps, and never displays private error text.

- [ ] **Step 4: Run the focused tests and verify failure**

Run: `$env:QT_QPA_PLATFORM='offscreen'; python -m pytest tests/unit/ui/test_rendering.py tests/unit/ui/test_camera_panel.py -q`

Expected: FAIL because `aot.ui` does not exist.

- [ ] **Step 5: Implement rendering and `CameraPanel`**

Convert to RGB and call `QImage.copy()` before leaving the conversion function. Use `QPainter` only on the UI-owned image; do not mutate `FramePacket.frame` or perform model work in paint events.

- [ ] **Step 6: Run focused tests**

Run: `$env:QT_QPA_PLATFORM='offscreen'; python -m pytest tests/unit/ui/test_rendering.py tests/unit/ui/test_camera_panel.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/aot/ui tests/unit/ui
git commit -m "feat: add native camera panel rendering"
```

### Task 5: Asynchronous Qt Runtime Controller and Main Window

**Files:**
- Create: `src/aot/ui/background_task.py`
- Create: `src/aot/ui/controller.py`
- Create: `src/aot/ui/main_window.py`
- Create: `tests/unit/ui/test_background_task.py`
- Create: `tests/unit/ui/test_controller.py`
- Create: `tests/unit/ui/test_main_window.py`
- Modify: `src/aot/ui/__init__.py`

**Interfaces:**
- Consumes: `LiveRuntime.start`, `snapshot`, and `stop` plus three `CameraPanel` widgets.
- Produces: `FunctionTask(QRunnable)`, `RuntimeController(QObject)`, and `MainWindow(QMainWindow)` with Start, Stop, Settings, three camera slots, and a bounded `QTimer` refresh loop.

- [ ] **Step 1: Write failing background-task tests**

Assert a blocking callable runs outside `QApplication.thread()`, emits exactly one success or sanitized failure signal, has no unbounded submission queue, and discards late callbacks after its owner is closed.

- [ ] **Step 2: Write failing controller tests**

Assert `request_start()` and `request_stop()` disable conflicting commands while one operation is active, never call runtime lifecycle methods on the GUI thread, and emit `started`, `stopped`, or fixed safe failure summaries. Repeated clicks must not start duplicate tasks.

- [ ] **Step 3: Write failing main-window tests**

Assert the window creates exactly three stable panels, maps one to three sources in configured order, shows unused slots as unconfigured, enables Start/Stop according to controller state, refreshes from one snapshot per timer tick, and opens Settings without stopping healthy feeds.

- [ ] **Step 4: Write close-during-work tests**

Block fake start/stop work, close the window, and assert the Qt event loop still processes a heartbeat timer, shutdown is requested once, late task results do not access deleted widgets, and no modal exception detail is shown.

- [ ] **Step 5: Run the focused tests and verify failure**

Run: `$env:QT_QPA_PLATFORM='offscreen'; python -m pytest tests/unit/ui/test_background_task.py tests/unit/ui/test_controller.py tests/unit/ui/test_main_window.py -q`

Expected: FAIL because the Qt controller/window classes are absent.

- [ ] **Step 6: Implement the task wrapper, controller, and window**

Use a fixed-size `QThreadPool` for lifecycle commands and a GUI-thread `QTimer` for snapshot presentation. The controller owns asynchronous command state; the window owns widgets only.

- [ ] **Step 7: Run focused tests**

Run: `$env:QT_QPA_PLATFORM='offscreen'; python -m pytest tests/unit/ui/test_background_task.py tests/unit/ui/test_controller.py tests/unit/ui/test_main_window.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/aot/ui tests/unit/ui
git commit -m "feat: add responsive three-camera main window"
```

### Task 6: Settings Hardware Testing Module

**Files:**
- Create: `src/aot/ui/hardware_testing_panel.py`
- Create: `src/aot/ui/settings_dialog.py`
- Create: `tests/unit/ui/test_hardware_testing_panel.py`
- Create: `tests/unit/ui/test_settings_dialog.py`
- Modify: `src/aot/ui/main_window.py`
- Modify: `src/aot/ui/__init__.py`

**Interfaces:**
- Consumes: configured non-sensitive source metadata; injected callables compatible with `discover_usb_devices`, `probe_sources`, and the composed detector-probe service; `SourceTestResult`, `DetectorTestResult`, and `UsbDeviceCandidate`.
- Produces: `HardwareTestingPanel` and `SettingsDialog` with asynchronous Run Source Tests, Discover USB, and Run Detector Test actions plus sanitized result tables.

- [ ] **Step 1: Write failing Settings structure tests**

Assert Settings contains a clearly named Hardware Testing section, lists only non-sensitive source names/IDs/types, exposes the three applicable actions, and does not show RTSP environment values, URLs, credentials, or raw exceptions.

- [ ] **Step 2: Write failing asynchronous probe tests**

Block each injected probe callable and assert a heartbeat timer continues, all Hardware Testing actions are disabled while any one job is active, a second click cannot enqueue a duplicate, and success/degraded/failed/unsupported records render deterministically when the task returns.

- [ ] **Step 3: Write failing lifecycle and stale-result tests**

Close Settings while a probe is active and assert late results are ignored safely. Start a newer test after an older one and assert an older completion cannot overwrite the newer visible result.

- [ ] **Step 4: Write the named-profile boundary test**

Assert no profile is persisted in Gate 4. The panel may explain that successful configurations become saveable named profiles at Gate 10, but it must not write YAML, SQLite, `.env`, or credentials.

- [ ] **Step 5: Run the focused tests and verify failure**

Run: `$env:QT_QPA_PLATFORM='offscreen'; python -m pytest tests/unit/ui/test_hardware_testing_panel.py tests/unit/ui/test_settings_dialog.py -q`

Expected: FAIL because the Settings widgets are absent.

- [ ] **Step 6: Implement Hardware Testing and Settings**

Reuse `FunctionTask`; render only project-owned `to_dict()` fields or explicit safe properties. Keep successful source preview in the main camera grid for Gate 4; do not extend probe result records with retained images.

- [ ] **Step 7: Run focused and existing probe regressions**

Run: `$env:QT_QPA_PLATFORM='offscreen'; python -m pytest tests/unit/ui/test_hardware_testing_panel.py tests/unit/ui/test_settings_dialog.py tests/unit/hardware_testing tests/integration/test_probe_cameras.py tests/integration/test_probe_detector.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/aot/ui tests/unit/ui
git commit -m "feat: expose hardware testing in Settings"
```

### Task 7: Application Composition and Native Entry Point

**Files:**
- Create: `src/aot/app.py`
- Create: `scripts/run_app.py`
- Create: `tests/integration/test_app.py`
- Modify: `src/aot/__init__.py`

**Interfaces:**
- Consumes: `load_settings`, `load_model_manifest`, `probe_compute_device`, `create_detector`, `CaptureManager`, `LiveRuntime`, `MainWindow`, and the existing hardware-test services.
- Produces: `build_runtime(...) -> LiveRuntime`, `create_main_window(...) -> MainWindow`, and `main(argv: Sequence[str] | None = None) -> int`.

- [ ] **Step 1: Write failing composition tests**

Assert the entry point accepts `--app-config`, `--camera-config`, `--env-file`, `--model-manifest`, and `--models-dir`; loads configuration before Qt construction; creates no detector when detection is disabled; and passes the exact `DetectionConfig`, loaded `ModelManifest`, model-store root, and Gate 1 `ComputeDeviceInfo` through `create_detector` when enabled. Artifact resolution remains owned by the Gate 3 factory.

- [ ] **Step 2: Write safe-startup failure tests**

Assert malformed configuration, more than three enabled sources, missing model declarations, and detector-load failures return a nonzero code with a fixed project-owned summary. Private paths, RTSP values, credentials, exception chains, and third-party stdout/stderr must not appear.

- [ ] **Step 3: Write clean-exit tests**

Inject a fake `QApplication`/window and assert normal close returns zero only after runtime stop finishes; shutdown failure returns nonzero and reports a sanitized summary without bypassing resource cleanup.

- [ ] **Step 4: Run the focused tests and verify failure**

Run: `$env:QT_QPA_PLATFORM='offscreen'; python -m pytest tests/integration/test_app.py -q`

Expected: FAIL because `aot.app` does not exist.

- [ ] **Step 5: Implement composition and the thin script**

Keep command parsing and dependency assembly outside widgets. `scripts/run_app.py` imports and exits through `aot.app.main`; it contains no application logic.

- [ ] **Step 6: Run focused and Gate 1 composition regressions**

Run: `$env:QT_QPA_PLATFORM='offscreen'; python -m pytest tests/integration/test_app.py tests/integration/test_check_environment.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/aot/app.py src/aot/__init__.py scripts/run_app.py tests/integration/test_app.py
git commit -m "feat: compose the native Gate 4 application"
```

### Task 8: Offline Three-Camera Acceptance, Documentation, and Evidence

**Files:**
- Create: `tests/offline/test_gate4_three_camera_ui.py`
- Create: `docs/evidence/gate-4-three-camera-ui.md`
- Modify: `README.md`
- Modify: `tests/fixtures/media/README.md`
- Modify: `docs/traceability/MANUSCRIPT_TO_SOURCE_MATRIX.md`

**Interfaces:**
- Consumes: the complete Gate 4 native application boundary, the CC0 three-frame AVI fixture, three independent fake/offline sources, and deterministic fake detector outputs.
- Produces: reproducible UI acceptance evidence and honest AOT-002/AOT-009/AOT-023 traceability updates.

- [ ] **Step 1: Write the failing offline acceptance test**

Launch the real `MainWindow` offscreen with three configured source IDs and an injected `LiveRuntime`. Drive Start, wait for each panel to render source-local frames and allowed detection labels, and prove a GUI heartbeat fires while fake inference is blocked. Release inference and assert all three panels advance without frame/detection provenance mismatch.

- [ ] **Step 2: Add failure isolation and shutdown assertions**

Make one source enter `FAILED` and another reach `EOF`; assert the third panel continues updating, every visible error is sanitized, Stop remains usable, window close leaves no capture/processing/task worker alive, and repeated shutdown is safe.

- [ ] **Step 3: Run Gate 2 through Gate 4 offline acceptance**

Run: `$env:QT_QPA_PLATFORM='offscreen'; python -m pytest tests/offline/test_three_source_capture.py tests/offline/test_gate3_detection_acceptance.py tests/offline/test_gate4_three_camera_ui.py -q`

Expected: PASS.

- [ ] **Step 4: Update README usage and scope**

Document `python scripts/run_app.py`, native PyQt5 controls, one-to-three source behavior, raw-preview mode, model requirements, Settings Hardware Testing, offscreen test commands, and the fact that named-profile saving, selection, tracking, ReID, persistence, and cross-camera identity are later gates.

- [ ] **Step 5: Update fixture provenance and traceability**

Record that Gate 4 reuses the generated AVI only for deterministic display/concurrency proof. Update AOT-002, AOT-009, and AOT-023 only to the level proved by tests; keep physical three-camera, real-model, and end-to-end performance claims `PARTIAL` or `NOT RUN`.

- [ ] **Step 6: Run complete verification**

Run:

```powershell
$env:QT_QPA_PLATFORM='offscreen'
python -m pytest -q
python -m pip check
python scripts/check_environment.py --json
git diff --check
```

Expected: all tests pass with only the existing documented skip, `pip check` reports no broken requirements, the environment report has `"ok": true`, and `git diff --check` is clean.

- [ ] **Step 7: Perform opt-in runtime checks only when authorized**

If a valid detector artifact and physical sources are explicitly authorized, run the existing source/detector probes before launching the UI and record hardware identifiers, model hash, configuration, command, result, and proof boundary. Otherwise record real-model and physical three-camera UI checks as `NOT RUN`; never infer them from offline tests.

- [ ] **Step 8: Write Gate 4 evidence**

Create `docs/evidence/gate-4-three-camera-ui.md` with date, machine, commit, commands, exact test totals, offscreen platform, fixture hash, responsiveness/failure-isolation/shutdown outcomes, and unrun hardware/model boundaries.

- [ ] **Step 9: Commit**

```bash
git add README.md tests/fixtures/media/README.md tests/offline/test_gate4_three_camera_ui.py docs/evidence/gate-4-three-camera-ui.md docs/traceability/MANUSCRIPT_TO_SOURCE_MATRIX.md
git commit -m "test: verify Gate 4 native three-camera UI"
```

## Final Review and Gate 4 Acceptance

- [ ] Request one whole-branch review after all task commits. Resolve every Critical and Important finding with focused regression tests before refreshing evidence.
- [ ] Re-run the complete verification commands on the reviewed commit and record the final commit and counts in Gate 4 evidence.
- [ ] Confirm no generated runtime logs, model weights, camera URLs, credentials, retained images, or Graphify outputs were added to Gate 4 commits.
- [ ] Confirm the native window remains usable with detection disabled, exactly three offline sources, one failed source, one EOF source, and blocked fake inference.
- [ ] Confirm Hardware Testing actions execute outside the GUI thread and expose only sanitized project-owned results.
- [ ] Confirm target selection, tracking, ReID, global identity, persistence, and named-profile saving did not enter this gate.

Gate 4 is ready for owner review only when the automated UI and offline acceptance suite is green, dependency/environment checks pass, evidence is current, the final review has no unresolved Critical or Important findings, and physical/model checks are labeled accurately. Do not begin Gate 5 until the owner approves the Gate 4 result.
