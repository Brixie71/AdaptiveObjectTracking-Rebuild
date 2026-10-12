# Gate 5 Target Selection and Reference Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user click a detection on any live camera panel and deterministically create, display, replace, and disengage an in-memory selected-target reference with an owned crop and traceable metadata.

**Architecture:** A new `aot.selection` package owns frame-space geometry, candidate choice, crop validation, and the current in-memory target. `CameraPanel` emits a click against the exact `ProcessedFrame` it displays, while `MainWindow` coordinates the selection service and a dedicated selected-target widget. Gate 5 creates no tracker, embedding, cross-camera match, database record, retained file, or named profile.

**Tech Stack:** Python 3.11, NumPy, PyQt5 5.15.11, the Gate 3 immutable detection records, the Gate 4 runtime/UI snapshot boundary, pytest, and pytest-qt.

**Spec:** `docs/superpowers/specs/2026-10-11-adaptive-object-tracking-architecture-design.md`

## Global Constraints

- Preserve the approved Gate 0–11 order: Gate 5 adds target selection and reference capture only; per-camera tracking begins at Gate 6, dual encoders at Gate 7, cross-camera identity at Gate 8, and persistence at Gate 10.
- Windows 10/11 x64 and CPython 3.11 remain the initial supported runtime; keep the native PyQt5 UI.
- Selection must use the exact `ProcessedFrame` shown when the click occurs; never combine click geometry, detections, or pixels from different source IDs or frame sequences.
- Continue supporting one to three enabled sources in configured order without allowing one failed source to block the others.
- Keep the selected crop in memory only. Do not write crops, embeddings, targets, events, SQLite data, configuration, or media files.
- Detection-disabled raw preview remains supported, but it has no selectable candidates.
- UI code owns interaction and presentation only; frame-space hit testing, candidate precedence, crop creation, and selected-target state belong to `aot.selection`.
- Errors shown to users must be fixed project-owned summaries without raw exceptions, paths, URLs, credentials, or backend output.
- A successful selection may replace the current in-memory target. An empty or invalid click must leave the current target unchanged. Disengagement explicitly clears it.

## Review Focus

- Aspect-fit previews contain letterbox/pillarbox margins: clicks in a margin produce no request, while content-edge clicks map to the correct source pixel.
- Overlapping boxes require stable precedence: highest confidence wins, then smaller clipped area, then original detection order.
- A mutated or mismatched detection batch, source ID, sequence, capture timestamp, or declared frame size is rejected without changing the active target.
- Partially out-of-frame boxes are clipped; zero-area, inverted, non-finite, malformed-frame, or empty crops are rejected; the accepted crop owns its bytes and is read-only.
- Clicks while stopped, unconfigured, detection-disabled, or without a displayed result do nothing; rapid replacement and repeated disengagement remain safe.

---

## Planned Gate 5 File Structure

```text
src/aot/selection/
|-- __init__.py             Public Gate 5 selection API
|-- geometry.py             Frame clipping and preview-to-frame mapping
|-- types.py                Immutable pixel box and selected-target records
`-- target_selector.py      Candidate selection, crop creation, and current state

src/aot/ui/
|-- camera_panel.py         Emits clicks for the exact displayed frame
|-- main_window.py          Coordinates selection and disengagement
`-- selected_target_panel.py  Selected crop and safe metadata presentation

tests/unit/selection/
|-- __init__.py
|-- test_geometry.py
|-- test_target_selector.py
`-- test_types.py

tests/unit/ui/
|-- test_camera_panel.py
|-- test_main_window.py
`-- test_selected_target_panel.py

tests/integration/test_app.py
tests/offline/test_gate5_target_selection.py
docs/evidence/gate-5-target-selection.md
```

### Task 1: Canonical Selection Types and Geometry

**Files:**
- Create: `src/aot/selection/__init__.py`
- Create: `src/aot/selection/types.py`
- Create: `src/aot/selection/geometry.py`
- Create: `tests/unit/selection/__init__.py`
- Create: `tests/unit/selection/test_types.py`
- Create: `tests/unit/selection/test_geometry.py`
- Modify: `src/aot/ui/rendering.py`
- Modify: `tests/unit/ui/test_rendering.py`

**Interfaces:**
- Consumes: `BoundingBox`, NumPy BGR frame dimensions, and the Gate 4 aspect-fit preview layout.
- Produces: `PixelBox(left: int, top: int, right: int, bottom: int)`, `SelectedTarget`, `clip_box(box: BoundingBox, frame_width: int, frame_height: int) -> PixelBox | None`, and `map_preview_point_to_frame(...) -> tuple[float, float] | None`.

- [ ] **Step 1: Write failing immutable-record and geometry tests**

  Add tests proving:

  - `PixelBox` is half-open, requires `0 <= left < right` and `0 <= top < bottom`, and reports its area.
  - `SelectedTarget` preserves a fixed target ID, source ID, source-local frame sequence, capture monotonic timestamp, UTC selection time, class ID/name, confidence, initial clipped box, and a contiguous copied crop.
  - Mutating the source frame after construction does not change the target crop, and the stored crop rejects item assignment.
  - `clip_box` clips finite partial boxes to `[0, width) x [0, height)`, uses floor for left/top and ceil for right/bottom, and rejects inverted, non-finite, or empty results.
  - `map_preview_point_to_frame(click_x, click_y, preview_width, preview_height, frame_width, frame_height)` handles exact-fit, letterboxed, and pillarboxed previews and returns `None` for margins or the exclusive right/bottom edge.
  - `rendering.describe_overlays()` consumes the shared clipping rule so painted and selectable geometry cannot drift.

- [ ] **Step 2: Run the new tests and verify RED**

  Run:

  ```powershell
  $env:QT_QPA_PLATFORM = "offscreen"
  python -m pytest tests/unit/selection/test_types.py tests/unit/selection/test_geometry.py tests/unit/ui/test_rendering.py -q
  ```

  Expected: FAIL because the selection package and shared geometry do not exist.

- [ ] **Step 3: Implement the canonical records and geometry**

  Define:

  ```python
  @dataclass(frozen=True)
  class PixelBox:
      left: int
      top: int
      right: int
      bottom: int

      @property
      def area(self) -> int: ...

  @dataclass(frozen=True, eq=False)
  class SelectedTarget:
      target_id: str
      source_id: str
      frame_sequence: int
      captured_at_monotonic_ns: int
      selected_at_utc: datetime
      class_id: int
      class_name: str
      confidence: float
      initial_box: PixelBox
      reference_crop: np.ndarray
  ```

  `SelectedTarget.__post_init__` must validate timezone-aware UTC time, scalar metadata, `uint8 HxWx3` crop shape, exact box/crop dimensions, and then install a contiguous owned copy with `writeable=False`. Keep NumPy arrays out of generated equality and repr behavior.

  Define the two pure geometry functions with finite-number and positive-dimension validation. Update `describe_overlays()` to use `clip_box`; keep existing label formatting and QImage ownership behavior.

- [ ] **Step 4: Run the focused tests and verify GREEN**

  Run the Step 2 command.

  Expected: all focused tests pass.

- [ ] **Step 5: Commit Task 1**

  ```powershell
  git add src/aot/selection src/aot/ui/rendering.py tests/unit/selection tests/unit/ui/test_rendering.py
  git commit -m "feat: define Gate 5 selection geometry"
  ```

### Task 2: Deterministic Target Selector and In-Memory State

**Files:**
- Create: `src/aot/selection/target_selector.py`
- Create: `tests/unit/selection/test_target_selector.py`
- Modify: `src/aot/selection/__init__.py`

**Interfaces:**
- Consumes: `ProcessedFrame`, `DetectionBatch`, `PixelBox`, `SelectedTarget`, `clip_box`, an injected `id_factory: Callable[[], str]`, and an injected UTC `clock: Callable[[], datetime]`.
- Produces: `TargetSelector.current -> SelectedTarget | None`, `TargetSelector.select(result: ProcessedFrame, frame_x: float, frame_y: float) -> SelectedTarget | None`, `TargetSelector.disengage() -> SelectedTarget | None`, and sanitized `TargetSelectionError`.

- [ ] **Step 1: Write failing selector tests**

  Cover:

  - A point inside one valid detection creates a target with the injected ID/time, exact source/frame metadata, clipped half-open box, and owned crop.
  - Point containment uses left/top inclusive and right/bottom exclusive boundaries.
  - Overlap precedence is highest confidence, then smallest clipped area, then original tuple order.
  - A valid later selection atomically replaces `current`.
  - An empty click returns `None` and preserves `current`.
  - `disengage()` returns the previous target, clears state, and is idempotent.
  - Missing detections return `None`; malformed frame data or non-finite click coordinates raise only `TargetSelectionError("Target selection unavailable")`.
  - Source, sequence, capture timestamp, declared width/height, and actual frame shape must all agree. Use deliberate post-construction mutation to prove stale/mismatched provenance is rejected without replacing `current`.

- [ ] **Step 2: Run the selector tests and verify RED**

  Run:

  ```powershell
  python -m pytest tests/unit/selection/test_target_selector.py -q
  ```

  Expected: FAIL because `TargetSelector` is not implemented.

- [ ] **Step 3: Implement `TargetSelector`**

  Use a lock to guard current-state replacement and disengagement. Perform validation, hit testing, precedence selection, and crop creation before acquiring the state lock; publish only a completely valid `SelectedTarget`. Default IDs to `str(uuid.uuid4())` and the clock to `datetime.now(timezone.utc)`, while tests inject fixed values.

  Do not catch `BaseException`, expose backend details, retain the full frame, create an embedding, or write the crop to disk.

- [ ] **Step 4: Run Gate 5 selection and upstream detection tests**

  Run:

  ```powershell
  python -m pytest tests/unit/selection tests/unit/detection tests/unit/runtime/test_types.py -q
  ```

  Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

  ```powershell
  git add src/aot/selection tests/unit/selection/test_target_selector.py
  git commit -m "feat: create deterministic target selector"
  ```

### Task 3: Clickable Camera Panels Bound to the Displayed Frame

**Files:**
- Modify: `src/aot/ui/camera_panel.py`
- Modify: `tests/unit/ui/test_camera_panel.py`

**Interfaces:**
- Consumes: `map_preview_point_to_frame`, the exact most recently rendered `ProcessedFrame`, and live-selection enablement from `MainWindow`.
- Produces: `CameraPanel.selection_requested = pyqtSignal(object, float, float)` carrying that displayed `ProcessedFrame` and its frame-space point; `CameraPanel.set_selection_enabled(enabled: bool) -> None`.

- [ ] **Step 1: Write failing camera-panel interaction tests**

  Prove that:

  - A click inside an aspect-fit image emits once with the exact displayed `ProcessedFrame` object and correct frame coordinates.
  - Letterbox and pillarbox margin clicks do not emit.
  - A later snapshot replaces the retained selectable frame; a click cannot emit the earlier frame.
  - Unconfigured state, missing result, malformed preview, and `set_selection_enabled(False)` clear or suppress selection.
  - Resize recomputes mapping without changing frame ownership.

- [ ] **Step 2: Run camera-panel tests and verify RED**

  Run:

  ```powershell
  $env:QT_QPA_PLATFORM = "offscreen"
  python -m pytest tests/unit/ui/test_camera_panel.py -q
  ```

  Expected: FAIL because camera panels do not emit selection requests.

- [ ] **Step 3: Add the preview click boundary**

  Introduce a focused internal clickable-label class or event filter that emits label-local coordinates. Retain the exact displayed `ProcessedFrame` only while it has a valid rendered image; map clicks through the pure geometry function and emit the immutable result plus frame point.

  Do not run selection, detection, cropping, or persistence inside the widget.

- [ ] **Step 4: Run camera-panel and rendering tests**

  Run:

  ```powershell
  $env:QT_QPA_PLATFORM = "offscreen"
  python -m pytest tests/unit/ui/test_camera_panel.py tests/unit/ui/test_rendering.py -q
  ```

  Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

  ```powershell
  git add src/aot/ui/camera_panel.py tests/unit/ui/test_camera_panel.py
  git commit -m "feat: emit frame-bound camera selections"
  ```

### Task 4: Selected-Target Presentation

**Files:**
- Create: `src/aot/ui/selected_target_panel.py`
- Create: `tests/unit/ui/test_selected_target_panel.py`
- Modify: `src/aot/ui/__init__.py`

**Interfaces:**
- Consumes: `SelectedTarget | None` and the existing owned BGR-to-QImage conversion helper.
- Produces: `SelectedTargetPanel.apply_target(target: SelectedTarget | None) -> None`, public safe labels/crop preview for assertions, a `disengage_requested` signal, and a Disable / Disengage Target button.

- [ ] **Step 1: Write failing selected-target widget tests**

  Assert empty-state text and disabled disengage control, then apply a target and assert copied crop display, class/confidence, source ID, frame sequence, UTC selection time, and clipped box text. Assert the raw crop bytes, UUID internals beyond the safe target ID, NumPy repr, and private error content never appear in labels. Confirm the button emits once and applying `None` clears its prior pixmap and metadata.

- [ ] **Step 2: Run the widget tests and verify RED**

  Run:

  ```powershell
  $env:QT_QPA_PLATFORM = "offscreen"
  python -m pytest tests/unit/ui/test_selected_target_panel.py -q
  ```

  Expected: FAIL because the widget does not exist.

- [ ] **Step 3: Implement `SelectedTargetPanel`**

  Keep this widget presentation-only. Use the shared image conversion for the crop, fixed field labels, aspect-fit preview scaling, and explicit empty state. The button emits intent but does not clear application state itself.

- [ ] **Step 4: Run all focused UI presentation tests**

  Run:

  ```powershell
  $env:QT_QPA_PLATFORM = "offscreen"
  python -m pytest tests/unit/ui/test_selected_target_panel.py tests/unit/ui/test_camera_panel.py tests/unit/ui/test_rendering.py -q
  ```

  Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

  ```powershell
  git add src/aot/ui/selected_target_panel.py src/aot/ui/__init__.py tests/unit/ui/test_selected_target_panel.py
  git commit -m "feat: present selected target references"
  ```

### Task 5: Main-Window Selection and Disengagement Flow

**Files:**
- Modify: `src/aot/ui/main_window.py`
- Modify: `tests/unit/ui/test_main_window.py`

**Interfaces:**
- Consumes: injected `TargetSelector`, `CameraPanel.selection_requested`, runtime running state, and `SelectedTargetPanel`.
- Produces: one application-level selected-target surface, safe selection status, valid-target replacement, and explicit disengagement without changing live runtime ownership.

- [ ] **Step 1: Write failing main-window flow tests**

  Add tests proving:

  - The window contains exactly one selected-target panel and connects all three camera-panel selection signals to the same selector.
  - Selection is enabled only while the runtime snapshot reports running; raw preview or no detections cannot create a target.
  - A valid click updates selector state and presentation with `"Target selected"`.
  - A click without a candidate leaves the existing target unchanged and shows `"No selectable target at this point"`.
  - `TargetSelectionError` shows `"Target selection unavailable"` without private details and preserves the current target.
  - A second valid camera click replaces the target deterministically.
  - Disable / Disengage Target clears selector and presentation, shows `"Target disengaged"`, and is safe when repeated.
  - Start, Stop, Settings, refresh, source-failure isolation, and close behavior remain unchanged.

- [ ] **Step 2: Run main-window tests and verify RED**

  Run:

  ```powershell
  $env:QT_QPA_PLATFORM = "offscreen"
  python -m pytest tests/unit/ui/test_main_window.py -q
  ```

  Expected: FAIL because the window has no selection coordinator or target surface.

- [ ] **Step 3: Integrate selection into `MainWindow`**

  Add `target_selector: TargetSelector` as an injected keyword-only dependency. Place `SelectedTargetPanel` beside or below the responsive camera grid without changing the three stable slots. During each `refresh_once()`, set panel selection enablement from the same `LiveRuntimeSnapshot.running` value used for that refresh.

  Catch only `TargetSelectionError` at the UI boundary. Do not clear the selected target when Stop is clicked; selection remains in memory until replacement, disengagement, or application exit.

- [ ] **Step 4: Run the complete UI unit suite**

  Run:

  ```powershell
  $env:QT_QPA_PLATFORM = "offscreen"
  python -m pytest tests/unit/ui tests/unit/selection -q
  ```

  Expected: all tests pass.

- [ ] **Step 5: Commit Task 5**

  ```powershell
  git add src/aot/ui/main_window.py tests/unit/ui/test_main_window.py
  git commit -m "feat: coordinate live target selection"
  ```

### Task 6: Application Composition and Compatibility

**Files:**
- Modify: `src/aot/app.py`
- Modify: `tests/integration/test_app.py`

**Interfaces:**
- Consumes: `TargetSelector` and the existing runtime/settings/probe composition.
- Produces: one fresh selector per native application window, with injectable selector construction for deterministic tests.

- [ ] **Step 1: Write failing composition tests**

  Verify `create_main_window()` injects one supplied selector into the window, `main()` creates no selector before configuration/runtime validation succeeds, and composition failures retain existing sanitized process messages and runtime cleanup.

- [ ] **Step 2: Run application integration tests and verify RED**

  Run:

  ```powershell
  $env:QT_QPA_PLATFORM = "offscreen"
  python -m pytest tests/integration/test_app.py -q
  ```

  Expected: FAIL because the composition root does not build the selection dependency.

- [ ] **Step 3: Compose the target selector**

  Extend `create_main_window(..., target_selector_factory=TargetSelector)` and construct exactly one selector after settings/runtime validation. Pass it into `MainWindow`; keep CLI arguments and configuration schemas unchanged.

- [ ] **Step 4: Run composition, runtime, and UI suites**

  Run:

  ```powershell
  $env:QT_QPA_PLATFORM = "offscreen"
  python -m pytest tests/integration/test_app.py tests/unit/selection tests/unit/ui tests/unit/runtime -q
  ```

  Expected: all tests pass.

- [ ] **Step 5: Commit Task 6**

  ```powershell
  git add src/aot/app.py tests/integration/test_app.py
  git commit -m "feat: compose Gate 5 target selection"
  ```

### Task 7: Offline Gate 5 Acceptance, Documentation, and Evidence

**Files:**
- Create: `tests/offline/test_gate5_target_selection.py`
- Modify: `README.md`
- Modify: `docs/traceability/MANUSCRIPT_TO_SOURCE_MATRIX.md`
- Create: `docs/evidence/gate-5-target-selection.md`

**Interfaces:**
- Consumes: the complete Gate 5 application boundary, the existing CC0 AVI, three deterministic source-local detections including overlap and edge clipping, and the real native widgets offscreen.
- Produces: repeatable proof of selection from each panel, exact crop/metadata creation, replacement, empty-click preservation, disengagement, and Gate 2–5 regression safety.

- [ ] **Step 1: Write the failing offline acceptance test**

  Launch the real `MainWindow`, `LiveRuntime`, selector, processing worker, and three offline sources. Use deterministic fake detector output to:

  - select once from Camera A, B, and C through real widget clicks;
  - verify source/frame/class/confidence/box metadata and crop pixels match the displayed frame;
  - prove an overlapping click follows the documented precedence;
  - prove margin and empty clicks preserve the target;
  - replace the target across cameras and disengage it;
  - keep a Qt heartbeat active and shut down without retained workers or detector ownership.

- [ ] **Step 2: Run Gate 2–5 offline acceptance and verify RED for Gate 5 only**

  Run:

  ```powershell
  $env:QT_QPA_PLATFORM = "offscreen"
  python -m pytest tests/offline/test_three_source_capture.py tests/offline/test_gate3_detection_acceptance.py tests/offline/test_gate4_three_camera_ui.py tests/offline/test_gate5_target_selection.py -q
  ```

  Expected: prior gates pass and Gate 5 fails until the acceptance flow is complete.

- [ ] **Step 3: Complete only the minimum acceptance integration fixes**

  Keep changes within selection/UI test seams. Do not add tracking IDs, encoders, ReID matching, file retention, uploads, target naming, save/load, SQLite, or named profiles.

- [ ] **Step 4: Update user and traceability documentation**

  Document the selection interaction, overlap precedence, in-memory-only lifetime, explicit disengagement, test command, and later-gate exclusions. Update AOT-010, AOT-011, and AOT-024 only to the level proved by the automated acceptance. Preserve manuscript-reported `28/30` values as historical claims, not rebuild results.

- [ ] **Step 5: Run complete verification**

  Run:

  ```powershell
  $env:QT_QPA_PLATFORM = "offscreen"
  python -m pytest -q
  python -m pip check
  python scripts/check_environment.py --json
  git diff --check
  ```

  Expected: all tests pass with only the existing documented skip, `pip check` reports no broken requirements, environment JSON contains `"ok": true`, and the diff check is clean.

- [ ] **Step 6: Write Gate 5 evidence**

  Record the reviewed commit, exact command results, fixture provenance/hash, tested selection semantics, and these explicit proof boundaries:

  - physical camera selection: `NOT RUN` unless separately authorized and performed;
  - real-model selection: `NOT RUN` unless a verified model artifact is provided and tested;
  - tracking, encoders, ReID, cross-camera identity, persistence, retained media, uploads, and named profiles: not implemented in Gate 5.

- [ ] **Step 7: Commit Task 7**

  ```powershell
  git add README.md docs/evidence/gate-5-target-selection.md docs/traceability/MANUSCRIPT_TO_SOURCE_MATRIX.md tests/offline/test_gate5_target_selection.py
  git commit -m "test: verify Gate 5 target selection"
  ```

## Final Review and Gate 5 Acceptance

- [ ] Request one whole-branch review against this plan and the approved architecture specification after every task commit.
- [ ] Resolve all Critical and Important findings with focused regression tests before refreshing evidence.
- [ ] Re-run complete verification on the reviewed commit and record the final commit and counts in Gate 5 evidence.
- [ ] Confirm Gate 5 commits contain no generated Graphify output, camera credentials/URLs, model weights, runtime logs, retained crops, uploads, database files, or `.env` data.
- [ ] Keep physical-camera and real-model checks accurately labeled `NOT RUN` unless separately authorized and executed.
- [ ] Do not begin Gate 6 until the owner approves the Gate 5 result.

Gate 5 is ready for owner review only when clicking any of the three native panels deterministically creates the correct owned crop and metadata from that exact displayed frame, empty/invalid clicks cannot corrupt current state, disengagement is safe, the full automated suite is green, evidence is current, and final review has no unresolved Critical or Important findings.
