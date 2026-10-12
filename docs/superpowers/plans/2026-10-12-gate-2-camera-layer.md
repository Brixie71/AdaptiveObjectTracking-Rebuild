# Gate 2 Camera Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a common, secret-safe camera layer for USB, RTSP, and offline-video sources, including bounded capture workers and the source-testing backend for the future Settings > Hardware Testing module.

**Architecture:** Project-owned capture contracts isolate OpenCV behind injectable backends. Each enabled source runs in an independent worker that normalizes frames into a canonical packet and publishes only recent frames through a bounded drop-oldest buffer. A source probe consumes the same contracts to produce sanitized Hardware Testing results; UI and profile persistence remain later-gate consumers.

**Tech Stack:** Python 3.11, OpenCV 4 (`cv2`), NumPy, immutable dataclasses, standard-library protocols/threading/queues/timing, PyYAML configuration, pytest.

**Spec:** `docs/superpowers/specs/2026-10-11-adaptive-object-tracking-architecture-design.md`

## Global Constraints

- Windows 10 or 11 x64 and Python `3.11.*` remain the first acceptance platform.
- Support USB, RTSP, and offline video through one capture interface.
- Historical files under `docs/reference/` and the `ThesisV2` repository remain read-only evidence and are never runtime imports.
- RTSP URLs are resolved only at connection time from environment-variable keys; credentials never appear in configuration, public errors, logs, reports, tests, or Git.
- Each source owns its capture handle, worker, state, and bounded newest-frame buffer; one failed or slow source cannot block another.
- Canonical output is a contiguous NumPy `uint8` BGR frame with the configured rotation applied and final dimensions equal to the configured width and height.
- Queue overflow discards the oldest unprocessed frame and increments a dropped-frame counter.
- USB and RTSP reconnect with bounded exponential backoff; offline end-of-file is a normal terminal state and never reconnects.
- Gate 2 adds no detector, tracker, ReID, identity, persistence, or PyQt implementation.
- Hardware Testing begins with reusable source probes and sanitized results. The Settings UI arrives with the UI gate; saving named profiles remains Gate 10.
- Automated tests must not require cameras, network access, a GUI, model weights, or historical runtime code.
- Real hardware tests are opt-in and remain separate from the default suite.

## Review Focus

- A backend exception containing a credential-bearing RTSP URL must become a generic sanitized capture error; Task 4 adds this regression.
- A malformed frame (empty, unsupported rank/channel count, or non-`uint8`) must fail predictably rather than reaching later vision code; Task 2 adds these regressions.
- A full buffer under a fast producer and slow consumer must remain at its configured capacity and return the newest frame; Task 5 adds this regression.
- One worker repeatedly failing and reconnecting must not delay frames or shutdown for another source; Task 5 adds this regression.
- `close()` during open, read, backoff, EOF, or partial startup must release every acquired handle exactly once and leave no live worker; Tasks 3 and 5 add these regressions.

---

### Task 1: Capture Policies and Domain Contracts

**Files:**
- Create: `src/aot/capture/__init__.py`
- Create: `src/aot/capture/types.py`
- Modify: `src/aot/config/types.py`
- Modify: `src/aot/config/loader.py`
- Modify: `config/app.example.yaml`
- Test: `tests/unit/config/test_loader.py`
- Test: `tests/unit/capture/test_types.py`

**Interfaces:**
- Consumes: `CameraSourceConfig`, `SourceType`, and the strict Gate 1 configuration boundary.
- Produces: `CaptureRuntimeConfig`, `SourceState`, `CaptureProperties`, `FramePacket`, `CaptureError`, `CaptureOpenError`, `CaptureReadError`, `EndOfStream`, and `FrameSource`.
- `CaptureProperties` fields are `requested_width: int`, `requested_height: int`, `requested_fps: float`, `negotiated_width: int`, `negotiated_height: int`, and `negotiated_fps: float`.
- `FramePacket` fields are `source_id: str`, `sequence: int`, `captured_at_monotonic_ns: int`, and `frame: np.ndarray`.
- `FrameSource` exposes `source_id: str`, `source_type: SourceType`, `reconnectable: bool`, `state: SourceState`, `properties: CaptureProperties | None`, `open() -> CaptureProperties`, `read() -> FramePacket`, and `close() -> None`.

- [ ] **Step 1: Write failing configuration tests for the capture policy**

Add tests asserting these defaults when `capture` is absent:

```python
assert settings.capture.queue_capacity == 2
assert settings.capture.reconnect_initial_seconds == 0.25
assert settings.capture.reconnect_max_seconds == 5.0
assert settings.capture.shutdown_timeout_seconds == 5.0
```

Add strict-boundary tests for unknown capture keys; boolean, zero, negative, non-finite, and wrong-type values; `reconnect_max_seconds < reconnect_initial_seconds`; and `queue_capacity` values other than positive integers.

- [ ] **Step 2: Run the capture-policy tests to verify RED**

Run: `python -m pytest tests/unit/config/test_loader.py -k capture -v`

Expected: FAIL because `CaptureRuntimeConfig` and `AppSettings.capture` do not exist.

- [ ] **Step 3: Add `CaptureRuntimeConfig` and load it strictly**

In `src/aot/config/types.py`, add the frozen dataclass:

```python
@dataclass(frozen=True)
class CaptureRuntimeConfig:
    queue_capacity: int = 2
    reconnect_initial_seconds: float = 0.25
    reconnect_max_seconds: float = 5.0
    shutdown_timeout_seconds: float = 5.0
```

Add `capture: CaptureRuntimeConfig` to `AppSettings`. In `loader.py`, add `_load_capture(value: Any, path: Path) -> CaptureRuntimeConfig`, reuse finite-number validation, reject booleans, and add `capture` to the allowed application keys. Document the defaults in `config/app.example.yaml`.

- [ ] **Step 4: Run the configuration tests to verify GREEN**

Run: `python -m pytest tests/unit/config/test_loader.py tests/unit/config/test_types.py -q`

Expected: PASS.

- [ ] **Step 5: Write failing contract tests**

In `tests/unit/capture/test_types.py`, assert:

- all records are frozen;
- `SourceState` contains `CLOSED`, `OPENING`, `OPEN`, `RECONNECTING`, `EOF`, and `FAILED`;
- `CaptureProperties` records requested and negotiated width, height, and FPS;
- `FramePacket` records `source_id`, zero-based `sequence`, `captured_at_monotonic_ns`, and the normalized frame;
- `CaptureError` string output contains only the supplied sanitized summary;
- a minimal fake satisfies the runtime-checkable `FrameSource` protocol.

- [ ] **Step 6: Run the contract tests to verify RED**

Run: `python -m pytest tests/unit/capture/test_types.py -v`

Expected: FAIL because `aot.capture.types` does not exist.

- [ ] **Step 7: Implement the capture contracts**

Use frozen dataclasses and a `@runtime_checkable Protocol`. `CaptureError` subclasses carry `source_id` and a sanitized summary but never accept or retain a raw connection string. `EndOfStream` is distinct from `CaptureReadError`.

- [ ] **Step 8: Run Task 1 tests and the full suite**

Run: `python -m pytest tests/unit/config tests/unit/capture/test_types.py -q`

Then: `python -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 9: Commit Task 1**

```powershell
git add config/app.example.yaml src/aot/config src/aot/capture tests/unit/config tests/unit/capture/test_types.py
git commit -m "feat: define Gate 2 capture contracts"
```

---

### Task 2: Canonical Frame Normalization

**Files:**
- Create: `src/aot/capture/normalize.py`
- Create: `tests/unit/capture/test_normalize.py`

**Interfaces:**
- Consumes: raw `numpy.ndarray` frames and `CameraSourceConfig.width`, `.height`, and `.rotation`.
- Produces: `normalize_frame(frame: np.ndarray, *, width: int, height: int, rotation: int) -> np.ndarray` and `FrameNormalizationError(ValueError)`.

- [ ] **Step 1: Write failing normalization tests**

Cover BGR pass-through, grayscale-to-BGR, BGRA-to-BGR, rotations `0/90/180/270`, resize after rotation, contiguous output, and final `shape == (height, width, 3)` with `dtype == np.uint8`. Reject `None`, empty arrays, unsupported ranks/channel counts, non-`uint8` data, invalid dimensions, and invalid rotations.

- [ ] **Step 2: Run normalization tests to verify RED**

Run: `python -m pytest tests/unit/capture/test_normalize.py -v`

Expected: FAIL because the normalization module does not exist.

- [ ] **Step 3: Implement `normalize_frame`**

Use OpenCV color conversion, `cv2.rotate`, and `cv2.resize`. Apply color conversion first, rotation second, and final resize third. Return `np.ascontiguousarray(result)` and translate OpenCV errors into `FrameNormalizationError` without embedding frame contents.

- [ ] **Step 4: Run Task 2 tests and the full suite**

Run: `python -m pytest tests/unit/capture/test_normalize.py -q`

Then: `python -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/aot/capture/normalize.py tests/unit/capture/test_normalize.py
git commit -m "feat: normalize captured frames"
```

---

### Task 3: OpenCV Backend, USB Adapter, and Offline Adapter

**Files:**
- Create: `src/aot/capture/backend.py`
- Create: `src/aot/capture/base.py`
- Create: `src/aot/capture/usb.py`
- Create: `src/aot/capture/offline.py`
- Modify: `src/aot/capture/__init__.py`
- Create: `tests/unit/capture/fakes.py`
- Create: `tests/unit/capture/test_usb.py`
- Create: `tests/unit/capture/test_offline.py`

**Interfaces:**
- Consumes: Task 1 contracts and Task 2 `normalize_frame`.
- Produces: `CaptureBackend` protocol, `OpenCvCaptureBackend`, `BackendFactory = Callable[[int | str], CaptureBackend]`, `UsbFrameSource`, and `OfflineFrameSource`.
- Constructors: `UsbFrameSource(config: CameraSourceConfig, backend_factory: BackendFactory = open_cv_backend, clock_ns: Callable[[], int] = time.monotonic_ns)` and `OfflineFrameSource(config: CameraSourceConfig, backend_factory: BackendFactory = open_cv_backend, clock_ns: Callable[[], int] = time.monotonic_ns)`.

- [ ] **Step 1: Create the fake backend and write failing USB tests**

The fake records its input, property-set calls, reads, and release count without importing hardware. Tests require:

- only USB configuration is accepted;
- the configured device index is passed to the backend;
- requested width, height, and FPS are applied before negotiated properties are read;
- `open()` fails generically when the backend is unavailable;
- successful reads return normalized, sequenced `FramePacket` values with injected monotonic timestamps;
- empty/failed reads raise sanitized `CaptureReadError`;
- repeated `close()` is idempotent and releases an acquired handle exactly once.

- [ ] **Step 2: Run USB tests to verify RED**

Run: `python -m pytest tests/unit/capture/test_usb.py -v`

Expected: FAIL because the backend and USB adapter do not exist.

- [ ] **Step 3: Implement the backend boundary and USB adapter**

`OpenCvCaptureBackend` is the only Gate 2 file that directly wraps `cv2.VideoCapture` and `CAP_PROP_*`. The project adapter must expose only project-owned properties/errors. Inject `clock_ns: Callable[[], int] = time.monotonic_ns` into the shared base so tests do not patch global time.

- [ ] **Step 4: Run USB tests to verify GREEN**

Run: `python -m pytest tests/unit/capture/test_usb.py -q`

Expected: PASS.

- [ ] **Step 5: Write failing offline-adapter tests**

Require a resolved absolute file path, reject missing/non-file paths before opening, use the file path as the backend input, normalize frames identically to USB, and translate a failed read after successful frames into `EndOfStream` with state `EOF`. Assert EOF never marks the source reconnectable and `close()` releases once from both open and EOF states.

- [ ] **Step 6: Run offline tests to verify RED**

Run: `python -m pytest tests/unit/capture/test_offline.py -v`

Expected: FAIL because `OfflineFrameSource` does not exist.

- [ ] **Step 7: Implement the offline adapter**

Share lifecycle and packet construction through `base.py`, but keep offline EOF semantics in `offline.py`. Do not add looping playback in Gate 2.

- [ ] **Step 8: Run Task 3 tests and the full suite**

Run: `python -m pytest tests/unit/capture/test_usb.py tests/unit/capture/test_offline.py -q`

Then: `python -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 9: Commit Task 3**

```powershell
git add src/aot/capture tests/unit/capture
git commit -m "feat: add USB and offline capture adapters"
```

---

### Task 4: Secret-Safe RTSP Adapter and Source Factory

**Files:**
- Create: `src/aot/capture/rtsp.py`
- Create: `src/aot/capture/factory.py`
- Modify: `src/aot/capture/__init__.py`
- Create: `tests/unit/capture/test_rtsp.py`
- Create: `tests/unit/capture/test_factory.py`

**Interfaces:**
- Consumes: Task 1 contracts, Task 3 backend boundary, and `resolve_rtsp_url(source, environment)`.
- Produces: `RtspFrameSource` and `create_frame_source(config: CameraSourceConfig, *, environment: Mapping[str, str] | None = None, backend_factory: BackendFactory = open_cv_backend, clock_ns: Callable[[], int] = time.monotonic_ns) -> FrameSource`.

- [ ] **Step 1: Write failing RTSP security and lifecycle tests**

Construct synthetic credentials from string fragments so repository scans do not contain URL credentials. Tests require:

- the environment key, not a URL, is retained on the source;
- the URL is resolved immediately before each `open()` attempt;
- the resolved URL is passed to the backend but is absent from `repr(source)`, `vars(source)`, public exceptions, and captured logs;
- a backend exception that includes the full credential-bearing URL becomes `CaptureOpenError("RTSP source could not be opened")`;
- RTSP is reconnectable, reads normalized packets, and releases exactly once per acquired backend.

- [ ] **Step 2: Run RTSP tests to verify RED**

Run: `python -m pytest tests/unit/capture/test_rtsp.py -v`

Expected: FAIL because the RTSP adapter does not exist.

- [ ] **Step 3: Implement `RtspFrameSource`**

Resolve the secret inside `open()`, pass it directly to `backend_factory`, and do not assign it to a source attribute. Catch backend errors and emit a fixed sanitized summary. Clear and release the backend on failed open/read paths.

- [ ] **Step 4: Run RTSP tests to verify GREEN**

Run: `python -m pytest tests/unit/capture/test_rtsp.py -q`

Expected: PASS.

- [ ] **Step 5: Write failing factory tests**

Parametrize all `SourceType` values and assert the exact adapter type, propagated dependency injections, and configuration rejection when the source-specific field is absent or inconsistent. Include an unknown future enum-like value and require a project-owned `CaptureError`, not a raw lookup error.

- [ ] **Step 6: Run factory tests to verify RED**

Run: `python -m pytest tests/unit/capture/test_factory.py -v`

Expected: FAIL because `create_frame_source` does not exist.

- [ ] **Step 7: Implement and export the source factory**

Keep source selection in `factory.py`; consumers must not branch on USB/RTSP/offline themselves.

- [ ] **Step 8: Run Task 4 tests, credential scan, and the full suite**

Run: `python -m pytest tests/unit/capture/test_rtsp.py tests/unit/capture/test_factory.py -q`

Run:

```powershell
git grep -n -I -E "rtsp://[^[:space:]]+@|password[[:space:]]*=|token[[:space:]]*=" -- . ":(exclude)docs/reference/**" ":(exclude)docs/superpowers/**" ":(exclude)REBUILD_PREREQUISITES.md" ":(exclude)tests/**"
```

Expected: no live credential match.

Then: `python -m pytest -q`

Expected: all tests PASS.

- [ ] **Step 9: Commit Task 4**

```powershell
git add src/aot/capture tests/unit/capture
git commit -m "feat: add secret-safe RTSP capture"
```

---

### Task 5: Bounded Frame Flow, Reconnection, and Independent Shutdown

**Files:**
- Create: `src/aot/capture/buffer.py`
- Create: `src/aot/capture/worker.py`
- Create: `src/aot/capture/manager.py`
- Modify: `src/aot/capture/__init__.py`
- Create: `tests/unit/capture/test_buffer.py`
- Create: `tests/unit/capture/test_worker.py`
- Create: `tests/integration/test_capture_manager.py`

**Interfaces:**
- Consumes: `FrameSource`, `FramePacket`, `CaptureRuntimeConfig`, and the source factory.
- Produces: `NewestFrameBuffer`, `ReconnectPolicy`, `CaptureWorker`, `CaptureManager`, `WorkerSnapshot`, and `ManagerShutdownError`.
- `NewestFrameBuffer(capacity: int)` exposes `put(packet) -> int`, `get_latest() -> FramePacket | None`, `size: int`, `dropped_count: int`, and `clear() -> None`; `put` returns the number dropped by that call.
- `CaptureManager(runtime_config: CaptureRuntimeConfig, source_factory: Callable[[CameraSourceConfig], FrameSource] = create_frame_source)` owns worker construction.
- `CaptureManager.start(sources: Iterable[CameraSourceConfig]) -> None`, `snapshot() -> dict[str, WorkerSnapshot]`, `latest(source_id: str) -> FramePacket | None`, and `stop() -> None`.

- [ ] **Step 1: Write failing newest-frame buffer tests**

Cover positive capacity validation, FIFO behavior before overflow, drop-oldest behavior at capacity, `get_latest()` discarding older queued frames, exact dropped counts, clear, and concurrent producer/consumer access without exceeding capacity.

- [ ] **Step 2: Run buffer tests to verify RED**

Run: `python -m pytest tests/unit/capture/test_buffer.py -v`

Expected: FAIL because `NewestFrameBuffer` does not exist.

- [ ] **Step 3: Implement the bounded buffer**

Use `collections.deque` plus one lock; do not use an unbounded `queue.Queue`. `get_latest()` returns the newest packet and removes all packets currently buffered.

- [ ] **Step 4: Run buffer tests to verify GREEN**

Run: `python -m pytest tests/unit/capture/test_buffer.py -q`

Expected: PASS.

- [ ] **Step 5: Write failing worker and reconnection tests**

Using scripted fake sources and injected `wait_for_stop(seconds) -> bool`, verify:

- the worker opens, reads, publishes, and closes in its own thread;
- delay sequence starts at `0.25`, doubles, caps at `5.0`, and resets after a successful frame;
- USB and RTSP retry after open/read failure while offline transitions to `EOF` without retry;
- backend/source errors never terminate the process or leak secrets into `WorkerSnapshot`;
- stopping during open, read, or backoff exits within `shutdown_timeout_seconds` and closes once;
- queue depth and dropped-frame counts appear in the snapshot.

- [ ] **Step 6: Run worker tests to verify RED**

Run: `python -m pytest tests/unit/capture/test_worker.py -v`

Expected: FAIL because the worker and reconnect policy do not exist.

- [ ] **Step 7: Implement `ReconnectPolicy` and `CaptureWorker`**

The worker owns one source and buffer. Use a `threading.Event` for interruptible backoff and shutdown. Never hold the buffer lock while opening, reading, sleeping, or closing a source.

- [ ] **Step 8: Run worker tests to verify GREEN**

Run: `python -m pytest tests/unit/capture/test_worker.py -q`

Expected: PASS.

- [ ] **Step 9: Write failing manager integration tests**

Create three fake enabled sources. Make one continuously succeed, one fail/reconnect, and one reach EOF. Assert successful frames continue while the failing worker backs off, states remain per-source, `latest()` is source-specific, disabled configurations are not started, duplicate starts are rejected, and repeated `stop()` joins every worker and closes every source.

- [ ] **Step 10: Run manager tests to verify RED**

Run: `python -m pytest tests/integration/test_capture_manager.py -v`

Expected: FAIL because `CaptureManager` does not exist.

- [ ] **Step 11: Implement `CaptureManager`**

Create one source, worker, and buffer per enabled configuration. If any worker remains alive after the configured shutdown timeout, finish closing every other worker and then raise `ManagerShutdownError` naming only source IDs, never source values.

- [ ] **Step 12: Run Task 5 tests and the full suite**

Run: `python -m pytest tests/unit/capture/test_buffer.py tests/unit/capture/test_worker.py tests/integration/test_capture_manager.py -q`

Then: `python -m pytest -q`

Expected: all tests PASS without hangs or leaked non-daemon threads.

- [ ] **Step 13: Commit Task 5**

```powershell
git add src/aot/capture tests/unit/capture tests/integration/test_capture_manager.py
git commit -m "feat: add bounded multi-source capture runtime"
```

---

### Task 6: Hardware Testing Source Probe and CLI

**Files:**
- Create: `src/aot/hardware_testing/__init__.py`
- Create: `src/aot/hardware_testing/types.py`
- Create: `src/aot/hardware_testing/source_probe.py`
- Create: `src/aot/hardware_testing/usb_discovery.py`
- Create: `scripts/probe_cameras.py`
- Create: `tests/unit/hardware_testing/test_source_probe.py`
- Create: `tests/integration/test_probe_cameras.py`

**Interfaces:**
- Consumes: `FrameSource`, `CaptureProperties`, the source factory, configuration loader, and final-output redaction.
- Produces: `HardwareTestStatus` (`UNSUPPORTED`, `FAILED`, `DEGRADED`, `SUCCESS`), `SourceTestResult`, `UsbDeviceCandidate`, `discover_usb_devices(*, max_index: int = 10, backend_factory: BackendFactory = open_cv_backend) -> tuple[UsbDeviceCandidate, ...]`, `probe_source(source: FrameSource, *, frame_count: int = 30, clock_ns: Callable[[], int] = time.perf_counter_ns) -> SourceTestResult`, and `probe_sources(...) -> tuple[SourceTestResult, ...]`.
- CLI options: `--app-config`, `--camera-config`, `--env-file`, repeatable `--source-id`, `--frames`, `--discover-usb`, `--max-usb-index`, and `--json`.

- [ ] **Step 1: Write failing source-probe tests**

Assert:

- `frame_count` must be a positive integer;
- complete reads are `SUCCESS`;
- partial reads followed by a recoverable capture failure are `DEGRADED`;
- zero frames, open failure, or unexpected EOF before any frame are `FAILED`;
- unsupported source construction is `UNSUPPORTED`;
- results include source ID/type, requested and negotiated properties, frames requested/received, mean and p95 capture latency, dropped frames, start/end timestamps, and a sanitized summary;
- result serialization cannot contain a supplied RTSP URL, username/password, or query token;
- the source is closed in every success and failure path.

Add discovery tests that validate nonnegative `max_index`, probe indices `0` through `max_index - 1`, include only successfully opened devices, report only index/non-sensitive negotiated properties, and release every attempted backend exactly once.

- [ ] **Step 2: Run probe tests to verify RED**

Run: `python -m pytest tests/unit/hardware_testing/test_source_probe.py -v`

Expected: FAIL because the Hardware Testing package does not exist.

- [ ] **Step 3: Implement source-test result types and probing**

Use monotonic time for latency and UTC for display timestamps. Calculate p95 by the nearest-rank method and document it in the result type. Do not include connection inputs or exception representations in results. Implement USB discovery in `usb_discovery.py` through the injectable backend boundary; discovery must never retain an opened handle.

- [ ] **Step 4: Run probe tests to verify GREEN**

Run: `python -m pytest tests/unit/hardware_testing/test_source_probe.py -q`

Expected: PASS.

- [ ] **Step 5: Write failing CLI tests**

Inject configuration, factory, discovery, and probe functions into `main(...)` so tests require no devices. Verify human and JSON output, USB discovery, source filtering, no-enabled-source failure, unknown source ID failure, cleanup after one probe fails, and exit code `0` only when every selected source is `SUCCESS`; `DEGRADED`, `FAILED`, and `UNSUPPORTED` produce exit code `1`.

- [ ] **Step 6: Run CLI tests to verify RED**

Run: `python -m pytest tests/integration/test_probe_cameras.py -v`

Expected: FAIL because `scripts/probe_cameras.py` does not exist.

- [ ] **Step 7: Implement `scripts/probe_cameras.py`**

Follow `scripts/check_environment.py`: keep the script thin and place behavior in `aot.hardware_testing`. Human output shows source IDs and sanitized summaries. JSON uses only `SourceTestResult.to_dict()` values. Never print resolved URLs.

- [ ] **Step 8: Run Task 6 tests, credential scan, and the full suite**

Run: `python -m pytest tests/unit/hardware_testing tests/integration/test_probe_cameras.py -q`

Run the Task 4 credential scan again.

Then: `python -m pytest -q`

Expected: all tests PASS and the scan reports no live credentials.

- [ ] **Step 9: Commit Task 6**

```powershell
git add src/aot/hardware_testing scripts/probe_cameras.py tests/unit/hardware_testing tests/integration/test_probe_cameras.py
git commit -m "feat: add camera hardware testing probe"
```

---

### Task 7: Offline Acceptance, Opt-In Hardware Checks, and Evidence

**Files:**
- Create: `tests/offline/test_three_source_capture.py`
- Create: `tests/hardware/test_camera_sources.py`
- Create: `scripts/generate_gate2_fixture.py`
- Create: `tests/fixtures/media/README.md`
- Create: `tests/fixtures/media/gate2-three-frame.avi`
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Create: `docs/evidence/gate-2-camera-layer.md`
- Modify: `docs/traceability/MANUSCRIPT_TO_SOURCE_MATRIX.md`

**Interfaces:**
- Consumes: all Gate 2 public interfaces and the `probe_cameras.py` CLI.
- Produces: deterministic three-source offline acceptance, explicitly gated USB/RTSP hardware checks, and conservative traceability/evidence.

- [ ] **Step 1: Add and document the synthetic offline fixture**

Implement `generate_fixture(output: Path) -> str` in `scripts/generate_gate2_fixture.py`. It writes exactly three visually distinct `640x480` NumPy-generated frames at `5 FPS` using Motion JPEG (`MJPG`), closes the writer in `finally`, reopens the result to verify three decodable frames, and returns the SHA-256. Run it to create `tests/fixtures/media/gate2-three-frame.avi`. Add the narrow `.gitignore` exception `!tests/fixtures/media/gate2-three-frame.avi` beneath the broad `*.avi` rule. Record the command, dimensions, frame count, FPS, codec, license (`CC0-1.0`), and returned SHA-256 in `tests/fixtures/media/README.md`.

- [ ] **Step 2: Write the failing three-source offline acceptance test**

Configure three `OfflineFrameSource` instances pointing at the fixture and start them through `CaptureManager`. Assert each source independently reaches `OPEN` then `EOF`, each emits at least one correctly normalized frame, all source IDs remain isolated, manager shutdown leaves no worker alive, and every backend is released.

- [ ] **Step 3: Run the offline acceptance test to verify RED if any integration gap remains**

Run: `python -m pytest tests/offline/test_three_source_capture.py -v`

Expected: PASS if Tasks 1-6 compose correctly; otherwise FAIL at the missing integration boundary, which must be fixed with a focused regression before continuing.

- [ ] **Step 4: Add opt-in hardware tests**

Register the `hardware` marker in `pyproject.toml`. In `tests/hardware/test_camera_sources.py`, skip the module unless `AOT_RUN_HARDWARE_TESTS=1`. When enabled:

- the ignored local camera configuration path comes from `AOT_HARDWARE_CAMERA_CONFIG`;
- USB and RTSP source IDs come from comma-separated `AOT_HARDWARE_SOURCE_IDS`;
- RTSP source IDs resolve through that configuration's normal environment-variable keys;
- each selected source must open, read the requested number of frames, report negotiated properties, and close;
- a three-source test runs configured IDs concurrently and verifies independent shutdown;
- captured output and failure messages are sanitized.

Do not place URLs or passwords in pytest parameters, node IDs, skip reasons, or assertions.

- [ ] **Step 5: Run the default suite and prove hardware tests are safely skipped**

Run: `python -m pytest -q`

Expected: all automated tests PASS; hardware tests SKIP unless explicitly enabled.

- [ ] **Step 6: Run hardware acceptance when authorized devices are available**

Example:

```powershell
$env:AOT_RUN_HARDWARE_TESTS = "1"
$env:AOT_HARDWARE_CAMERA_CONFIG = "config/cameras.local.yaml"
$env:AOT_HARDWARE_SOURCE_IDS = "usb-1"
python -m pytest tests/hardware/test_camera_sources.py -v
python scripts/probe_cameras.py --camera-config config/cameras.local.yaml --source-id usb-1 --frames 30 --json
```

Expected: selected sources report `SUCCESS`, negotiated properties are present, output contains no credential value, and all handles close. If the required hardware is unavailable, record `NOT RUN` rather than claiming acceptance.

- [ ] **Step 7: Update README, evidence, and traceability conservatively**

Document normal test commands, offline acceptance, CLI usage, opt-in hardware variables, secret handling, and current Gate 2 limitations. In `docs/evidence/gate-2-camera-layer.md`, record exact commands, machine/date, automated results, fixture SHA-256, and hardware results or `NOT RUN`.

Update AOT-002, AOT-003, AOT-004, AOT-006, AOT-022, AOT-030, and AOT-031 with implementation/test links. Use `REBUILD_VERIFIED` only where the matrix's required evidence actually passed; keep hardware-dependent capabilities `PARTIAL` when hardware is `NOT RUN`.

- [ ] **Step 8: Run final Gate 2 verification**

Run:

```powershell
python -m pytest -q
python -m pip check
python scripts/check_environment.py --json
python scripts/probe_cameras.py --app-config config/app.example.yaml --camera-config config/cameras.example.yaml --json
git diff --check
git status --short
```

Expected:

- automated tests PASS, with hardware tests either PASS or explicitly SKIP;
- `pip check` reports no broken requirements;
- Gate 1 diagnostic remains `"ok": true`;
- the camera probe exits `1` for the committed examples because all example sources remain disabled, with an actionable sanitized message and no crash;
- no implementation appears under detection, tracking, ReID, identity, persistence, or UI modules;
- only intended Gate 2 files and documentation differ from the task base.

- [ ] **Step 9: Commit Task 7**

```powershell
git add .gitignore README.md pyproject.toml scripts/generate_gate2_fixture.py tests/offline tests/hardware tests/fixtures/media docs/evidence/gate-2-camera-layer.md docs/traceability/MANUSCRIPT_TO_SOURCE_MATRIX.md
git commit -m "test: verify Gate 2 camera layer"
```

---

## Final Review and Acceptance

After all task commits:

1. Request a fresh whole-branch review against this plan and the approved architecture specification.
2. Grade findings as Critical, Important, or Minor.
3. Fix Critical and Important findings with focused RED-to-GREEN regressions, then rerun the entire suite.
4. Record Minor findings as deferred follow-up rather than expanding Gate 2 silently.
5. Refresh `docs/evidence/gate-2-camera-layer.md` after the final verification run.
6. Do not claim physical USB/RTSP acceptance unless the opt-in hardware suite was actually run on named hardware.
7. Do not begin Gate 3 until this plan's automated acceptance is green, evidence is current, and the owner approves the Gate 2 result.

Gate 2 implementation acceptance requires all automated tests, the offline three-source acceptance, dependency consistency, secret scans, clean shutdown checks, and Gate 1 regression diagnostics to pass. Hardware-dependent matrix entries remain `PARTIAL` until their authorized live checks pass.
