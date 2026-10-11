# Manuscript-to-Source Traceability Matrix

**Baseline date:** 2026-10-06

**Canonical manuscript:** `docs/reference/FINAL MANUSCRIPT PAST TENSE (Revised 4.0).docx`

**Historical review:** `docs/reference/AI_REPOSITORY_HANDOFF.md`

**Purpose:** Map thesis requirements and claims to historical source evidence, rebuild scope, and an objective acceptance gate. This matrix must be maintained as the rebuild progresses.

## Status vocabulary

| Status | Meaning |
|---|---|
| `HISTORICAL_VERIFIED` | The inspected historical repository contains direct source or artifact evidence for the capability. This does not prove current runtime behavior. |
| `PARTIAL` | Related historical or rebuild material exists, but the complete capability is not evidenced. |
| `MANUSCRIPT_ONLY` | The manuscript describes or reports the capability, but inspected historical source does not substantiate it. |
| `REBUILD_SCAFFOLDED` | Configuration or structure exists in the rebuild, but behavior has not been implemented and accepted. |
| `REBUILD_VERIFIED` | The rebuild implements the capability and the listed acceptance evidence passes. |
| `DECISION_REQUIRED` | Owner/team input is required before implementation can be specified. |

`HISTORICAL_VERIFIED` means “present in inspected source,” not “runtime-tested.” `REBUILD_VERIFIED` is the only status that confirms reconstructed behavior.

## Capability matrix

| ID | Capability or requirement | Manuscript evidence | Historical source evidence | Baseline status | Rebuild target | Required acceptance evidence |
|---|---|---|---|---|---|---|
| AOT-001 | Windows 10/11 x64 application | Windows is the implementation platform and other OS support is out of scope. | Historical scripts and PyQt/OpenCV workflow are Windows-oriented. | `HISTORICAL_VERIFIED` | Retain Windows as the first acceptance platform. | Environment check passes on the named Windows machine with recorded Python and package versions. |
| AOT-002 | Three simultaneous camera feeds | Objectives, network architecture, iterations, UI, and actual-testing sections describe three cameras. | `MainDir/MainUITest1.py` constructs three `CameraWidget` instances. | `PARTIAL` | Three independent sources with bounded buffers and clean shutdown. | Offline three-source test passes first; hardware test then opens and closes three configured cameras independently. |
| AOT-003 | USB/local camera support | Manuscript describes three webcams and `cv2.VideoCapture(i)`. | Visible code uses local indices such as `0`, `1`, and `2`. | `HISTORICAL_VERIFIED` | A project-owned USB camera adapter. | Automated source-parsing tests plus hardware open/read/close evidence. |
| AOT-004 | RTSP/IP camera support | Iterations 7-10 describe router setup and three IP CCTV feeds. | RTSP strings appear historically, but the strongest visible UI passes local indices instead of those URLs. | `PARTIAL` | RTSP adapter with environment-sourced URLs and reconnect handling. | Offline/mock reconnect test and authorized live-camera smoke test with secrets redacted. |
| AOT-005 | Externalized camera credentials | Manuscript assumes IP camera access but does not define secure secret handling. | Historical Python embeds plaintext camera credentials. Rebuild has `.env.example` and ignores `.env`. | `REBUILD_VERIFIED` | Environment-backed secret resolution with redacted logs is implemented in `src/aot/config/loader.py` and `src/aot/telemetry/logging_setup.py`. | `tests/unit/config/test_loader.py`, `tests/unit/telemetry/test_logging_setup.py`, and `tests/unit/telemetry/test_environment.py` pass; the 2026-10-11 credential scan and sanitized diagnostic contain no live credential values. |
| AOT-006 | Camera normalization | Manuscript reports severe sensitivity to resolution, lighting, exposure, contrast, hue, and ISO. | Historical code sets some width/height properties but has no unified normalization layer. | `PARTIAL` | Record negotiated properties and convert frames to a canonical format. | Tests cover resize/color/orientation; hardware report records each camera's negotiated properties. |
| AOT-007 | YOLOv8 object/person detection | Core objective, iterations, screenshots, black-box results, and benchmarks. | Multiple scripts load Ultralytics YOLO and model artifacts are present. | `HISTORICAL_VERIFIED` | Detector adapter returning project-owned `Detection` objects. | Deterministic offline smoke test with model hash, sample hash, thresholds, and expected output invariants. |
| AOT-008 | YuNet face detection | Core objective, iterations, screenshots, black-box results, and benchmark. | Local wrappers call OpenCV `FaceDetectorYN`; YuNet ONNX models are present. | `HISTORICAL_VERIFIED` | OpenCV-4-compatible YuNet adapter with documented model provenance. | Offline face-detection test records model hash and validates bounded face boxes. |
| AOT-009 | Multi-detection display | Manuscript describes multi-tracking mode and colored YOLO/YuNet boxes. | Visible camera widget draws YOLO and YuNet boxes, but not robust identities. | `PARTIAL` | Responsive three-panel PyQt display of detections and status. | UI test proves three result streams update without inference on the GUI thread. |
| AOT-010 | User selects a bounding box | Manuscript describes clicking a detected target on any camera; reported result is 28/30 per camera. | Selection experiments exist, but the strongest UI controls are incomplete. | `PARTIAL` | Deterministic point-to-box selection and validated crop creation. | Unit tests cover hit-testing, overlap precedence, clipping, empty clicks, and invalid crops. |
| AOT-011 | Selected-target state | Manuscript describes a selected-target display and locked target. | Historical variables and UI concepts exist without a complete global target model. | `PARTIAL` | Project-owned `GlobalTarget` with source, time, box, image, and embedding metadata. | State-creation test proves immutable selection metadata and valid reference crop. |
| AOT-012 | Per-camera local tracking | Manuscript describes continuous IDs and multi-tracking. | A standard SORT/Kalman implementation exists in archives but is not integrated into the visible app. | `PARTIAL` | Replaceable tracker backend with one independent instance per camera. | Recorded single-camera sequence meets an agreed identity-continuity metric. |
| AOT-013 | ResNet50-style ReID embedding | Iteration 15 describes pretrained ResNet50 feature extraction. | No visible `ReIDModel`, ResNet50 pipeline, weights, or threshold was found. | `MANUSCRIPT_ONLY` | Explicit crop preprocessing, normalized embedding, and model manifest. | Same-person/different-person labeled samples produce recorded similarity distributions. |
| AOT-014 | Feature similarity matching | Manuscript describes feature-distance comparison and a locked match. | No complete visible matcher implementation was found. | `MANUSCRIPT_ONLY` | Configurable cosine similarity/distance with calibrated threshold. | Threshold is selected from validation data; tests cover both sides of the decision boundary. |
| AOT-015 | Global cross-camera identity | Manuscript claims a selected target moves between Cameras 1-3 and reports 30/30 feature matching/global ReID under matched settings. | Independent detections exist, but no global identity manager or camera-handoff source was found. | `MANUSCRIPT_ONLY` | Identity manager associates local tracks with a camera-independent target. | Labeled three-camera scenarios report handoff attempts, successes, false matches, and failures. |
| AOT-016 | Mismatched-camera robustness | Manuscript reports only 3/30 success when camera resolutions/settings differ. | README and source review support sensitivity, not a solution. | `DECISION_REQUIRED` | Define supported variance and normalization/calibration policy. | Acceptance set varies lighting/settings and reports performance separately from matched-camera tests. |
| AOT-017 | Lost-target state machine | Flowchart describes visible, missing, global search, and reidentified behavior. | No explicit global target state machine was found. | `MANUSCRIPT_ONLY` | `VISIBLE`, `TEMPORARILY_LOST`, `SEARCHING`, `REIDENTIFIED`, and `DISENGAGED`. | Unit tests cover valid transitions, timeouts, recovery, and disengagement. |
| AOT-018 | Kalman prediction | Manuscript flow describes prediction while the selected target is missing. | SORT/Kalman code exists in archives, but no verified integration with global identity exists. | `PARTIAL` | Prediction remains inside the tracking layer and cannot create identity by itself. | Occlusion test measures prediction window and recovery without false identity assignment. |
| AOT-019 | SQLite persistence | Iteration 16 and schema discussion describe targets, images, cameras, and tracking sessions. | No SQLite code, schema, or database was found in visible historical source. | `MANUSCRIPT_ONLY` | Non-blocking persistence for targets, images, cameras, sessions, and events. | Schema migration/creation and CRUD tests pass in a temporary database. |
| AOT-020 | Save and load target | Iteration 20 and UI sections describe persistent target features. | No `save_target` or `load_target` implementation was found. | `MANUSCRIPT_ONLY` | Save/load through the same `GlobalTarget` and repository abstractions. | Round-trip test preserves required metadata and produces a usable loaded target. |
| AOT-021 | Uploaded-image target | Iteration 21 describes detecting a target in an uploaded image. | No `upload_image` tracking flow was found. | `MANUSCRIPT_ONLY` | Validate image, detect/select candidate, extract embedding, create `GlobalTarget`. | Tests cover valid image, unsupported format, no candidate, multiple candidates, and crop selection. |
| AOT-022 | Start/stop and clean shutdown | Storyboard and requirements describe start/stop controls. | Historical UI prototypes exist; resource release is incomplete or inconsistent. | `PARTIAL` | Coordinated worker lifecycle and release of every capture/model/UI resource. | Repeated start/stop test leaves no camera handle or worker running. |
| AOT-023 | Responsive PyQt UI | Manuscript describes a three-camera dashboard and controls. | PyQt layouts exist, but heavy work occurs in timer callbacks/import paths. | `PARTIAL` | GUI thread renders state and emits commands only. | `pytest-qt` test plus runtime telemetry confirms UI responsiveness under offline load. |
| AOT-024 | Target disengagement | Requirements and storyboard describe disabling/disengaging the target. | A button/placeholder exists without complete identity state handling. | `PARTIAL` | Clear global target, emit event, and return to multi-detection safely. | State and UI tests prove embeddings, selection display, and lock state are cleared. |
| AOT-025 | Tracking event history | Manuscript schema describes tracking sessions and camera events. | No historical event store was found. | `MANUSCRIPT_ONLY` | Persist selected, visible, lost, reidentified, disengaged, and camera connection events. | Event-order and database integration tests pass without blocking the frame pipeline. |
| AOT-026 | End-to-end performance measurement | Manuscript reports model benchmarks and JMeter process results. | Training outputs exist, but reported whole-system numbers were not reproduced. | `PARTIAL` | Measure capture, detection, face, tracking, ReID, render, resource use, and stale frames separately. | Reproducible benchmark report names hardware, model hashes, configuration, and three-camera end-to-end results. |
| AOT-027 | Minimum displayed/processed FPS | README reports roughly 15-25 FPS; prerequisites propose an initial 15 FPS target if hardware permits. | No reproducible three-camera benchmark was verified. | `DECISION_REQUIRED` | Owner approves per-camera FPS and latency thresholds before performance acceptance. | Automated telemetry report satisfies the approved thresholds on named hardware. |
| AOT-028 | CPU-safe startup | Manuscript expects GPU acceleration; rebuild prerequisites require graceful CPU fallback. | Historical code may call CUDA-only APIs after reporting CPU fallback. | `REBUILD_VERIFIED` | `src/aot/telemetry/device_probe.py` checks import and CUDA capabilities before device-only APIs and returns a controlled CPU result. | `tests/unit/telemetry/test_device_probe.py` passes CUDA available, unavailable, invalid-index, query-failure, and PyTorch-import-failure paths; the 2026-10-11 diagnostic resolved CPU with a warning. |
| AOT-029 | Reproducible dependencies | Manuscript lists technologies but no locked environment. | Historical repository has no dependency manifest. | `REBUILD_VERIFIED` | Python 3.11 package metadata, the hardware-selected PyTorch rule, editable installation, and the Gate 1 diagnostic are documented and implemented. | `docs/evidence/gate-1-environment.md` records 67 passing tests, clean `pip check`, Python 3.11.8 x64, package versions, and successful human/JSON diagnostics on 2026-10-11. |
| AOT-030 | Offline-first verification | Manuscript emphasizes live cameras; rebuild prerequisites require prerecorded evidence first. | Historical AVI outputs and datasets exist, but no automated offline suite was found. | `REBUILD_SCAFFOLDED` | Licensed/provenanced offline samples for deterministic integration tests. | Offline suite runs without cameras, network, GUI interaction, or historical runtime imports. |
| AOT-031 | Historical/rebuild evidence separation | Not a thesis feature; required to preserve academic and engineering honesty. | Handoff identifies a major manuscript/source mismatch. | `REBUILD_SCAFFOLDED` | Maintain this matrix and reference manifest throughout development. | Every completed capability links to implementation, tests, evidence, and whether it is reconstructed or historical. |

## Manuscript-reported results retained as historical claims

These values guide test design but are not rebuild acceptance evidence:

| Reported item | Manuscript result | Rebuild treatment |
|---|---:|---|
| Target selection from Camera 1 | 28/30 | Recreate as deterministic hit-testing plus scenario testing. |
| Target selection from Camera 2 | 28/30 | Same acceptance method as Camera 1. |
| Target selection from Camera 3 | 28/30 | Same acceptance method as Camera 1. |
| YOLOv8 object detection | 30/30 | Replace count-only evidence with labeled offline metrics and invariants. |
| YuNet face detection | 30/30 | Replace count-only evidence with labeled offline metrics and invariants. |
| Feature extraction | 30/30 | Define valid-crop and embedding-quality criteria. |
| Feature matching | 30/30 | Measure genuine/impostor score distributions and false matches. |
| Global re-identification under matched settings | 30/30 | Measure handoff success and false-match rate on labeled scenarios. |
| Global re-identification with different camera settings | 3/30 | Treat camera variation as an explicit adverse test dimension. |

## Gate 0 decisions and actions

| Item | State | Owner/action |
|---|---|---|
| Canonical manuscript selected | Complete | DOCX recorded in `docs/reference/REFERENCE_MANIFEST.md`. |
| Reference artifacts fingerprinted | Complete | Sizes and SHA-256 values recorded in the reference manifest. |
| Historical repository frozen/backed up | Owner confirmation required | Preserve `ThesisV2`, its untracked artifacts, and commit `469ddba`. |
| Historical camera credentials rotated | Owner action required | Rotate credentials on the camera/NVR; do not edit historical evidence first. |
| Final demonstration source located | Open | Search other devices, storage, branches, and group-member copies. |
| Initial acceptance decisions | Open | Resolve the questions below before their associated implementation gates. |

## Decisions required before advanced implementation

1. Is the rebuild person-only, or must it track generic objects too?
2. Must both USB and RTSP sources be included in the final acceptance demo?
3. What minimum per-camera FPS and maximum end-to-end latency are acceptable?
4. How long may a target remain missing before its global identity is abandoned?
5. What labeled data and metric define a successful cross-camera handoff?
6. Is face information mandatory, optional, or an auxiliary identity cue?
7. Is CPU-only operation an acceptance requirement or only a graceful fallback?
8. Which YOLO weight is the initial baseline, and must the historical custom `best.pt` be evaluated?
9. What target images, embeddings, and session data may be persisted under the project's privacy rules?
10. Which manuscript-reported results must be reproduced rather than replaced with better-defined rebuild metrics?

## Maintenance rule

When a capability changes status:

1. link the implementing file/class/function;
2. link the exact test or evidence artifact;
3. record model/data/config hashes where relevant;
4. state whether the capability was recovered historically or reconstructed;
5. change to `REBUILD_VERIFIED` only after its acceptance evidence passes.
