# Adaptive Object Tracking Rebuild — Prerequisites and Project Structure

**Document purpose:** Hand-off specification for Claude/ChatGPT/Codex before implementation.  
**Status:** Rebuild planning baseline. No behavior in `ThesisV2` is to be modified.  
**Date:** 2026-10-06  
**New project folder:** `AdaptiveObjectTracking-Rebuild/`  
**Historical reference:** `ThesisV2/`

---

## 1. Non-negotiable repository rule

`ThesisV2` is the historical thesis repository and must be treated as **read-only source material**.

Do **not**:

- reorganize `ThesisV2`;
- delete duplicates, archives, datasets, models, generated runs, or manuscripts;
- rewrite its history;
- "clean up" credentials by editing the historical files before a preserved copy/tag exists;
- develop the rebuild inside `ThesisV2`;
- import Python modules directly from `ThesisV2` at runtime;
- make the new application depend on relative paths inside `ThesisV2`;
- silently copy code and present the copied implementation as a newly verified feature.

The rebuild must live beside it, not inside it.

Recommended workspace:

```text
ThesisWorkspace/
|
|-- ThesisV2/                         # READ-ONLY historical/source-of-truth repository
|
`-- AdaptiveObjectTracking-Rebuild/  # ACTIVE development repository
```

The role of `ThesisV2` is to answer questions such as:

- What did the original project attempt?
- Which models, datasets, scripts, UI concepts, benchmarks, and experimental outputs existed?
- What does the manuscript claim the final system did?
- Which behavior can be traced to actual source?
- Which assets might be reusable after provenance and licensing checks?

The role of `AdaptiveObjectTracking-Rebuild` is different:

- provide a reproducible implementation;
- separate camera capture, detection, tracking, ReID, persistence, and UI;
- prove behavior through tests and recorded evidence;
- make all configuration explicit;
- keep credentials out of source control;
- distinguish historical thesis claims from newly reconstructed behavior.

---

## 2. Why a separate rebuild exists

The current handoff concludes that the manuscript describes a more complete system than the visible source tree.

The strongest visible application path is approximately:

```text
Camera(s)
    |
    v
YOLOv8 object/person detection
    +
YuNet face detection
    |
    v
PyQt5 display / incomplete controls
```

The manuscript's intended system is substantially broader:

```text
Three USB / RTSP camera feeds
        |
        v
Frame capture and normalization
        |
        +--------------------+
        |                    |
        v                    v
YOLOv8 detection       YuNet face detection
        |                    |
        +---------+----------+
                  |
                  v
          Multi-detection view
                  |
                  v
      User selects bounding box
                  |
                  v
       Capture selected target
                  |
                  v
      Visual feature extraction
         (ResNet50-style ReID)
                  |
                  v
         Target feature gallery
                  |
                  v
   Per-camera detection + tracking
                  |
                  v
     Cross-camera feature matching
                  |
                  v
       Global target identity
            /           \
           /             \
          v               v
 target visible       target missing
      |                    |
      v                    v
 continue track       Kalman prediction /
      |               global re-identification
      +---------+----------+
                |
                v
       Three-camera UI state
                |
                v
   SQLite target/session persistence
                |
        +-------+--------+
        |                |
        v                v
 save/load target   uploaded-image target
```

This workflow is the **functional target** of the rebuild because it reflects the manuscript-defined intended behavior.

It does **not** mean every box above is already repository-verified in `ThesisV2`.

---

## 3. Source-of-truth hierarchy

Claude should use the following evidence order when resolving implementation questions.

### Tier 1 — Historical evidence

1. Final manuscript confirmed by the project owner.
2. `AI_REPOSITORY_HANDOFF.md`.
3. Source actually present in `ThesisV2`.
4. Git history / historical archives.
5. Saved training and inference outputs.

Historical evidence answers **what the thesis claimed or previously implemented**.

### Tier 2 — Official technical references

Official/upstream documentation answers **how the selected technology should be implemented now**.

Primary references for the rebuild:

- YuNet / libfacedetection  
  https://github.com/ShiqiYu/libfacedetection

- OpenCV Zoo YuNet model  
  https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md

- Ultralytics YOLOv8 documentation  
  https://docs.ultralytics.com/models/yolov8

- Ultralytics YOLOv8 model repository  
  https://huggingface.co/Ultralytics/YOLOv8

- Ultralytics Python API  
  https://docs.ultralytics.com/usage/python

- Ultralytics tracking mode  
  https://docs.ultralytics.com/modes/track

- PyTorch installation matrix  
  https://pytorch.org/get-started/locally/

### Tier 3 — Rebuild engineering decisions

If the manuscript does not define an implementation detail, Claude may recommend a modern implementation, but it must label it as a **rebuild decision**, not as an original thesis fact.

Examples:

- ByteTrack vs BoT-SORT vs a recovered SORT implementation;
- threading architecture;
- queue sizes;
- database access layer;
- configuration schema;
- logging library;
- test framework.

---

## 4. Technology baseline

### 4.1 Operating system

Primary target:

```text
Windows 10/11 x64
```

This matches the historical project orientation and the expected USB-camera development environment.

Linux support can be considered later, but Windows is the initial acceptance platform.

### 4.2 Python

Recommended baseline:

```text
Python 3.11 x64
```

Reason:

- the inspected historical environment used Python 3.11-era tooling;
- current PyTorch supports Python 3.11 on Windows;
- PyQt5 5.15.x has Windows wheels compatible with this baseline;
- it avoids introducing a Python-version migration while the functional rebuild is still being stabilized.

Do not develop against the system Python.

Create a dedicated environment:

```powershell
cd <workspace>\AdaptiveObjectTracking-Rebuild

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
```

Verify:

```powershell
python --version
where python
python -m pip --version
```

---

## 5. GPU / CUDA prerequisite

GPU acceleration is recommended, but CPU startup must not crash.

The rebuild must support this behavior:

```text
CUDA available
    -> use configured CUDA device

CUDA unavailable
    -> log CPU fallback
    -> continue running
```

Never call CUDA-only APIs before checking capability.

Required runtime probe:

```python
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

### PyTorch installation rule

Do not guess a CUDA wheel.

Use the official PyTorch installer for the actual machine:

https://pytorch.org/get-started/locally/

For NVIDIA systems, install the correct CUDA-enabled `torch` + `torchvision` pair **before** installing the remaining project requirements.

Then run:

```powershell
pip install -r requirements.txt
```

The supplied baseline requirements currently pair:

```text
torch==2.14.1
torchvision==0.29.1
```

If the official PyTorch selector recommends a different supported pair for the target machine, update these two lines together and record the change in the rebuild README.

---

## 6. Camera prerequisites

The manuscript intends three feeds.

Supported source types should be represented by one interface:

```text
USB/local camera
    0
    1
    2

RTSP/IP camera
    rtsp://...

Recorded/offline video
    sample.mp4
```

Do not hard-code credentials.

Use configuration such as:

```yaml
cameras:
  - id: camera_1
    type: usb
    source: 0

  - id: camera_2
    type: usb
    source: 1

  - id: camera_3
    type: usb
    source: 2
```

For RTSP:

```yaml
cameras:
  - id: camera_1
    type: rtsp
    source_env: CAMERA_1_URL
```

And a local ignored `.env`:

```text
CAMERA_1_URL=rtsp://user:password@example
```

Never commit the real value.

### Camera normalization

The historical manuscript reports that inconsistent camera settings severely harmed global re-identification.

The rebuild must therefore record and normalize, where hardware permits:

- width;
- height;
- FPS;
- exposure;
- brightness;
- contrast;
- color handling;
- orientation.

At minimum, each frame must be converted into a known canonical format before detection/ReID.

Recommended initial development resolution:

```text
640 x 480
```

YOLO may independently use an inference `imgsz`, such as 640.

Do not confuse camera capture resolution with YOLO's square inference size.

---

## 7. YOLOv8 requirement

Use the maintained Ultralytics Python API:

```python
from ultralytics import YOLO

model = YOLO("models/yolo/yolov8n.pt")
results = model.predict(frame, imgsz=640, verbose=False)
```

Initial baseline model:

```text
yolov8n.pt
```

Reason: establish correctness and multi-camera real-time behavior before increasing model size.

Candidate later models:

```text
yolov8s.pt
yolov8m.pt
custom best.pt
```

A larger model must not be selected simply because it is more accurate in isolation. Three-camera aggregate latency must be benchmarked.

### Person-only optimization

For a person-following thesis workflow, prefer filtering the COCO `person` class when generic pretrained weights are used.

The detector adapter must return project-owned data classes rather than allowing Ultralytics `Results` objects to propagate through the whole application.

Conceptual result:

```python
Detection(
    xyxy=(x1, y1, x2, y2),
    confidence=0.91,
    class_id=0,
    class_name="person",
)
```

This prevents the application from becoming tightly coupled to one Ultralytics result API.

---

## 8. YuNet requirement

YuNet remains the intended face detector.

Preferred Python integration:

```text
OpenCV FaceDetectorYN
+
OpenCV Zoo YuNet ONNX model
```

The upstream OpenCV Zoo documentation identifies YuNet as a lightweight face detector and documents OpenCV camera/image demos.

### Important OpenCV-version decision

This rebuild intentionally begins on:

```text
opencv-python==4.14.0.94
```

Therefore begin with an OpenCV-4-compatible YuNet ONNX model and verify its input-shape behavior.

Do **not** silently switch to an OpenCV 5 dynamic-shape YuNet model without updating:

- requirements;
- model manifest;
- smoke tests;
- model/input handling documentation.

Store the chosen model at:

```text
models/yunet/
```

Document:

- original URL;
- filename;
- SHA-256;
- OpenCV version tested;
- input behavior;
- license.

---

## 9. Tracking requirement

There are two different tracking problems and they must not be confused.

### A. Per-camera tracking

Goal:

> Maintain a local track ID for detections across consecutive frames from one camera.

Interface:

```python
tracks = tracker.update(detections, frame)
```

Each camera gets independent tracker state.

Do not reuse one mutable tracker instance for unrelated camera streams.

### B. Global cross-camera identity

Goal:

> Decide whether a detected/tracked person on Camera B or Camera C is the selected target previously observed on Camera A.

This is **not solved merely by YOLO track IDs**.

The global identity layer requires appearance/ReID features and explicit target state.

---

## 10. Recommended tracker baseline — rebuild decision

This is a **new engineering recommendation**, not a claim about the original thesis implementation.

Create an abstraction:

```python
class TrackerBackend:
    def update(self, frame, detections):
        ...
```

Then allow interchangeable backends.

Initial candidate:

```text
ByteTrack
```

Alternative:

```text
BoT-SORT
```

Ultralytics currently provides tracking APIs and configurable tracker backends. This can be used to establish a stable per-camera baseline.

However:

- per-camera tracking is not cross-camera ReID;
- the tracker backend must remain replaceable;
- recovered historical SORT/Kalman code should only be reused after provenance and behavior review.

---

## 11. Re-identification requirement

The manuscript describes ResNet50-style feature extraction and feature-distance matching.

The rebuild should represent this explicitly.

Conceptual interface:

```python
embedding = reid_model.extract(person_crop)
score = matcher.compare(target_embedding, embedding)
```

Baseline candidate:

```text
torchvision.models.resnet50
```

Recommended initial pipeline:

```text
person bounding box
      |
      v
crop + validation
      |
      v
resize / normalize
      |
      v
ResNet50 backbone
      |
      v
feature vector
      |
      v
L2 normalization
      |
      v
cosine similarity / distance
```

Do not lock the thesis to one similarity threshold before calibration.

Thresholds belong in configuration:

```yaml
reid:
  match_threshold: null   # calibrate from validation data
```

The first ReID acceptance test should use prerecorded, labeled cross-camera examples.

---

## 12. Face information and ReID

Do not assume a detected face alone is always available.

The manuscript itself notes sensitivity to orientation, covering, lighting, and camera differences.

The identity manager should be capable of combining cues later, for example:

```text
person appearance embedding
face detection availability
face crop/quality
spatial/temporal context
tracker continuity
```

The initial architecture should support those cues without requiring all of them.

---

## 13. Lost-target handling

The manuscript describes:

```text
target disappears
    |
    v
position prediction
    +
global re-identification
```

Represent lost-target state explicitly:

```text
VISIBLE
TEMPORARILY_LOST
SEARCHING
REIDENTIFIED
DISENGAGED
```

Kalman prediction belongs to the tracking layer and must not be hidden inside UI code.

Do not claim successful lost-target recovery until an offline test demonstrates it.

---

## 14. Target selection requirement

The user must be able to click/select a detected target.

Selection flow:

```text
mouse click
    |
    v
find bounding box under cursor
    |
    v
validate candidate
    |
    v
capture target crop
    |
    v
extract ReID embedding
    |
    v
create GlobalTarget
    |
    v
switch UI into selected-target mode
```

Store enough metadata for traceability:

```python
GlobalTarget(
    target_id=...,
    source_camera_id=...,
    selected_at=...,
    initial_bbox=...,
    reference_image_path=...,
    embedding=...,
)
```

The embedding should not be rendered directly into UI code.

---

## 15. SQLite requirement

Use Python's built-in `sqlite3` initially.

No extra package is required.

Persist at minimum the manuscript concepts:

```text
targets
target_images
cameras
tracking_sessions
tracking_events
```

Recommended event examples:

```text
TARGET_SELECTED
TARGET_VISIBLE
TARGET_LOST
TARGET_REIDENTIFIED
TARGET_DISENGAGED
CAMERA_CONNECTED
CAMERA_DISCONNECTED
```

Database writes must not block the camera/UI loop.

---

## 16. Save/load and uploaded-image target requirement

The architecture should support both:

```text
Live target selection
        |
        v
GlobalTarget
```

and:

```text
Uploaded reference image
        |
        v
detect/validate target
        |
        v
extract embedding
        |
        v
GlobalTarget
```

The matching pipeline after `GlobalTarget` creation should be shared.

Do not implement two unrelated tracking systems.

---

## 17. UI requirement

Keep PyQt5 initially because it matches the historical application.

The UI should own:

- window layout;
- camera panels;
- target-selection interaction;
- buttons and control state;
- status display;
- logs/notifications.

The UI should **not** own:

- model loading logic;
- ReID math;
- database queries;
- camera reconnect algorithms;
- tracker implementation;
- CUDA probing.

Expected top-level UI controls:

```text
Start
Stop
Multi Tracking
Single Target Tracking
Disable / Disengage Target
Save Target
Load Target
Upload Target Image
Camera Status
Tracking Logs
Threshold / settings controls
```

Exact controls should be reconciled with the final manuscript before acceptance.

---

## 18. Concurrency requirement

Three camera feeds plus inference must not run as one blocking UI callback.

Minimum architecture:

```text
Camera capture worker(s)
        |
        v
bounded latest-frame buffer
        |
        v
inference / pipeline worker
        |
        v
thread-safe result/state update
        |
        v
PyQt UI render
```

Rules:

1. Never let an unbounded frame queue grow.
2. Real-time tracking should prefer the newest frame rather than processing a large stale backlog.
3. Do not call long-running inference from the PyQt GUI thread.
4. A slow camera must not freeze other camera panels.
5. Camera disconnect/reconnect must be contained within that camera source.
6. Shutdown must release all `VideoCapture` objects and workers.

---

## 19. Performance measurements

Measure the whole application, not only YOLO inference.

Track separately:

```text
capture FPS per camera
detection latency
face-detection latency
tracking latency
ReID latency
end-to-end frame latency
render FPS
GPU utilization
CPU utilization
RAM
VRAM
dropped/stale frames
```

Do not report a single-model benchmark as three-camera application FPS.

Initial rebuild target:

```text
Correctness first.
Then establish >= 15 FPS displayed/processed per active camera if hardware permits.
30 FPS is a performance goal, not an assumed guarantee.
```

After baseline functionality is stable, investigate:

- smaller YOLO model;
- reduced inference resolution;
- FP16 on compatible GPU;
- TensorRT/ONNX only after PyTorch baseline;
- asynchronous pipelines;
- batched inference where it actually improves latency;
- detector frame skipping with tracker updates between detection frames.

---

## 20. `requirements.txt`

The actual `requirements.txt` is included beside this document.

Baseline:

```text
torch==2.14.1
torchvision==0.29.1

ultralytics==8.4.158
opencv-python==4.14.0.94
numpy>=1.26,<3
Pillow>=10,<13

PyQt5==5.15.11

PyYAML>=6,<7
python-dotenv>=1,<2
psutil>=5.9,<8

onnx>=1.16,<2
onnxruntime>=1.18,<2

pytest>=8,<9
pytest-qt>=4.4,<5
```

### Why `supervision` is not in the initial baseline

The historical repository used an old `supervision` API, and some of its calls are version-sensitive.

For the first rebuild, avoid adding an annotation abstraction that is not required for the core thesis behavior.

Bounding boxes can be rendered using:

- OpenCV; or
- data already returned by Ultralytics.

`supervision` may be added later if it provides a clear requirement-driven benefit.

---

## 21. Proposed folder structure

```text
AdaptiveObjectTracking-Rebuild/
|
|-- README.md
|-- REBUILD_PREREQUISITES.md
|-- requirements.txt
|-- .gitignore
|-- .env.example
|
|-- src/
|   `-- aot/
|       |-- __init__.py
|       |-- app.py
|       |
|       |-- config/
|       |   |-- __init__.py
|       |   `-- settings.py
|       |
|       |-- capture/
|       |   |-- __init__.py
|       |   |-- base.py
|       |   |-- usb_camera.py
|       |   |-- rtsp_camera.py
|       |   `-- camera_manager.py
|       |
|       |-- detection/
|       |   |-- __init__.py
|       |   |-- types.py
|       |   |-- yolo_detector.py
|       |   `-- yunet_detector.py
|       |
|       |-- selection/
|       |   |-- __init__.py
|       |   `-- target_selector.py
|       |
|       |-- tracking/
|       |   |-- __init__.py
|       |   |-- base.py
|       |   |-- per_camera_tracker.py
|       |   |-- kalman.py
|       |   `-- track_state.py
|       |
|       |-- reid/
|       |   |-- __init__.py
|       |   |-- feature_extractor.py
|       |   |-- matcher.py
|       |   `-- gallery.py
|       |
|       |-- identity/
|       |   |-- __init__.py
|       |   |-- global_target.py
|       |   `-- identity_manager.py
|       |
|       |-- persistence/
|       |   |-- __init__.py
|       |   |-- database.py
|       |   |-- schema.sql
|       |   `-- repositories.py
|       |
|       |-- pipeline/
|       |   |-- __init__.py
|       |   |-- frame_pipeline.py
|       |   `-- orchestrator.py
|       |
|       |-- ui/
|       |   |-- __init__.py
|       |   |-- main_window.py
|       |   |-- camera_panel.py
|       |   `-- target_panel.py
|       |
|       `-- telemetry/
|           |-- __init__.py
|           |-- fps.py
|           |-- metrics.py
|           `-- logging_setup.py
|
|-- config/
|   |-- cameras.example.yaml
|   |-- detection.yaml
|   |-- tracking.yaml
|   `-- reid.yaml
|
|-- models/
|   |-- README.md
|   |-- manifest.yaml
|   |-- yolo/
|   |-- yunet/
|   `-- reid/
|
|-- data/
|   |-- offline_samples/
|   `-- uploads/
|
|-- runtime/
|   |-- logs/
|   |-- snapshots/
|   `-- database/
|
|-- scripts/
|   |-- check_environment.py
|   |-- probe_cameras.py
|   |-- test_yolo.py
|   |-- test_yunet.py
|   |-- test_reid.py
|   `-- run_offline_pipeline.py
|
|-- tests/
|   |-- unit/
|   |-- integration/
|   |-- offline/
|   `-- hardware/
|
`-- docs/
    |-- architecture.md
    |-- acceptance_criteria.md
    |
    |-- reference/
    |   |-- THESISV2_SOURCE_MAP.md
    |   `-- MODEL_PROVENANCE.md
    |
    `-- traceability/
        `-- MANUSCRIPT_TO_SOURCE_MATRIX.md
```

---

## 22. Folder responsibilities

### `src/aot/capture`

Owns:

- `cv2.VideoCapture`;
- USB indices;
- RTSP URLs;
- reconnect behavior;
- resolution/FPS negotiation;
- timestamps;
- frame ownership.

No YOLO or UI code belongs here.

### `src/aot/detection`

Owns:

- YOLO model loading/inference;
- YuNet model loading/inference;
- conversion from library-specific results into project data classes.

### `src/aot/tracking`

Owns:

- local/per-camera track IDs;
- motion prediction;
- tracker state;
- optional Kalman handling.

### `src/aot/reid`

Owns:

- visual feature extraction;
- preprocessing;
- embedding normalization;
- similarity/distance calculation;
- target feature gallery.

### `src/aot/identity`

Owns:

- the selected global target;
- matching local tracks to the global target;
- camera handoff;
- visible/lost/reidentified state transitions.

### `src/aot/persistence`

Owns:

- SQLite;
- target/session/event records;
- save/load operations.

### `src/aot/pipeline`

Owns orchestration only.

Example:

```text
frame
 -> detection
 -> local tracking
 -> ReID candidate extraction
 -> global identity update
 -> event generation
 -> UI result
```

### `src/aot/ui`

Consumes application state and emits user commands.

It must not become the location where the computer-vision pipeline is implemented.

### `runtime/`

Ignored by Git.

Contains:

- SQLite database;
- logs;
- snapshots;
- temporary outputs.

### `models/`

Do not casually commit large weights.

Each model must be described in `models/manifest.yaml`.

Example:

```yaml
models:
  yolo:
    name: yolov8n
    file: yolo/yolov8n.pt
    source: https://...
    sha256: TBD

  yunet:
    name: face_detection_yunet
    file: yunet/<selected-model>.onnx
    source: https://...
    sha256: TBD
```

---

## 23. Historical-reference boundary

The rebuild may consult:

```text
../ThesisV2/
```

during engineering review.

It must not do this in production code:

```python
sys.path.append("../ThesisV2")
from YoloV8 import ...
```

It must not do:

```python
MODEL_PATH = "../ThesisV2/yolov8n.pt"
```

Instead, if an old model is deliberately reused:

1. identify the historical file;
2. record its SHA-256;
3. record where it came from;
4. copy it deliberately into the rebuild's model store;
5. document the action;
6. verify its license;
7. test it independently.

The historical repository remains untouched.

---

## 24. Secrets rule

The handoff identified camera credentials embedded in historical Python files.

Treat those secrets as compromised.

Before live RTSP testing:

1. rotate the camera/NVR password;
2. use `.env` or another ignored local secret store;
3. commit only `.env.example`;
4. never put credentials in test fixtures, screenshots, Claude prompts, Git issues, or logs.

Example `.env.example`:

```text
CAMERA_1_URL=
CAMERA_2_URL=
CAMERA_3_URL=
```

---

## 25. Suggested `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Virtual environments
.venv/
venv/

# Secrets
.env
*.secret
config/cameras.local.yaml

# Runtime
runtime/
logs/

# IDE
.vscode/
.idea/

# Models / generated inference artifacts
models/**/*.pt
models/**/*.onnx
runs/
*.avi
*.mp4

# Datasets / local samples
data/uploads/
```

Model policy may later permit selected small weights, but that should be deliberate rather than accidental.

---

## 26. Rebuild milestones

### Gate 0 — Historical freeze

Before coding:

- back up `ThesisV2`;
- record current commit;
- preserve untracked thesis artifacts;
- rotate historical camera credentials;
- confirm the final manuscript;
- create the traceability matrix.

**Exit condition:** historical source is preserved and no rebuild work is occurring inside it.

### Gate 1 — Reproducible environment

Implement only:

- Python environment;
- dependencies;
- configuration loader;
- device/CUDA probe;
- structured logging.

**Exit condition:**

```text
python scripts/check_environment.py
```

works on the intended Windows machine.

### Gate 2 — Camera layer

Implement one source interface for:

- USB;
- RTSP;
- offline video.

**Exit condition:** three configured test sources can open independently and shutdown cleanly.

### Gate 3 — Detection

Implement:

- YOLO adapter;
- YuNet adapter;
- project-owned `Detection` data structures.

**Exit condition:** offline sample frames pass deterministic detection smoke tests.

### Gate 4 — Three-camera UI

Implement PyQt display without tracking/ReID.

**Exit condition:** UI stays responsive while three feeds run.

### Gate 5 — Target selection

Implement click-on-bounding-box selection.

**Exit condition:** a selected target crop and metadata are created deterministically.

### Gate 6 — Per-camera tracking

Implement tracker abstraction and first backend.

**Exit condition:** a local identity persists through a recorded single-camera test.

### Gate 7 — ReID

Implement ResNet50-style feature extraction and matching.

**Exit condition:** labeled same-person/different-person samples produce measurable similarity distributions and a documented threshold-selection process.

### Gate 8 — Global cross-camera identity

Associate the selected target with tracks on other cameras.

**Exit condition:** prerecorded three-camera scenarios show a documented handoff success rate.

### Gate 9 — Lost-target recovery

Implement:

- `TEMPORARILY_LOST`;
- prediction/tracker buffer;
- global search;
- `REIDENTIFIED`.

**Exit condition:** recovery tests include occlusion and camera handoff.

### Gate 10 — Persistence and upload

Implement:

- SQLite schema;
- target save/load;
- tracking session/event history;
- uploaded reference image target.

### Gate 11 — Performance

Only after functional correctness:

- profile three-camera throughput;
- optimize resolution/model size;
- evaluate FP16/export formats;
- compare CPU/GPU behavior.

---

## 27. Required tests

### Unit

Examples:

```text
config parsing
camera source parsing
bbox hit-testing
bbox clipping
crop validation
embedding normalization
cosine matching
identity state transitions
database CRUD
```

### Offline integration

Use prerecorded material before live cameras.

Examples:

```text
video -> YOLO -> detections
video -> detection -> tracker IDs
selected crop -> ReID -> similarity
three videos -> global target handoff
temporary loss -> recovery
```

### Hardware

Separate from normal CI:

```text
USB camera open/close
three simultaneous cameras
RTSP reconnect
CUDA inference
camera property normalization
```

---

## 28. Initial acceptance criteria to agree before coding advanced features

Do not allow Claude to invent these silently.

The owner/team must define:

1. What exact object classes are in scope: person only or generic objects?
2. Is the final target primarily a **person-tracking** system?
3. Must all three cameras run simultaneously?
4. Must both USB and RTSP be supported in the final demo?
5. Minimum acceptable per-camera FPS?
6. Maximum acceptable end-to-end latency?
7. How many seconds can a target disappear before the track is abandoned?
8. How will cross-camera ReID accuracy be measured?
9. What is considered a successful camera handoff?
10. Is face information mandatory, optional, or only an auxiliary cue?
11. Is CPU-only mode required or only graceful fallback?
12. Which YOLO weight is the final baseline?
13. Is the custom thesis `best.pt` still required?
14. What data may be persisted for privacy/ethics purposes?
15. Which historical thesis test results must be reproduced?

---

## 29. Decisions Claude must not make silently

Claude should stop and surface a decision before changing any of these:

```text
Python major/minor baseline
PyTorch/CUDA build
OpenCV 4 -> OpenCV 5
YuNet model generation/version
YOLOv8 -> newer YOLO family
YOLO weight size
tracking backend
ReID architecture
ReID threshold
database schema breaking changes
camera resolution
secret storage
historical model/dataset reuse
copying code from ThesisV2
```

A rebuild decision is acceptable. It simply has to be documented as a rebuild decision.

---

## 30. First coding task for Claude

Do **not** start by implementing the full tracker.

First task:

> Bootstrap `AdaptiveObjectTracking-Rebuild` so it can prove the development machine, configuration, models, and offline test sources are usable without modifying `ThesisV2`.

The first pull/commit should contain only approximately:

```text
project structure
.gitignore
.env.example
requirements.txt
configuration loader
device/CUDA probe
camera source data model
model manifest
logging
basic tests
scripts/check_environment.py
```

Then create three separate smoke-test scripts:

```text
scripts/probe_cameras.py
scripts/test_yolo.py
scripts/test_yunet.py
```

Only after those are reproducible should the PyQt multi-camera pipeline be implemented.

---

## 31. Current upstream observations used for this rebuild plan

### YuNet

OpenCV Zoo describes YuNet as a lightweight face detector and currently distinguishes:

- a newer dynamic-input model intended for OpenCV 5.x ONNX Runtime behavior;
- the `2023mar` model with fixed input shape used with OpenCV 4.x behavior.

This is why the initial rebuild intentionally avoids casually mixing OpenCV 4 and OpenCV 5 model assumptions.

### libfacedetection

The upstream project provides the YuNet lineage and also points users to the OpenCV Zoo ONNX model.

The rebuild does not need to compile the C++ `libfacedetection` implementation for the initial Python architecture. It is a technical/reference source for YuNet.

### YOLOv8

Ultralytics continues to support YOLOv8 model loading through:

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
```

The Ultralytics Python API accepts images, arrays, video sources, webcams, and streams.

### Tracking

Ultralytics provides real-time tracking modes with configurable trackers.

Use that capability only as a **per-camera tracker baseline** unless testing proves a design for cross-camera identity.

### PyTorch

PyTorch installation is hardware-specific for CUDA.

Use the official install selector rather than embedding a guessed CUDA URL in the project documentation.

---

## 32. Reference URLs

Historical project source:

```text
ThesisV2/
AI_REPOSITORY_HANDOFF.md
Final manuscript selected by owner
```

Technical references:

```text
https://github.com/ShiqiYu/libfacedetection
https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/README.md
https://docs.ultralytics.com/models/yolov8
https://docs.ultralytics.com/usage/python
https://docs.ultralytics.com/modes/track
https://huggingface.co/Ultralytics/YOLOv8
https://pytorch.org/get-started/locally/
```

---

## 33. Working definition of "done"

The rebuild is not complete merely because bounding boxes appear.

Minimum thesis-level rebuild completion means:

```text
[ ] Three camera sources can operate concurrently
[ ] YOLOv8 detections are produced
[ ] YuNet face detections are produced
[ ] UI remains responsive
[ ] User can select a detected target
[ ] Target features are extracted
[ ] Local track continuity exists
[ ] Target can be matched on another camera
[ ] Global identity is independent of local tracker ID
[ ] Temporary target loss has defined behavior
[ ] Re-identification is measured, not assumed
[ ] Target can be saved and loaded
[ ] Uploaded image can initialize a target
[ ] SQLite records target/session events
[ ] Credentials are externalized
[ ] Dependencies are reproducible
[ ] Offline tests exist
[ ] Hardware acceptance tests exist
[ ] Three-camera performance is measured end-to-end
[ ] Manuscript-to-source/rebuild traceability is maintained
```

The final rebuild should be able to state clearly:

> This capability is historical and source-verified.

or:

> This capability was reconstructed from the manuscript requirement and is verified by the rebuild tests.

Those statements must never be conflated.
