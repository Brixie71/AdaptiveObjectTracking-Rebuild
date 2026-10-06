codex resume 01a1114d-d720-7df1-aa00-dac7abf7914c

# ThesisV2 Repository Handoff for Claude and ChatGPT

**Prepared:** 2026-10-06

**Repository:** `ThesisV2`

**Branch / revision inspected:** `main` at `469ddba` (`Update README.md`, 2025-01-18)

**Purpose:** Give an AI assistant enough verified context to review, explain, stabilize, and eventually improve this project without guessing which artifacts are authoritative.

## 1. Executive summary

This repository is a Windows-oriented Python computer-vision thesis prototype intended to detect, select, track, and re-identify people or objects across three camera feeds. Its intended workflow combines:

- YOLOv8 for person/object detection;
- YuNet through OpenCV `FaceDetectorYN` for face detection;
- PyQt5 for a multi-camera desktop interface;
- OpenCV for capture, frame processing, drawing, and display;
- PyTorch/CUDA for GPU acceleration;
- a small custom Roboflow person dataset plus a bundled COCO128 dataset;
- saved training and prediction outputs.

The final manuscript describes a much larger system than the visible source tree. According to the manuscript, the user selects a detected bounding box, the application extracts target features, matches the target across three USB or RTSP feeds, supports single- and multi-tracking modes, performs global re-identification, predicts temporarily lost targets, and can save, load, or upload a target. It also describes ResNet50-style feature extraction, SQLite persistence, Kalman prediction, and tracking-session history.

The visible repository does **not provide source-code evidence for most of those advanced capabilities**. Its accessible Python files primarily demonstrate simultaneous YOLO/YuNet detection, camera display, and incomplete UI controls. No visible implementation was found for the manuscript's ReID model, SQLite schema, global re-identification, Kalman integration, target persistence, or uploaded-image tracking. The archives include a standard SORT/Kalman tracker, but it is not integrated into the visible application code.

The strongest current interpretation is that this repository is an incomplete or earlier source snapshot of the system documented and demonstrated in the manuscript. It should be treated as a historical prototype/workbench until the actual final demonstration source is found. The tree contains several versions and copies of the application, trained models, datasets, generated results, compiled bytecode, and archives. There is no dependency manifest, automated test suite, installation guide, or unambiguous canonical entry point.

## 2. Evidence and limits of this review

Verified in this review:

- read `README.md`, all Python filenames/imports/definitions, dataset metadata, YAML configuration, and Git metadata;
- inspected the most likely UI/detection path in `MainDir/` and the major root-level experiments;
- built a deterministic AST inventory: **138 nodes and 267 relationships** from 23 detected code/config files;
- parsed/compiled the Python sources successfully as syntax validation;
- inventoried tracked files, model artifacts, datasets, generated runs, duplicates, and likely credentials;
- checked whether the currently active Python environment can locate the main third-party modules.
- read the complete DOCX manuscript body: 925 paragraphs, 24 tables, and the available figure captions and explanatory prose;
- inspected the manuscript's objectives, 21 development iterations, requirements, testing results, benchmarks, conclusions, and recommendations;
- searched the visible Python source and the `MainDir.zip`/`MainDir.7z` file listings and source contents for the advanced components claimed in the manuscript.

Not verified:

- the GUI was not launched;
- cameras and RTSP feeds were not contacted;
- model inference, CUDA execution, FPS, detection accuracy, and cross-camera behavior were not runtime-tested;
- the 15 generated AVI prediction files were not inspected or transcribed because the optional video-analysis dependency was unavailable;
- the manuscript's embedded screenshots and diagrams were not visually validated against the source or runtime application;
- archive contents were inspected read-only but were not assumed to be the authoritative final source;
- the three late-arriving PDF manuscript copies were inventoried but not content-extracted; they may duplicate or differ from the DOCX and require a dedicated comparison.

Any assistant should keep static evidence separate from runtime claims.

## 3. Intended thesis system versus visible implementation

### 3.1 Manuscript-defined objective

The thesis defines an **Adaptive Object Tracking and Following System** with supervised target selection. Its intended user flow is:

1. Initialize three USB webcams or IP CCTV/RTSP feeds.
2. Run YOLOv8 for object/person detection and YuNet for face detection.
3. Display all detections in multi-tracking mode.
4. Let the user click a bounding box and capture the selected target.
5. Extract visual features from the target.
6. Switch to single-tracking mode and compare the target features against detections from all three cameras.
7. Mark a match as locked and continue following it as it moves between camera views.
8. Attempt global re-identification or position prediction if the target disappears.
9. Save or load target information, or initialize tracking from an uploaded image.
10. Record targets, images, cameras, and tracking sessions in a database.

The manuscript describes the following conceptual components:

- a `ReIDModel` based on a pretrained ResNet50 feature extractor;
- feature-distance matching and unique person IDs;
- a per-camera tracker plus global cross-camera ReID;
- Kalman-filter prediction when a target is temporarily lost;
- SQLite tables for targets, images, cameras, and tracking sessions;
- target save/load and uploaded-image tracking;
- start/stop, disable target, tracking-mode switch, threshold, and camera controls.

These are **manuscript claims and requirements**, not verified descriptions of the checked-in source.

### 3.2 Visible repository implementation

The most complete application candidate is under `MainDir/`:

```text
MainDir/MainUITest1.py
  -> imports MainDir/YoloV8.py
  -> creates a PyQt5 window with three camera tabs
  -> each tab creates CameraWidget(camera_index)

MainDir/YoloV8.py
  -> opens the camera with OpenCV VideoCapture
  -> loads yolov8n.pt using Ultralytics
  -> loads YuNet_model.onnx using the local YuNet wrapper
  -> runs both detectors on timer-driven frames
  -> annotates YOLO and YuNet detections
  -> converts frames into QImage/QPixmap for the PyQt5 UI

MainDir/yunet.py
  -> thin wrapper around cv2.FaceDetectorYN
```

This visible path performs simultaneous **detection**, not the complete end-to-end system described in the manuscript. The top-level scripts and `Yunet/` directory look like experiments that led toward the `MainDir/` UI.

`MainDir.zip` and `MainDir.7z` contain additional historical material, including a standard SORT implementation with Kalman boxes and MOT sample data. However, the archived main UI/detector code still does not implement the manuscript's ResNet50 ReID, SQLite persistence, target save/load, uploaded-image flow, or global cross-camera identity layer.

## 4. Repository structure

```text
ThesisV2/
|-- README.md                         Project overview and known limitations
|-- AI_REPOSITORY_HANDOFF.md          This AI review handoff
|-- FINAL MANUSCRIPT ... .docx        Untracked thesis manuscript (late arrival)
|-- FINAL MANUSCRIPT ... .pdf         Untracked PDF manuscript copy (late arrival)
|-- FINAL MANUSCRIPT ... (1).pdf      Untracked PDF manuscript copy (late arrival)
|-- FINAL MANUSCRIPT ... (2).pdf      Untracked PDF manuscript copy (late arrival)
|
|-- MainDir/                          Strongest candidate for the final UI version
|   |-- MainUITest1.py                Three-tab PyQt UI and unfinished controls
|   |-- YoloV8.py                     YOLO + YuNet frame-processing widget
|   |-- yunet.py                      OpenCV YuNet wrapper
|   `-- __pycache__/                  Checked-in generated Python bytecode
|
|-- MainDir copy/                     Alternate/older UI variant
|   |-- MainUITest1.py
|   |-- YoloV8.py                     Byte-identical to MainDir/YoloV8.py
|   |-- yunet.py
|   `-- __pycache__/
|
|-- Yunet/                            Standalone YuNet/YOLO experiments
|   |-- ImprovedEight.py              Detection/selection experiment
|   |-- yunet.py                      Another YuNet wrapper copy
|   |-- demo.cpp                      OpenCV YuNet C++ demo
|   |-- CMakeLists.txt
|   |-- remove_initializer_from_input.py
|   `-- yolov8n.pt
|
|-- datasets/                         Training and evaluation data
|   |-- train/, valid/, test/         16-image Roboflow custom person dataset
|   |-- coco128/                      Bundled COCO128 sample dataset
|   |-- data.yaml                     YOLO dataset configuration
|   |-- README.dataset.txt
|   `-- README.roboflow.txt
|
|-- runs/detect/                      Generated Ultralytics outputs
|   |-- train/, train2/               Training metrics, plots, weights, events
|   `-- predict*, predict2*...        15 saved AVI prediction runs
|
|-- MainUITest1.py                    Simpler one-camera PyQt UI
|-- MainUITest1 copy.py               Larger alternate UI/detection experiment
|-- YoloV8.py                         Root one-camera YOLO/PyQt experiment
|-- YOLOX.py                          Ultralytics camera inference experiment
|-- YOLOX copy.py                     ONNX-style experiment; references YoloV8
|-- run.py                            Standalone keyboard-controlled YuNet/YOLO flow
|-- mouse hays.py                     Mouse-selection experiment; stale imports
|-- demo.py                           Standalone YuNet demo
|-- start.py                          PyQt start-screen prototype
|-- SingleGPUandCPUTrain.py           Minimal Ultralytics training script
|-- conv.py                           ONNX dynamic-quantization helper
|-- yunet.py                          Root YuNet wrapper
|
|-- *.pt                              PyTorch/Ultralytics weights
|-- *.onnx                            ONNX YOLO/YuNet models
|-- data.yaml                         Duplicate dataset configuration
|-- *.png                             UI/reference images
|-- MainDir.zip, MainDir.7z           Archived source snapshots
|-- CCSStuff.v1i.yolov8.zip           Roboflow dataset archive
`-- __pycache__/                      Checked-in generated Python bytecode
```

## 5. Important files and likely roles

| Path | Likely role | Current observation |
|---|---|---|
| `README.md` | High-level thesis description | Explains intent and performance constraints, but has no setup/run procedure. |
| `FINAL MANUSCRIPT PAST TENSE (Revised 4.0).docx` | Primary requirements and results source | Describes the intended final system, 21 development iterations, test results, benchmarks, and conclusions. Its implementation claims must be checked against source. |
| `MainDir/MainUITest1.py` | Likely final UI candidate | Creates three local camera widgets and six control buttons. Several controls are incomplete or incorrectly wired. |
| `MainDir/YoloV8.py` | Likely main inference component | Runs YOLOv8 and YuNet per frame in a PyQt timer callback. Strongly coupled to global models and CUDA assumptions. |
| `MainDir/yunet.py` | Face detector adapter | Wraps `cv2.FaceDetectorYN.create()` and exposes `infer()`. |
| `MainUITest1.py` | Simpler root UI | Creates one active camera tab; other camera tabs are commented out. |
| `run.py` | Standalone detector/selection experiment | Switches between YuNet and a custom YOLOv8 ONNX wrapper, but imports a lowercase `yolov8` module that is not present in the repository. |
| `YOLOX.py` | Camera inference experiment | Uses Ultralytics YOLO; contains embedded camera connection information that must be removed/rotated. |
| `SingleGPUandCPUTrain.py` | Training experiment | Loads several YOLO variants sequentially and starts 100 epochs against `data.yaml`; no CLI, logging policy, or reproducibility controls. |
| `conv.py` | Model utility | Quantizes `model.onnx` to `quantized_model.onnx` using ONNX Runtime. |
| `datasets/data.yaml` / `data.yaml` | Training dataset config | Both declare one class, `Person`; relative paths need validation because they do not visibly align with the checked-in directory layout. |
| `runs/detect/train*` | Training evidence | Contains weights, metrics, plots, event logs, and training arguments from two runs. |
| `MainDir.zip` / `MainDir.7z` | Historical source archives | Include SORT/Kalman code and older UI/detection variants, but no verified implementation of the complete manuscript-defined system. |

## 6. Data and model inventory

- Git tracks **406 files**.
- Dominant tracked types: 164 JPG, 148 TXT, 21 Python, 17 PNG, 15 AVI, 11 PYC, 8 PT, and 5 ONNX files.
- The custom Roboflow export says it contains **16 images**, one class (`Person`), resized to 640x640 with no augmentation, licensed CC BY 4.0.
- `datasets/coco128/` is also bundled, bringing a large generic sample dataset into the same repository.
- Model artifacts include YOLO `.pt` weights, YOLO `.onnx` exports, YuNet ONNX models, and training-run best/last checkpoints.
- The largest root models are roughly 6-13 MB each. Archives duplicate additional source/data.

These assets make the repository self-contained in places, but also make source review and version ownership difficult. Model provenance, intended production model, expected hashes, metrics, and licensing should be documented explicitly.

## 7. Dependency and environment state

Imports indicate these primary dependencies:

- Python 3.x;
- OpenCV (`cv2`);
- NumPy;
- PyTorch with CUDA support;
- Ultralytics;
- Supervision;
- PyQt5;
- ONNX and ONNX Runtime.

The inspected machine currently exposes all of those modules. Observed versions include OpenCV 4.10.0.84, PyTorch 2.2.0+cu118, Ultralytics 8.2.22, Supervision 0.2.1, PyQt5 5.15.10, ONNX 1.16.1, and ONNX Runtime 1.18.0.

However, the repository has no `requirements.txt`, `pyproject.toml`, Conda environment, or lock file. Those local versions are **not** a reproducible project specification. Some source calls are version-sensitive, including the older `sv.Detections.from_yolov8(...)` API.

### Manuscript files

Four untracked manuscript files appeared in the workspace after the initial repository scan: one DOCX (about 33 MB) and three PDFs (about 7.7 MB each). The complete DOCX body and tables were reviewed read-only. It is currently the richest requirements source and defines user-supervised bounding-box selection, object and face detection, feature matching, and cross-camera re-identification across three USB and/or RTSP feeds, with Windows as the implementation target.

The document also records known limitations involving lighting, face orientation or coverings, limited training data, camera-setting consistency, resource degradation, and controlled-environment evaluation. The DOCX and three PDFs have not been reconciled. An assistant should not assume that the newest filename, timestamp, or file type is authoritative; compare them and ask the owner which manuscript is final.

## 8. Current state assessment

### What is present

- A functioning-looking PyQt/OpenCV frame-processing design.
- Three-camera tab construction in the `MainDir` UI.
- YOLOv8 object/person detection integration.
- YuNet face detection integration.
- CUDA-aware code paths and saved GPU-era experiments.
- User-interface concepts for single tracking, multi tracking, target selection, target lock, disengage, logs, and camera status.
- A custom annotated person dataset and saved training outputs.
- A standalone SORT/Kalman implementation inside the source archives, although no integration into the visible application was verified.

### What the manuscript claims was implemented

- Twenty-one incremental development iterations.
- Three-camera USB and RTSP operation.
- Mouse/bounding-box target selection.
- Combined YOLOv8 object and YuNet face detection.
- ResNet50-style feature extraction and feature-distance matching.
- Cross-frame and cross-camera person re-identification.
- Per-camera multi-object tracking and single-target tracking.
- Kalman prediction and global re-identification for lost targets.
- SQLite CRUD operations and persistent targets/tracking sessions.
- Target save/load and uploaded-image target selection.
- Start/stop, disable-target, tracking-mode, threshold, and camera controls.

None of the advanced items above should be marked repository-verified until their actual source is located and exercised.

### What appears incomplete or ambiguous

- No single documented entry point. `MainDir/MainUITest1.py` is the strongest candidate, but this is inferred.
- The intended RTSP camera URLs are declared in the UI but not passed to `CameraWidget`; the active widgets use local indices `1`, `2`, and `0`.
- The control callbacks in `MainDir/MainUITest1.py` are nested functions, several accept or use `self` even though no owning controller object is supplied, and several only print text.
- The “Camera Logs” box is not connected to detector events.
- No persistent tracker, subject identity representation, or cross-camera handoff mechanism is evident.
- No visible `ReIDModel`, ResNet50 feature-extraction pipeline, SQLite access, database schema, global ReID routine, Kalman integration, target save/load function, or image-upload tracker was found.
- Several root experiments refer to missing or stale module names (for example lowercase/custom `yolov8` and `Algorithms`).
- Module import itself performs heavy work in places: argument parsing, model loading, CUDA inspection, and device setup occur at top level.
- CUDA device detail calls occur even after a CPU fallback message, so CPU-only startup may still fail.
- Multiple copies make it unclear where fixes should be applied.
- Dataset paths and model paths are relative to the process working directory rather than resolved from each source/config file.

### Repository hygiene

- Generated `.pyc` files are tracked.
- Training outputs, prediction videos, models, datasets, and archives are committed beside source.
- No `.gitignore` was found.
- No automated tests or test configuration were found.
- No CI configuration, packaging metadata, or release procedure was found.
- The latest source-bearing commits are from April 2024; the two January 2025 commits update only the README.
- At final verification, the four manuscript files were untracked alongside the pre-existing `.vscode/` directory; treat all of them as user-owned work.

### Manuscript testing and reported results

The manuscript reports the following black-box results:

| Function | Reported result |
|---|---:|
| Select/capture a target from Camera 1 | 28/30 |
| Select/capture a target from Camera 2 | 28/30 |
| Select/capture a target from Camera 3 | 28/30 |
| YOLOv8 object detection | 30/30 |
| YuNet face detection | 30/30 |
| Feature extraction | 30/30 |
| Feature matching across three cameras | 30/30 |
| Global re-identification | 30/30 |

The most important adverse result is reported in the surrounding discussion: when camera resolution or other camera settings differed, global re-identification succeeded only **3/30** times. The manuscript concludes that the three cameras needed matching resolution and image settings. This supports the README's warning that lighting, brightness, exposure, contrast, hue, and ISO changes can break tracking.

Reported individual YOLO model benchmarks vary significantly by export format and hardware. Examples include approximately 22.94 FPS for ONNX on battery-powered CPU testing, 39.38 FPS for ONNX on a charging CPU, and more than 100 FPS for some GPU formats. These are isolated model benchmarks and must not be treated as combined three-camera application throughput.

The JMeter/OS Process Sampler section reports average process runtimes near 58-60 seconds and zero execution errors, but also describes a major throughput collapse during sustained testing. The manuscript recommends investigating memory leaks, resource exhaustion, caching, and resource pooling. These results have not been reproduced from raw test files in this repository.

### Manuscript technical consistency issues

Future academic or engineering review should verify these points against primary sources and raw outputs:

- YuNet is repeatedly labeled “Fast R-CNN,” but the implementation uses OpenCV's anchor-free `FaceDetectorYN`; YuNet is not Fast R-CNN.
- YOLOv8 is associated with earlier YOLO authors in places even though the implementation is the Ultralytics package.
- `mAP50-95` values such as `0.4911` are described as `0.49%`; the conventional interpretation is approximately **49.11%**, not 0.49%.
- Some input-resolution descriptions conflate a square `imgsz=640` benchmark with 640x480 video.
- Several benchmark rows and narrative values appear inconsistent and should be checked against the original terminal output.
- The reported JMeter throughput units do not obviously reconcile with the number and duration of process executions.
- Some bibliography entries are placeholders, mismatched, or unrelated to the claim they appear to support.
- The conclusion says the advanced system was successfully implemented, while its recommendations propose adding a database that earlier chapters say already exists.

## 9. High-priority risks

### Critical: embedded camera credentials

Plaintext camera connection strings with usernames/passwords appear in multiple Python files, including `MainDir/MainUITest1.py`, `YOLOX.py`, and `MainUITest1 copy.py`. Treat those credentials as compromised:

1. rotate them on the camera/NVR side;
2. remove them from current source;
3. load secrets from environment variables or a local ignored config file;
4. consider purging them from Git history if this repository has ever been shared;
5. never reproduce the current secret values in AI prompts, issues, or documentation.

### High: manuscript/source mismatch

The manuscript describes and evaluates an advanced ReID/database/tracking system that is absent from the visible source. The final demonstrated source may be missing, stored elsewhere, or represented only in screenshots. Rebuilding before locating it risks replacing historical work with a new implementation and making the evidence trail less clear.

Detection boxes on independent feeds do not establish that the same person is followed across cameras. The repository needs source-backed acceptance tests for target selection, within-camera identity continuity, cross-camera handoff, and recovery after target loss.

### High: reported results need source evidence

The manuscript's success counts and benchmark interpretations are not backed by an automated test suite or clearly linked raw artifacts in the current tree. Preserve the claims as historical evidence, but do not treat them as currently reproducible.

### High: startup fragility

Global argument parsing, model loading, CUDA calls, working-directory-relative paths, and hardware access happen during imports. This makes testing and CPU fallback difficult.

### Medium: unclear source of truth

`MainDir`, `MainDir copy`, root scripts, archives, and near-duplicate detector wrappers can receive inconsistent fixes.

### Medium: unreproducible environment

There is no pinned dependency set, documented CUDA/toolkit requirement, model manifest, or verified run command.

## 10. Suggested stabilization sequence

Do not begin with a broad rewrite. Preserve the historical prototype and establish an executable baseline first.

1. **Security checkpoint**
   - Rotate and externalize camera credentials.
   - Add a safe `.env.example` or documented config schema without real secrets.

2. **Choose the source of truth**
   - Search other computers, drives, cloud folders, branches, group-member repositories, and archives for the final demonstrated implementation.
   - Confirm whether `MainDir/MainUITest1.py` is only an early version or the intended final application.
   - Mark other scripts as experiments/archive before deleting or moving anything.

3. **Build a manuscript-to-source traceability matrix**
   - Map each functional requirement and iteration to its implementing file, class, function, and test evidence.
   - Mark every item as verified, partial, missing, or manuscript-only.
   - Link benchmark claims to raw commands, hardware, model hashes, and output files.

4. **Reproduce the current application**
   - Document Python, CUDA, GPU, camera, and package requirements.
   - Create a pinned dependency file.
   - Add one verified launch command from the repository root.
   - Validate model and camera paths on a clean environment.

5. **Separate responsibilities**
   - Configuration: cameras, models, thresholds, device.
   - Capture: local camera/RTSP lifecycle and reconnect behavior.
   - Detection: YOLO and YuNet adapters.
   - Tracking/ReID: explicit identity state and cross-camera association.
   - UI: display and user commands only.

6. **Build a testable core**
   - Move side effects out of module imports.
   - Add unit tests for config, frame transforms, selection state, and detector result adaptation.
   - Add an offline smoke test using a recorded video or fixed images before camera tests.

7. **Define thesis acceptance criteria**
   - What qualifies as the same person across cameras?
   - What user intervention is required?
   - What are the minimum FPS, latency, precision/recall, and recovery targets?
   - How should lighting changes and temporary occlusion be handled?

8. **Clean the repository after behavior is preserved**
   - Add `.gitignore`.
   - Separate source, tests, config, models, datasets, runs, and archive material.
   - Remove duplicate files only after hashes, provenance, and runtime equivalence are confirmed.

## 11. Open questions for the project owner

1. Where is the exact source version shown in the manuscript's actual-testing screenshots?
2. Is `MainDir/MainUITest1.py` the version that was demonstrated, or an earlier detector-only snapshot?
3. Should the application consume three RTSP cameras, three local capture devices, or support both?
4. Was ResNet50 actually used for ReID, and where are its source, weights, and feature-distance threshold?
5. Where is the SQLite database or schema described in the manuscript?
6. Which YOLO weight is authoritative: `yolov8n.pt`, root `best.pt`, or a `runs/detect/train*/weights/best.pt` file?
7. Which dataset trained the intended production model: the 16-image custom set, COCO128, or another dataset not present here?
8. Are the saved `runs/` results and manuscript screenshots thesis evidence that must remain unchanged?
9. Is CPU operation required, or is an NVIDIA CUDA GPU a hard requirement?
10. Are the current camera credentials still active, and has the repository ever been public or shared?

## 12. Instructions for the next Claude/ChatGPT review

Before modifying code:

- read this report, `README.md`, and the confirmed final manuscript;
- inspect `git status` and preserve existing user changes;
- do not print or repeat embedded credentials;
- ask the owner to locate or confirm the exact source used for the manuscript's actual testing;
- treat manuscript-described capabilities as requirements/claims until matched to source and runtime evidence;
- distinguish source, generated output, models, datasets, and archives;
- propose a staged plan with an acceptance gate before cleanup or restructuring;
- avoid claiming runtime success from syntax/static checks;
- do not delete duplicate folders, archives, runs, datasets, or models without explicit approval.

A useful first task is not implementation: create a **manuscript-to-source traceability matrix** and locate the final demonstrated source. After that source is identified and preserved, the first implementation task should be a non-behavior-changing reproducibility pass: externalized configuration, pinned dependencies, a safe launch script, path normalization, CPU/GPU capability checks, and an offline smoke test.

## 13. Bottom line

The manuscript describes a substantially more capable system than the visible repository contains. The repository preserves useful detector experiments, models, datasets, training output, UI prototypes, and a standalone SORT tracker, but it does not currently substantiate the manuscript's final ReID/database/tracking implementation.

The highest-value next action is to locate and preserve the exact final source used for the thesis demonstration. Only then should the team establish a reproducible environment, secure configuration, requirement-to-code traceability, and runtime acceptance tests. If the final source cannot be recovered, the manuscript should be treated as the requirements and historical evidence for a staged reconstruction—not as proof that the checked-in code already implements those capabilities.
