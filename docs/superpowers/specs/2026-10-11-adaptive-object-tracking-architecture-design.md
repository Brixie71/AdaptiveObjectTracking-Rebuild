# Adaptive Object Tracking Rebuild Architecture Design

**Status:** Approved

**Date:** 2026-10-11

**Project:** AdaptiveObjectTracking-Rebuild

## 1. Purpose

This specification defines the first production-oriented architecture for the Adaptive Object Tracking rebuild. The application is a Windows desktop system that detects and tracks people and configurable generic object classes across multiple cameras. It combines per-camera tracking with cross-camera identity resolution, supports live and offline sources, and preserves a clear evidence trail for thesis evaluation.

The design replaces the historical prototype with testable module boundaries while retaining the thesis intent. Historical materials under `docs/reference/` are evidence only and are not runtime dependencies.

## 2. Goals

The first release shall:

- Run as a locally installable PyQt5 desktop application on Windows 10 or 11 with Python 3.11.
- Support USB cameras, RTSP streams, and offline video through one capture interface.
- Detect both people and user-allowed generic object classes.
- Maintain isolated local tracking state for every camera.
- Resolve identities across cameras for people and generic objects.
- Use separate person ReID and generic-object embedding encoders.
- Apply an optional face cue only to person identities.
- Provide confidence-aware automatic matching and human review.
- Support named, testable hardware profiles without storing credentials.
- Persist configuration, targets, sessions, events, decisions, and diagnostics locally.
- Remain responsive through bounded queues and off-GUI-thread inference.
- Operate on a compatible GPU when available and degrade safely to CPU.

## 3. Non-goals

The first release will not:

- Distribute inference across networked camera nodes or cloud services.
- Provide multi-user accounts, remote administration, or internet synchronization.
- Record full video by default.
- Treat face recognition as the primary person identity mechanism.
- Compare embeddings across incompatible object classes or encoder spaces.
- Promise a fixed recognition-accuracy target before a validation dataset establishes an empirical baseline.

## 4. Architectural Approach

The application will be a modular desktop application. Modules run within one installation and communicate through explicit interfaces and immutable or narrowly mutable data models. Capture and compute-intensive work run in bounded workers; inference never runs on the GUI thread.

This approach is preferred over local microservices or distributed camera nodes because it minimizes deployment and synchronization complexity while preserving interfaces that can later be moved into separate processes.

### 4.1 Major modules

| Module | Responsibility |
| --- | --- |
| UI | PyQt5 views, commands, status presentation, and review workflows |
| Capture | USB, RTSP, and offline-video adapters behind one source contract |
| Detection | Object detection and configured class-allowlist filtering |
| Tracking | Independent short-term motion tracking for each camera |
| Embedding | Person ReID, generic visual embeddings, and optional person-only face evidence |
| Identity | Cross-camera candidate search, scoring, decision states, merge, and split |
| Persistence | SQLite repositories, migrations, transactions, and managed-media references |
| Runtime | Worker lifecycle, bounded queues, scheduling, shutdown, and health state |
| Diagnostics | Timing, throughput, resource, device, model, and dropped-frame metrics |
| Configuration | Validated settings, environment-based secrets, and named hardware profiles |

Each module will expose an interface that states its inputs, outputs, errors, and lifecycle. UI code will not directly open cameras, invoke models, or issue raw SQL.

## 5. Runtime Data Flow

1. A selected hardware profile is validated and activated.
2. Each configured source captures frames into its own bounded newest-frame queue.
3. The detector emits observations containing source, frame, timestamp, class, confidence, and bounding box.
4. The active allowlist discards disabled classes before downstream processing.
5. A tracker owned by that camera assigns local track IDs and maintains motion state.
6. Stable tracks with sufficient visual quality are routed to an encoder:
   - People use the person ReID encoder and may receive an optional face cue.
   - Other allowed classes use the generic-object encoder.
7. The identity resolver searches only compatible class and encoder galleries.
8. Matching combines appearance, observation time, track quality, and optional camera-topology constraints.
9. The resolver applies the configured confidence policy:
   - High confidence links automatically to an existing global identity.
   - Medium confidence creates an ambiguous review item with ranked candidates.
   - Low confidence creates a new session-scoped global identity.
10. Events and identity decisions are persisted and sent to the UI.
11. The renderer displays the most recent processed state without waiting for every captured frame.

Queues must have explicit capacity and overflow policy. The normal overflow policy is to discard the oldest unprocessed frame, increment the relevant dropped-frame metric, and preserve recent-state responsiveness.

## 6. Identity Model

### 6.1 Identity scopes

Every allow-listed detection is eligible for automatic session-scoped cross-camera identity resolution. A session identity exists only for the current tracking session unless a user promotes it.

A user may promote a session identity into a named persistent target. Promotion stores approved reference images and target metadata. Persistent targets remain available to later sessions until archived or deleted by an explicit user action.

### 6.2 Dual encoders

Person and generic-object embeddings are separate feature spaces with independent models, quality gates, galleries, distance functions where required, and configurable thresholds. Embeddings from the two spaces are never directly compared.

Person matching may incorporate an optional face cue when a suitable face observation exists. This cue may strengthen or weaken a candidate score but cannot independently override incompatible person ReID, time, topology, or quality evidence.

Generic-object matching compares exact visual instances within compatible detector classes. It does not imply that visually identical commodity objects can always be uniquely distinguished; uncertain cases enter review or receive separate temporary identities.

### 6.3 Human decisions

Users can confirm an ambiguous candidate, create a new identity, merge identities, or split observations assigned to one identity. These actions append audit records describing the actor-visible action, time, affected records, and previous relationship. Original observations are not silently rewritten or destroyed.

### 6.4 Camera topology

A hardware profile may assign source locations, adjacency relationships, and plausible minimum and maximum travel times. The identity resolver uses topology as supporting or excluding evidence. When topology is absent, matching falls back to appearance, timestamps, and track quality.

## 7. Configuration and Hardware Profiles

Settings will cover:

- Camera and video sources.
- Detector and encoder selections.
- Per-class allowlist state.
- Detection, tracking, quality, and identity thresholds.
- Camera topology.
- Retention and managed-media policy.
- Preferred compute device and performance limits.
- Diagnostic and display options.

Named hardware profiles group compatible runtime settings for a physical or demonstration environment. Example profiles include `Laptop Webcams`, `Thesis CCTV Lab`, and `Offline Demo`.

A profile may contain:

- Stable source identifier and display name.
- Source type.
- USB index, RTSP environment-variable key, or offline-video reference.
- Requested resolution, frame rate, orientation, and supported capture properties.
- Preferred compute device.
- Compatible detector and encoder configuration.
- Camera topology.
- Last successful test timestamp.
- Non-sensitive benchmark and compatibility results.

Profiles are validated before activation and applied transactionally. If validation or activation fails, the current working configuration remains active. Profiles may be duplicated, tested independently, imported, and exported in sanitized form.

Real RTSP URLs containing credentials, usernames, passwords, or tokens are never stored in profiles, SQLite, logs, or diagnostic exports. Profiles reference environment-variable keys whose values are resolved only at connection time.

## 8. Settings: Hardware Testing Module

Hardware Testing is a dedicated Settings section that uses the same capture and model interfaces as live tracking. It will provide:

- USB camera enumeration and connection testing.
- RTSP connection testing with redacted source presentation.
- Offline-video readability and decode testing.
- Live preview for successful sources.
- Negotiated resolution and frame-rate reporting.
- Capture latency and dropped-frame measurements.
- CPU and RAM reporting.
- CUDA, GPU, and VRAM availability and compatibility reporting.
- Detector and encoder loading tests.
- Short, repeatable inference benchmarks.
- Sanitized diagnostic-report export.
- Saving a successful result into a new or existing named hardware profile.

A test result distinguishes unsupported, failed, degraded, and successful states. A profile can only be marked successfully tested when its required sources and configured models have completed their applicable checks.

## 9. Persistence and Media Ownership

SQLite is the local system of record. The persistence layer owns migrations, foreign-key enforcement, WAL configuration, prepared operations, and serialized writes.

The conceptual schema includes:

- `hardware_profiles`
- `camera_sources`
- `camera_topology`
- `application_settings`
- `tracking_sessions`
- `session_configuration_snapshots`
- `local_tracks`
- `global_identities`
- `persistent_targets`
- `target_reference_images`
- `tracking_events`
- `identity_candidates`
- `identity_decisions`
- `identity_audit_log`
- `hardware_test_runs`

Exact columns and indexes will be defined in the implementation plan and migrations, without changing the ownership described here.

Large binary media is not embedded in SQLite. The database stores metadata, hashes where useful, managed paths, retention state, and deletion state. Crops and recordings live under managed data directories.

By default, the application stores structured event metadata but not crops or full video. Users may explicitly enable image-crop or video retention. When retained media expires or is deleted, its event remains and its media reference is marked expired or deleted.

## 10. User Interface

### 10.1 Live Tracking

The primary view is a multi-camera grid. Each tile exposes source health and may show object class, local track ID, global identity label, match state, and optional performance metrics. Degraded modes remain visible without blocking other sources.

### 10.2 Identity Review

Medium-confidence matches appear in a review queue. The review presents the new observation beside ranked candidates with source, time, class, match evidence, and available history. The user can confirm a candidate or create a new identity.

### 10.3 Targets

The Targets view supports creating, naming, inspecting, archiving, merging, and splitting persistent targets, plus managing approved reference images.

### 10.4 Sessions and Events

The session timeline supports filters for camera, class, identity, confidence, and event type. Retained media appears only when enabled and still available.

### 10.5 Settings

Settings groups sources, hardware testing, profiles, models, encoders, allowlists, thresholds, topology, retention, compute, and diagnostics. Destructive actions such as deleting media or splitting identities require confirmation.

## 11. Reliability and Error Handling

- A source failure affects only that source; other sources and the UI continue operating.
- USB and RTSP sources reconnect using bounded exponential backoff with visible state.
- Offline end-of-file is a normal state and is distinct from decode failure.
- Queue overflow drops stale frames and records metrics instead of accumulating latency.
- Compatible GPU initialization or runtime failure triggers controlled CPU fallback with a visible degraded-performance warning.
- An incompatible model blocks only its affected pipeline and produces an actionable diagnostic.
- Profile activation is transactional and cannot replace a working profile with a failed one.
- Database writes use transactions. Recoverable failures may retry; unrecoverable failures disable new event recording while leaving safe live preview available.
- Shutdown stops producers, completes required event writes, releases devices, and closes SQLite cleanly.
- Structured logs include source, session, track, and identity context while redacting credentials and sensitive source values.

## 12. Performance and Observability

The initial performance target uses three active 640x480 sources:

- Supported GPU mode: at least 15 processed and displayed frames per second per camera.
- Supported GPU mode: end-to-end p95 latency no greater than 250 ms.
- CPU fallback: functional and stable, with measured results but no initial 15 FPS guarantee.
- UI: responsive during capture and inference, without unbounded queues or memory growth.

Metrics will distinguish capture, detection, tracking, embedding, identity matching, rendering, end-to-end latency, queue depth, and dropped frames. Aggregate FPS alone is insufficient for acceptance.

## 13. Security and Privacy

- Secrets are supplied through environment variables and are never committed.
- RTSP values and credentials are redacted from logs, errors, profiles, reports, screenshots produced by diagnostics, and exports.
- Media retention is opt-in.
- Profile exports include only non-sensitive configuration and metrics.
- Database and managed-media paths remain local unless the user explicitly exports them.
- Persistent target creation and identity correction require explicit user actions and remain auditable.

## 14. Testing Strategy

### 14.1 Automated tests

- Unit tests cover configuration validation, allowlists, queue overflow, scoring, confidence states, retention, profile activation, and secret redaction.
- Adapter contract tests require USB, RTSP, and offline sources to produce equivalent frame, timestamp, health, and error models.
- Deterministic recorded fixtures exercise detection, local tracking, both encoders, identity resolution, topology, and merge/split behavior.
- Integration tests run complete sessions using temporary databases and managed-media directories.
- UI tests cover profiles, hardware testing, identity review, settings, and failure-state presentation.
- Performance tests emit per-stage measurements and verify queue bounds and responsiveness.

### 14.2 Hardware acceptance

Manual or hardware-in-the-loop acceptance covers real USB, RTSP, offline-video, CPU, and available CUDA configurations. Results are stored as sanitized hardware test runs.

### 14.3 Functional acceptance

The first release is acceptable when:

- All three source types operate through the common interface.
- People and enabled generic object classes are detected and tracked locally.
- People and generic object instances can retain global identities across cameras.
- The class allowlist is configurable and enforced before downstream identity work.
- High-, medium-, and low-confidence outcomes follow the approved resolution policy.
- Users can promote temporary identities and audit merge or split decisions.
- Named hardware profiles can be tested, saved, reloaded, and exported safely.
- Default operation stores event metadata without retaining visual media.
- GPU and CPU behavior follow the approved performance and fallback rules.
- The UI remains responsive and runtime queues remain bounded.

Accuracy will be evaluated separately for person and generic-object encoders, and by object class where sample size permits. For the first release, the acceptance requirement is a reproducible validation dataset, documented protocol, and reported baseline rather than an invented minimum score. Any later minimum accuracy threshold requires a versioned specification change based on that evidence.

## 15. Implementation Boundaries

Implementation will follow the repository's staged gates. Environment, configuration, device selection, and logging precede camera and inference work. Each later subsystem must depend on interfaces established by the preceding gate, and each gate must add focused tests before the next begins.

The implementation plan must preserve the established Gate 0-11 sequence:

0. Historical evidence freeze and traceability baseline.
1. Reproducible environment, validated configuration, compute-device selection, and logging.
2. Common capture contract and USB, RTSP, and offline-video adapters; Hardware Testing begins with source tests.
3. Detection and configurable class-allowlist enforcement.
4. Responsive three-source PyQt5 display with bounded frame flow.
5. Target selection and reference-capture workflow.
6. Independent per-camera tracking.
7. Person and generic-object encoders with separate quality gates and thresholds.
8. Cross-camera global identity resolution, confidence states, topology, and review workflow.
9. Lost-target handling and recovery behavior.
10. SQLite persistence, persistent targets, retention-managed events, and saved named hardware profiles.
11. Performance instrumentation, hardware validation, and thesis evaluation acceptance.

Hardware Testing grows incrementally with the modules it validates: source checks begin at Gate 2, model and encoder checks appear with their respective gates, and successful named-profile persistence is completed at Gate 10.

No implementation work is authorized by this design document alone. A separate implementation plan must be written and reviewed after this specification is approved.
