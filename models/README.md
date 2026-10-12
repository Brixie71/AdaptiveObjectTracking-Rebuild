# Model Store

Model weights are intentionally excluded from Git. `manifest.yaml` is the versioned registry for artifacts used by the rebuild.

Before adding a model declaration:

1. Record the model's upstream source and license.
2. Copy the artifact deliberately into the matching subdirectory under `models/`.
3. Compute and record its SHA-256 digest.
4. Keep the manifest path relative to this model store.
5. Validate the model independently in the implementation gate that introduces its adapter.

Historical thesis weights are evidence, not runtime dependencies. Reuse requires the same provenance, license, hash, copy, and independent-test process.

## Gate 3 detector provisioning

Choose a YOLOv8-compatible artifact whose source and license you have reviewed.
Do not point runtime configuration into `docs/reference/`, and do not rely on
Ultralytics auto-download during tests. Copy the approved file deliberately:

```powershell
$approvedDetector = 'C:\path\to\approved\yolov8n.pt'
Copy-Item -LiteralPath $approvedDetector -Destination 'models\yolo\yolov8n.pt'
$digest = (Get-FileHash -LiteralPath 'models\yolo\yolov8n.pt' -Algorithm SHA256).Hash.ToLowerInvariant()
$digest
```

Record the actual upstream source, reviewed license, and printed digest in
`models/manifest.yaml`:

```yaml
schema_version: 1
models:
  - model_id: yolov8n-coco
    role: detector
    path: yolo/yolov8n.pt
    source: owner-reviewed upstream source and license
    sha256: replace-with-the-printed-64-character-lowercase-digest
    required: true
```

Set `detection.enabled: true` and `detection.model_id: yolov8n-coco` in an
ignored local application configuration, then run:

```powershell
python scripts/probe_detector.py --app-config config/app.local.yaml --model-manifest models/manifest.yaml --models-dir models --json
```

Do not commit the weight file, a fabricated digest, an unreviewed download URL,
or probe output containing local paths. Gate 3 automated acceptance remains
fake-backed; only a successful opt-in probe records real-model load/inference
compatibility, and accuracy requires a later labeled evaluation dataset.
