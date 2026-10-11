# Model Store

Model weights are intentionally excluded from Git. `manifest.yaml` is the versioned registry for artifacts used by the rebuild.

Before adding a model declaration:

1. Record the model's upstream source and license.
2. Copy the artifact deliberately into the matching subdirectory under `models/`.
3. Compute and record its SHA-256 digest.
4. Keep the manifest path relative to this model store.
5. Validate the model independently in the implementation gate that introduces its adapter.

Historical thesis weights are evidence, not runtime dependencies. Reuse requires the same provenance, license, hash, copy, and independent-test process.
