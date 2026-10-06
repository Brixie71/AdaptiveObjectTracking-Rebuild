# Reference Evidence Manifest

**Baseline date:** 2026-10-06

**Repository:** `AdaptiveObjectTracking-Rebuild`

**Purpose:** Identify and fingerprint the historical documents used to define and review the rebuild. This manifest records evidence; it does not convert manuscript claims into repository-verified behavior.

## Canonical requirements source

The canonical manuscript for rebuild requirements is:

```text
docs/reference/FINAL MANUSCRIPT PAST TENSE (Revised 4.0).docx
```

This selection follows the project owner's direction to use the final DOCX for thesis context. The PDF files remain preserved as comparison artifacts. If the owner later selects a different revision, update this manifest and the traceability matrix explicitly rather than silently replacing the canonical source.

## Evidence inventory

| File | Role | Bytes | SHA-256 |
|---|---|---:|---|
| `AI_REPOSITORY_HANDOFF.md` | Read-only review of the historical `ThesisV2` repository and manuscript/source discrepancy | 29,135 | `B892122A7FEBBCF2DF42A0D216BCCD2D7AD038994EBDE4A2829E1973D7AB5905` |
| `FINAL MANUSCRIPT PAST TENSE (Revised 4.0).docx` | Canonical manuscript and primary requirements source | 32,937,786 | `7B34FAB8A6F601FD49E4AF146B046237DDA3ED69E84499ACBB82F07B2886A045` |
| `FINAL MANUSCRIPT PAST TENSE (Revised 4.0).pdf` | PDF manuscript copy | 7,654,778 | `B0B46FBA454CB3E0AE29F81AD0D1A0F4912641CFD9FCEDAAC231FA8E2448959B` |
| `FINAL MANUSCRIPT PAST TENSE (Revised 4.0) (2).pdf` | Byte-identical duplicate of the plain PDF | 7,654,778 | `B0B46FBA454CB3E0AE29F81AD0D1A0F4912641CFD9FCEDAAC231FA8E2448959B` |
| `FINAL MANUSCRIPT PAST TENSE (Revised 4.0) (1).pdf` | Distinct PDF revision requiring comparison before being treated as equivalent | 7,781,917 | `E7D360488A741B7783673DF479DF731562856D7E1CFF3BA322DE8221FCACB290` |

## Duplicate and revision observations

- The plain PDF and `(2).pdf` have identical byte length and SHA-256. They are exact binary duplicates.
- `(1).pdf` has a different byte length and SHA-256. It is a distinct artifact, not merely another filename for the same bytes.
- The DOCX is the requirements authority for the current rebuild baseline. No claim is made that any PDF renders exactly the same content until a dedicated document comparison is completed.
- `AI_REPOSITORY_HANDOFF.md` summarizes prior inspection. It is secondary evidence and must not override the canonical manuscript or directly inspected source.

## Provenance and evidence boundary

- The project owner moved these files from the historical thesis workspace into `docs/reference/` on 2026-10-06.
- The historical repository was inspected at `ThesisV2` commit `469ddba` (`Update README.md`, 2025-01-18).
- The manuscripts were untracked in the historical working tree during inspection, so their hashes are not established by that Git commit.
- Historical source remains outside this rebuild repository and is read-only.
- Screenshots, diagrams, reported success rates, and benchmark narratives in the manuscript are historical claims until linked to source and reproducible evidence.

## Handling rules

1. Do not edit reference artifacts in place.
2. Do not rename, delete, or deduplicate the PDFs without explicit owner approval.
3. If a reference changes, record the new size and SHA-256 and explain why.
4. Do not import code or load runtime assets directly from the historical repository.
5. Cite the exact reference filename when adding a traceability entry.
6. Label conclusions as manuscript-described, historical-source-verified, rebuild-decided, or rebuild-verified.

## Reproducing the hashes

From the repository root in PowerShell:

```powershell
Get-ChildItem -LiteralPath docs\reference -File |
    Where-Object Name -ne '.gitkeep' |
    Get-FileHash -Algorithm SHA256
```
