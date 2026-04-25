# AI Development Guide

This repo is safe for AI-assisted development when agents stay inside deterministic, testable lanes.

## Core Rule

AI can propose, compare, and explain candidates. The production converter should only promote a candidate when deterministic validation proves it is better and still printable.

## Good Agent Tasks

- Add focused conversion heuristics with before/after reports.
- Add validation checks for Bambu-compatible grouped OBJ/MTL and 3MF outputs.
- Improve docs, examples, tests, and report readability.
- Run benchmark samples and summarize what changed.
- Add candidate lanes that fail closed when they do not improve the baseline.

## Risky Agent Tasks

- Changing color semantics without tests on real duck-like samples.
- Committing generated model outputs from `outputs/`.
- Introducing provider or model calls without a candidate report and rollback path.
- Making DuckAgent workflow or marketplace decisions inside this repo.
- Treating a visually nice preview as print-ready without Bambu validation and component metrics.

## Standard Verification

```bash
PYTHONPATH=src python -m compileall -q src
PYTHONPATH=src python -m pytest
```

For a real repaired-production smoke run:

```bash
PYTHONPATH=src python -m color3dconverter.cli convert-repaired-production \
  /path/to/source.glb \
  --out-dir /tmp/paint_to_print_smoke \
  --max-colors 8 \
  --repair-backend voxel_marching_cubes \
  --repair-voxel-divisions 128 \
  --repair-smoothing-iterations 18
```

## Candidate Lane Contract

Every candidate lane should write:

- a machine-readable JSON report
- a human-readable summary when operator judgment is expected
- before/after metrics
- validation status
- clear promotion or rejection reasons
- artifact paths for preview, grouped OBJ/MTL, 3MF, palette, and face-label arrays

## Current Candidate Lanes

- Same-mesh production candidate selection.
- Provider-baked repaired same-mesh reduction.
- Repaired-geometry transfer.
- Deterministic paint-region cleanup for tiny-island absorption.
- Lane chooser report for comparing ready lanes without mutating assets.

## Adding AI/Provider Cleanup Later

An AI cleanup lane should be provider-optional and report-only at first:

1. Run the deterministic baseline.
2. Run deterministic paint-region cleanup.
3. Run the provider/AI candidate only if baseline remains noisy.
4. Compare candidates with the same Bambu validation and paint-intent checks.
5. Promote only if the candidate reduces fragmentation and preserves protected details.

Do not let provider output become the canonical source of truth without the same report gates as local output.
