# DuckAgent Handoff Contract

## Purpose

`build-duckagent-handoff` is the stable bridge from an approved textured model to DuckAgent's printable-model workflow.

It keeps conversion correctness inside `paint-to-print-3d`, while giving DuckAgent a small contract it can read without importing converter internals.

## Command

```bash
PYTHONPATH=src python -m color3dconverter.cli build-duckagent-handoff \
  /path/to/source.glb \
  --out-dir /path/to/duckagent_run/paint_to_print \
  --object-name "Monster Truck Duck" \
  --max-colors 8 \
  --repair-backend voxel_marching_cubes \
  --repair-voxel-divisions 128 \
  --repair-smoothing-iterations 18
```

## Stable Outputs

DuckAgent should read:

- `handoff_manifest.json`

DuckAgent can attach these stable artifact keys from the manifest:

- `bambu_3mf_path`
- `grouped_obj_path`
- `grouped_mtl_path`
- `export_preview_path`
- `palette_swatches_path`
- `qa_board_path`
- `handoff_markdown_path`

The lower-level conversion reports remain available for debugging:

- `production_report_path`
- `acceptance_summary_path`
- `conversion_report_path`
- `paint_intent_report_path`

## Readiness Rule

DuckAgent should only treat the bundle as ready when:

```json
{
  "ready_for_duckagent_handoff": true
}
```

If this is false, DuckAgent should show the QA board and failed gates to the operator instead of treating the model as print-ready.

## Required Gates

The handoff currently gates on:

- required files exist: OBJ, MTL, 3MF, preview, palette swatches
- Bambu material validation passes
- repaired transfer policy is ready for automatic use
- duck-style flat bottom support is preserved
- palette size is within the requested color range
- paint regions are not over-fragmented

Advisory gates may be added without breaking DuckAgent. New required gates should be treated as a contract change.

## Boundary

`paint-to-print-3d` owns:

- mesh loading
- local repair
- smoothing
- paint transfer
- material export
- QA artifacts
- readiness gates

DuckAgent owns:

- deciding which approved model to submit
- storing the handoff under the creative run
- showing the QA board and failed gates
- operator approval
- any publishing or product workflow that uses the printable bundle
