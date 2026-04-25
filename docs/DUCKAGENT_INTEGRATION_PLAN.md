# DuckAgent Integration Plan

## Purpose

Make `paint-to-print-3d` the reusable conversion layer that DuckAgent calls for:
- local Bambu-friendly color exports
- standards-based `colorgroup` 3MF generation
- grouped OBJ/MTL material exports
- future source-vs-export validation

DuckAgent should stop owning most of the low-level conversion logic directly and instead become:
- the orchestrator
- the UI/reporting layer
- the approval/preview surface

## Current State

Today the relevant logic is split across:

- `/Users/philtullai/ai-agents/3dcolor`
  - textured OBJ loading
  - baking and quantization experiments
  - region/material conversion experiments

- `/Users/philtullai/ai-agents/duckAgent`
  - GLB extraction helpers
  - standards-based `colorgroup` 3MF writer
  - local export previews
  - comparison and manufacturing UI

This causes:
- duplicated logic
- hard-to-share code
- hard-to-test integration boundaries
- a public/private code mismatch

## Target Architecture

### `paint-to-print-3d` owns

- mesh + texture ingestion
- texture-region segmentation
- face-label assignment
- grouped OBJ/MTL export
- `colorgroup` 3MF export
- conversion report generation
- future preview/comparison generation

### DuckAgent owns

- concept run selection
- source-asset choice
- operator-facing explanations
- viewer buttons and progress state
- artifact wiring into run output folders
- validation display and business workflow decisions

## Integration Contract

DuckAgent should prefer the handoff API when a textured concept/model is ready to become a printable bundle:

```python
run_duckagent_handoff(
    source_path=...,
    out_dir=...,
    object_name=...,
    max_colors=8,
)
```

The handoff writes `handoff_manifest.json`, `handoff_qa_board.png`, `handoff_summary.md`, and the lower-level Bambu artifacts/reports. DuckAgent should depend on the manifest contract in [DuckAgent Handoff Contract](DUCKAGENT_HANDOFF_CONTRACT.md).

The lower-level conversion API remains available for narrower workflows:

```python
convert_model_to_color_assets(
    source_path=...,
    source_format=...,
    out_dir=...,
    max_colors=...,
    strategy=...,
    object_name=...,
)
```

Expected output contract:

```json
{
  "status": "ok",
  "source_path": "...",
  "source_format": "glb",
  "strategy": "region_first",
  "palette_size": 5,
  "obj_path": "...",
  "mtl_path": "...",
  "threemf_path": "...",
  "report_path": "...",
  "preview_path": "...",
  "comparison_path": "...",
  "notes": []
}
```

DuckAgent should only depend on this contract, not on converter internals.

## Input Support Plan

### Phase A: Textured OBJ

Status: started

Use cases:
- `OBJ + MTL + texture`
- explicit `--texture-path`

Why first:
- simplest public-facing input
- easiest to validate
- already implemented in the new repo

### Phase B: GLB / GLTF

Status: in progress

Port from DuckAgent:
- GLB chunk parsing
- accessor reading
- embedded image extraction
- UV + texture payload extraction

Reference source:
- [local_bambu_bridge.py](/Users/philtullai/ai-agents/duckAgent/creative_agent/runtime/src/duck_creative_agent/local_bambu_bridge.py)

Why next:
- DuckAgent’s real upstream asset is often GLB
- makes the package directly callable from current concept runs

### Phase C: OBJ ZIP

Status: planned

Support:
- unzip bundle
- find primary OBJ
- resolve MTL + textures
- pass through normal OBJ ingestion

Why:
- 3D AI Studio and similar tools often hand back OBJ ZIP bundles

## Conversion Strategy Plan

### Strategy 1: Region-First Face Assignment

Default strategy for initial integration.

Pipeline:
1. ingest mesh + UV + texture
2. cluster texture pixels into `N` regions
3. clean regions morphologically
4. assign each face to the dominant region seen through UVs
5. export grouped OBJ/MTL
6. export `colorgroup` face-color 3MF

Why:
- cleaner than raw triangle-color transfer
- closer to actual printable color zones
- easier for Bambu cleanup

### Strategy 2: Geometry-Aware Cleanup

Second pass after initial integration.

Add:
- face adjacency graph
- small-island merge
- normal-aware boundary preservation
- optional region merging by shared palette + connectivity

Why:
- reduces mottled paint behavior
- creates more cohesive printable parts

### Strategy 3: Source/Target Color Transfer

Later phase.

Goal:
- use repaired geometry with color borrowed from richer source textures

This should be added only after GLB support is stable.

## Export Plan

### Primary exports

- grouped OBJ/MTL
- face-color `colorgroup` 3MF

### Future exports

- multipart 3MF
- flat face-color OBJ variants
- comparison preview PNGs

## DuckAgent Migration Plan

### Step 1: Add package as local dependency

DuckAgent should be able to import the `color3dconverter` Python package from `paint-to-print-3d` in local development without copying files around.

Possible approaches:
- editable install in the shared venv
- git submodule/subtree
- workspace-relative import during development

Recommended first step:
- editable install into DuckAgent’s venv

### Step 2: Build a thin adapter in DuckAgent

Create a single adapter module in DuckAgent that:
- converts run metadata into converter inputs
- calls `paint-to-print-3d`
- maps the report back into DuckAgent artifact records

DuckAgent should not import multiple internal converter modules directly.

Current execution note:
- use `paint-to-print-3d` as the primary backend for single-source and simplified single-source local exports
- dual-source repaired-geometry + transferred-color runs now use the package too
- use `build-duckagent-handoff` / `run_duckagent_handoff` when DuckAgent needs an operator-facing printable-model bundle with gates
- keep the legacy DuckAgent exporter only as a safety fallback if the package path errors on a specific run

### Step 3: Keep viewer/output contract stable

Map converter outputs to DuckAgent artifact names like:
- `local_bambu_palette.3mf`
- `local_bambu_vertex_colors.obj`
- `local_bambu_export.json`

The viewer should not need a major rewrite for the first migration.

### Step 4: Dual-run validation mode

For a short migration period:
- run existing DuckAgent exporter
- run `paint-to-print-3d`
- compare reports/previews side by side

This gives us confidence before removing old code.

### Step 5: Switch primary export path

Once stable:
- `paint-to-print-3d` becomes the default conversion backend
- old inline conversion code becomes fallback/legacy

### Step 6: Remove duplicated low-level logic

After the new path is stable:
- delete or shrink the duplicated conversion internals in DuckAgent
- keep only GLB selection, preview wiring, and UX code in DuckAgent

## Validation Gates

Each phase should pass:

### Unit tests

- OBJ ingestion
- GLB ingestion
- face-label assignment
- grouped OBJ/MTL export
- `colorgroup` 3MF structure

### Integration tests

- sample OBJ -> outputs written
- sample GLB -> outputs written
- report contract shape
- handoff manifest exposes stable artifact keys
- handoff QA board and Markdown summary are written
- failed required gates keep `ready_for_duckagent_handoff` false

### DuckAgent smoke tests

- concept run can trigger conversion
- output files land in expected run directory
- viewer still reads artifacts correctly

### Human validation

- Bambu Studio imports the generated assets
- colors are mappable and cleanup-friendly
- source/export preview drift is understandable

## Near-Term Implementation Order

1. finalize current OBJ-based public package
2. port GLB ingestion from DuckAgent
3. add a unified `convert_model_to_color_assets` pipeline entry point
4. add preview/comparison outputs into `paint-to-print-3d`
5. create DuckAgent adapter module
6. run dual-output comparison on known concept runs
7. switch DuckAgent local Bambu lane to this package

## Risks

- GLB ingestion is more complex than OBJ due to embedded image/accessor parsing
- Bambu compatibility may still prefer multipart/material semantics in some cases
- too much DuckAgent-specific logic leaking into the public package would make the repo messy

## Guiding Rule

If logic is:
- about conversion correctness: it belongs in `paint-to-print-3d`
- about run orchestration, UI, or operator workflow: it belongs in DuckAgent
