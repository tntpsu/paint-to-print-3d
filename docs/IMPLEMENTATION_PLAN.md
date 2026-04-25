# paint-to-print-3d Implementation Plan

## Goal

Build a reusable public package that converts textured 3D assets into cleaner printable color assets that can be consumed by Bambu-style multicolor workflows and called from DuckAgent.

## Product Boundaries

This package should own:
- loading textured geometry
- deriving printable color regions
- face/material assignment
- standards-based export primitives
- conversion reports

DuckAgent should own:
- run orchestration
- UI, previews, and approvals
- marketplace/operator workflow
- higher-level heuristics around which export lane to use

## Source Provenance

Primary source logic comes from:
- `/Users/philtullai/ai-agents/3dcolor`
- `/Users/philtullai/ai-agents/duckAgent`

Port only the clean, reusable pieces. Do not mirror every experimental script.

## Phase 1: Public Foundation

Status: in progress

Deliverables:
- repo scaffold
- README
- AGENTS guidance
- provenance notes
- package skeleton
- tests
- CI workflow

Core reusable modules:
- `model_io.py`
- `regions.py`
- `export_obj.py`
- `export_3mf.py`
- `pipeline.py`
- `cli.py`

## Phase 2: Region-First Conversion

Status: complete

Goal:
- move from raw texture colors toward stable printable regions

Key steps:
1. segment texture pixels into a limited set of color regions
2. clean those regions morphologically
3. assign dominant region labels to faces via UV coordinates
4. export grouped OBJ/MTL and `colorgroup` 3MF

Success criteria:
- stable region counts
- predictable face/material grouping
- imports with clearly mappable colors in Bambu Studio

## Phase 3: Geometry-Aware Cleanup

Status: active

Goal:
- reduce speckle and accidental patchiness that purely image-based segmentation can introduce

Key steps:
1. face adjacency graph
2. small-island merging
3. normal-aware region boundaries
4. optional multipart output by connected region
5. texture-region transfer for repaired-geometry workflows

Focused execution plan for the current hard problem:
- see [REGION_TRANSFER_SUCCESS_PLAN.md](/Users/philtullai/ai-agents/paint-to-print-3d/docs/REGION_TRANSFER_SUCCESS_PLAN.md)
- see [COLOR_CONVERSION_FINISH_PLAN.md](/Users/philtullai/ai-agents/paint-to-print-3d/docs/COLOR_CONVERSION_FINISH_PLAN.md)

## Phase 4: Broader Input Support

Status: active

Add:
- GLB/GLTF support
- embedded texture extraction
- repaired/source geometry transfer helpers

This phase should borrow selectively from DuckAgent’s local Bambu bridge.

## Phase 5: Validation + Comparison

Status: active

Add:
- rendered export previews
- source-vs-export comparison report
- synthetic probes for importer validation
- export inspection helpers

## Phase 6: DuckAgent Integration

Status: active

Deliverables:
- clean Python API callable from DuckAgent
- stable output contract
- packaged converter record for run artifacts

Planned integration point:
- DuckAgent local Bambu/export lane should call this repo instead of carrying the whole conversion stack inline

Detailed integration plan:
- see [DUCKAGENT_INTEGRATION_PLAN.md](/Users/philtullai/ai-agents/paint-to-print-3d/docs/DUCKAGENT_INTEGRATION_PLAN.md)

## Near-Term Build Order

1. Finish Phase 1 scaffold
2. Make region-first textured OBJ conversion reliable
3. Add grouped OBJ/MTL and `colorgroup` 3MF tests
4. Keep GLB support stable while adding loaded-mesh conversion entry points
5. Add preview and comparison rendering
6. Integrate the package into DuckAgent with a bridge + legacy fallback
7. Make the transferred-color repaired-geometry path use the package on real DuckAgent runs
8. Reduce remaining legacy conversion code in DuckAgent after the package path is stable

## Open Questions

- Should multipart 3MF be object-split, face-color, or both?
- Should material OBJ export stay a first-class path for Bambu cleanup?
- How much printer/Bambu-specific behavior belongs here versus in DuckAgent?
- Should part-aware duck segmentation live here as a generic strategy or remain a DuckAgent-specific higher-level wrapper?

## Current Best Bet

Based on the current evidence:

- keep the legacy single-source path as the source-region oracle
- stop focusing on generic label transfer
- make repaired-geometry success depend on region transfer, boundary cleanup, and target-geometry island pruning

The detailed success criteria and build order for that lane are in:
- [REGION_TRANSFER_SUCCESS_PLAN.md](/Users/philtullai/ai-agents/paint-to-print-3d/docs/REGION_TRANSFER_SUCCESS_PLAN.md)
- [COLOR_CONVERSION_FINISH_PLAN.md](/Users/philtullai/ai-agents/paint-to-print-3d/docs/COLOR_CONVERSION_FINISH_PLAN.md)
