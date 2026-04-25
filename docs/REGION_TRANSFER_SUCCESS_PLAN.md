# Region Transfer Success Plan

## Purpose

Turn the current textured-model conversion work into a path that can succeed predictably on real DuckAgent concepts, not just on one-off happy cases.

This plan is specifically about the remaining hard problem:

- preserving clean printable color regions
- while still using repaired geometry that is good enough for manufacturing

## Current Diagnosis

We have already eliminated several earlier unknowns.

### What is no longer the main blocker

- Not primarily `3MF` semantics.
  - `colorgroup` face-color export is working.
- Not primarily the Bambu import mechanism.
  - the standard `3MF` path now imports as colored.
- Not primarily geometry simplification.
  - unsimplified repaired geometry only improved the cowgirl result slightly.
- Not primarily “lack of a reusable package.”
  - `paint-to-print-3d` now owns the real conversion path and DuckAgent is calling it.

### What is actually failing now

The conversion loses intentional paint zones when moving from the richer source asset to repaired geometry.

More concretely:

- single-source legacy conversion produces cleaner region behavior than our repaired transfer path
- repaired-geometry transfer still turns those regions into mottled, smeared labels
- bold source art helps the concept stage, but the repaired transfer still breaks down

### Evidence we already have

1. The old Grinch path is reproducible.
- the old `3dcolor` script was rerun directly
- its saved “good” export was reproduced
- the package `legacy_fast_face_labels` path now matches that export structure and face distribution

2. The old export shape matters.
- simple `mat_*` grouped OBJ/MTL behaves well in Bambu
- this shape has already been pulled into `paint-to-print-3d`

3. Cowgirl failures are transfer failures more than extraction failures.
- single-source legacy looks cleaner than repaired transfer
- repaired transfer is where the result becomes blotchy

4. Provider-side OBJ ZIP is not reliable enough to be our only answer.
- it failed outright on the bold cowgirl run

## Success Criteria

We should not call this solved until all of these are true.

### A. Technical success

On test fixtures like Grinch and bold cowgirl:

- the converter preserves major intended regions such as body, hat, bandana, beak, boots, and dark accents
- repaired-geometry output keeps those regions recognizable without heavy speckle
- palette size stays in a practical range such as `5-10`
- tiny accidental islands are rare

### B. Visual success

For the same concept:

- the repaired-geometry export should look clearly closer to the concept image than the current transfer path
- the side-by-side should not show camouflage-style region breakup
- the exported preview should look like intentionally painted parts, not texture noise

### C. Workflow success

In DuckAgent:

- one concept run can go from concept image
- to 3D
- to repaired geometry
- to local Bambu export
- with one clearly preferred conversion lane

### D. Bambu success

When opened in Bambu Studio:

- colors are mapped as a manageable set of materials
- the regions are easy to inspect and remap
- the file feels like a cleanup-friendly starting point rather than a broken texture dump

## Working Theory

The winning architecture is:

1. extract clean paint regions on the source mesh using the legacy path
2. transfer region ownership to repaired geometry
3. smooth and clean boundaries on repaired geometry
4. export grouped OBJ/MTL and face-color `3MF`

The key change is step `2`.

Right now we mostly transfer quantized vertex labels.
That is too weak.

We need to transfer something closer to:

- region identity
- region confidence
- major-part ownership
- boundary-aware labels

not just local vertex color votes.

## Implementation Phases

## Phase 1: Lock the Oracles

Goal:
- make sure we know what “good” means before changing more code

Deliverables:

- preserve the existing Grinch legacy outputs as reference fixtures
- preserve the current bold cowgirl outputs as failure fixtures
- record expected region counts, palette distributions, and preview images

Files:

- `/Users/philtullai/ai-agents/3dcolor/examples/grinch/duck_conversion`
- `/Users/philtullai/ai-agents/paint-to-print-3d/examples/grinch_legacy_retest`
- `/Users/philtullai/ai-agents/duckAgent/creative_agent/runtime/runs/outputs/trend_concept_preview_20260408_211824`

Exit condition:

- we can compare any new algorithm against fixed good/bad reference cases quickly

## Phase 2: Source Region Extraction API

Goal:
- make the legacy source path expose stable reusable region data, not just exported files

Implement:

- extract source face labels
- connected region ids
- region palette metadata
- adjacency graph or region-neighbor metadata
- optional per-face confidence

Likely files:

- `/Users/philtullai/ai-agents/paint-to-print-3d/src/color3dconverter/pipeline.py`
- `/Users/philtullai/ai-agents/paint-to-print-3d/src/color3dconverter/face_regions.py`

New contract idea:

```python
extract_legacy_face_regions(source_mesh, *, max_regions=8) -> RegionTransferSource
```

Where `RegionTransferSource` contains:

- face labels
- connected components
- palette
- per-region face sets
- per-region centroids / normals / UV stats

Exit condition:

- we can inspect source regions directly before any repaired transfer happens

## Phase 3: Region-to-Geometry Transfer

Goal:
- transfer source regions onto repaired geometry without dissolving them

This is the core success phase.

Implement a new strategy that transfers:

- nearest source-face region
- weighted by geometric proximity
- and constrained by normal similarity
- with optional spatial smoothing on the repaired mesh

This should be face-region transfer, not just vertex-label transfer.

Possible algorithm:

1. sample centroids and normals for repaired faces
2. find candidate source faces or source regions using nearest-neighbor lookup
3. assign a provisional region id to each repaired face
4. run adjacency-based cleanup on repaired geometry
5. reassign tiny islands to dominant neighboring regions

Likely files:

- `/Users/philtullai/ai-agents/paint-to-print-3d/src/color3dconverter/face_regions.py`
- `/Users/philtullai/ai-agents/paint-to-print-3d/src/color3dconverter/pipeline.py`

New strategy name:

- `geometry_transfer_legacy_face_regions`

Exit condition:

- repaired geometry shows large coherent hat/body/bandana/beak regions on bold cowgirl

## Phase 4: Boundary Cleanup On Target Geometry

Goal:
- preserve large intentional regions and kill accidental islands

Implement:

- adjacency graph on repaired faces
- island pruning thresholds by area or face count
- neighbor-majority reassignment
- optional normal-aware edge protection so real part boundaries survive cleanup

Why this matters:

- even a good first transfer will still create freckles and ragged edges

Exit condition:

- target mesh regions look like paint parts, not noise

## Phase 5: Export Comparison Loop

Goal:
- compare repaired-region transfer against the best single-source legacy export

For each test concept:

1. single-source legacy export
2. repaired transfer export
3. side-by-side comparison image
4. palette distribution report

We should evaluate:

- body readability
- bandana integrity
- hat readability
- accent noise

Exit condition:

- repaired transfer is clearly closer to the single-source legacy look than today’s transfer path is

## Phase 6: DuckAgent Default Lane

Goal:
- once the new repaired-region transfer wins, make it the preferred local Bambu path

DuckAgent should prefer:

- `legacy single-source` for diagnosis and quick inspection
- `legacy face-region transfer to repaired geometry` for production candidate export

Files:

- `/Users/philtullai/ai-agents/duckAgent/creative_agent/runtime/src/duck_creative_agent/color3dconverter_bridge.py`

Exit condition:

- the default local export lane picks the new repaired-region strategy
- the old transferred-label path becomes fallback only

## Experiments We Should Run

These are the specific experiments that will tell us whether we are truly improving.

### Experiment 1: Grinch parity stays intact

Question:
- did we preserve the already-good single-source legacy behavior?

Pass:
- Grinch package output still matches the old structure and region layout closely

### Experiment 2: Bold cowgirl repaired transfer

Question:
- does repaired-region transfer preserve the cleaner bold cowgirl zones?

Pass:
- hat, bandana, body, beak, and boots are all still visually distinct and not mottled

### Experiment 3: Old cowgirl versus bold cowgirl

Question:
- how much of the problem is source art versus transfer logic?

Pass:
- bold cowgirl improves materially under the new repaired-region strategy

### Experiment 4: Provider OBJ ZIP as optional comparison only

Question:
- is provider OBJ ZIP ever better when it succeeds?

Pass:
- use it as a secondary reference lane, not a hard dependency

## What We Should Not Do

To stay successful, avoid these traps:

- do not keep layering generic global k-means heuristics on top of a broken transfer path
- do not treat provider OBJ ZIP as the primary plan, because it is unreliable
- do not optimize around the current bad render-preview score alone when the provider render itself is wrong
- do not remove the old legacy path until repaired-region transfer clearly beats the current repaired-label path

## How We Will Know We Are Winning

We are winning when:

- the repaired export stops looking camouflage-like
- the bold cowgirl keeps a large clean yellow body region
- the bandana stays mostly one strong red region
- the hat reads as a dominant tan part with a darker band instead of many brown fragments
- Bambu import feels like a manageable paint-map, not damage control

## Immediate Next Build

Current implementation status as of 2026-04-24:

- `geometry_transfer_legacy_face_regions` exists in `paint-to-print-3d`
- `convert-repaired-transfer` exists as a direct CLI/API bridge from textured source model plus repaired target geometry to grouped OBJ/MTL and colorgroup 3MF
- untextured repaired target OBJ input is supported by `load_geometry_model`
- reusable legacy source-region extraction exists via `SourceFaceRegionModel`
- region ownership transfer uses a KD-tree nearest-neighbor search instead of chunked all-pairs face matching
- bold cowgirl repaired-transfer bridge run completed successfully on the 1.17M-face repaired mesh, improving from about 145 seconds to about 90 seconds after the KD-tree change
- full local test suite passes with the bridge in place

Research update:

- 3D AI Studio's documented repair path can bake textures onto repaired/remeshed output, and its separate bake-texture API transfers source textures onto retopologized geometry with UV generation and ray casting.
- This means provider-baked output should become a first-class oracle lane before local repaired transfer is promoted.
- The rollout plan for that lane is captured in `docs/REPAIRED_TRANSFER_RESEARCH_AND_ROLLOUT_PLAN.md`.

The next implementation should be:

1. attach a repaired-transfer quality assessment to every `convert-repaired-transfer` report
2. inspect existing 3D AI Studio repair/bake artifacts and add a provider-bake oracle lane
3. run same-mesh, provider-bake, and local repaired-transfer lanes side by side on bold cowgirl
4. make DuckAgent prefer the repaired transfer lane only if it beats the provider-bake and same-mesh lanes under quality gates
5. extend `SourceFaceRegionModel` into corner-bake and blender-like source setup if those strategies become the winning candidates

That is the most direct path to success from the evidence we have today.
