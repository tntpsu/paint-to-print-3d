# Repaired Transfer Research And Rollout Plan

## Goal

Ship an explainable DuckAgent lane that turns AI-generated textured duck models into Bambu-friendly multicolor assets without pretending local repaired-geometry transfer is solved before it is.

The acceptance criterion is stricter than picking a lane: produce a repaired OBJ/MTL whose geometry is printable in Bambu Studio and whose colors are accurate enough to trust.

The production target is a lane chooser, not one magic algorithm:

- prefer provider-baked or same-mesh output when it preserves paint regions better
- run local repaired transfer as a candidate lane with quality gates
- reject noisy repaired transfers before they become the default printable asset
- tell the operator when the concept art itself needs stronger print-friendly color separation

## Research Findings

### 3D AI Studio already exposes the missing provider primitive

3D AI Studio's mesh repair endpoint does more than repair geometry. Its documented repair path remeshes the model and can bake original textures onto the new mesh with `bake_textures`, which defaults to true. It supports `texture_resolution`, `topology`, repair quality, hollowing, target height, and multiple output formats.

Source: [3D AI Studio Mesh Repair API](https://www.3daistudio.com/Platform/API/Documentation/tools/repair)

3D AI Studio also has a separate bake-texture endpoint. It transfers textures from an original textured GLB/glTF onto a retopologized target mesh, generates UVs, aligns/scales the target, and uses ray casting. This is the closest documented equivalent to what we need for a source-to-repaired topology transfer.

Source: [3D AI Studio Bake Texture API](https://www.3daistudio.com/Platform/API/Documentation/tools/bake-texture)

### The local converter should not compete with provider bake first

The first question should be:

- did the provider already create a repaired mesh with baked texture data that is good enough to use as an oracle?

If yes, our job becomes:

- convert the baked visual signal into Bambu-friendly grouped colors
- reduce texture/color noise into a practical filament palette
- preserve large intentional parts
- export grouped OBJ/MTL and standards-based 3MF colorgroup assets

That is easier and safer than trying to infer all topology correspondence locally from scratch.

### Bambu-friendly output still needs simplification

Bambu Studio has public release history around improved colored OBJ import, and 3MF has a formal materials/properties extension for full color and multi-material definitions. In practice, Bambu import is sensitive to how colors are represented, so our safest deliverables remain:

- grouped OBJ/MTL with one material per printable color region
- vertex-color OBJ as an inspection fallback
- 3MF colorgroup output as a standards-based package

Sources:

- [Bambu Studio 1.9.1 release](https://github.com/bambulab/BambuStudio/releases/tag/v01.09.01.66)
- [3MF Specification](https://3mf.io/spec/)

### Texture baking is the industry-shaped answer

The broader graphics workflow for different source/target topology is high-poly-to-low-poly texture baking:

- align source and target
- generate/prepare UVs on the target
- project or ray-cast source texture/material signal onto target
- then simplify the resulting texture/colors for the downstream use case

That means our local repaired-transfer lane should be treated as a fallback/experiment until it beats the provider-baked or same-mesh oracle on visual quality.

## Ownership

`paint-to-print-3d` owns:

- same-mesh conversion
- provider-baked model conversion
- local repaired-transfer candidate conversion
- quality assessment and lane-choice metrics
- Bambu asset export

`duckAgent` owns:

- collecting source, provider repair, and provider bake artifacts
- calling `paint-to-print-3d`
- recording chosen and rejected lanes in run artifacts
- exposing the result in approval emails or the creative console

`duck-ops` owns:

- health/reporting on conversion failures
- operator summaries when a lane is rejected
- future promotion tracking when the lane chooser proves reliable

## Dependencies

Required:

- 3D AI Studio repair/bake outputs from DuckAgent runs
- current same-mesh export path
- current `convert-repaired-transfer` path
- conversion reports with palette, component, island, and preview metrics

Nice-to-have:

- retained provider repair/bake result JSON and downloaded asset paths
- Bambu Studio smoke-import validation
- a small benchmark set with known-good same-mesh and known-bad repaired-transfer cases

Blockers:

- no automatic production preference for local repaired transfer until it passes quality gates
- no claim that a provider-baked texture is printable until it is reduced to practical filament colors

## Phased Plan

### Phase 1: Provider-Bake Oracle

Add a lane that inspects/downloads the provider repair result when `bake_textures=true` was used, or calls the 3D AI Studio bake-texture endpoint when we have both original and repaired assets.

Output:

- baked provider GLB/OBJ artifact path
- source metadata
- preview image
- conversion report

Exit condition:

- we can compare provider-baked output to same-mesh and local repaired-transfer output on the same duck.

### Phase 2: Bambu Conversion From Provider Bake

Convert the provider-baked model into practical printable regions.

Implementation shape:

- sample baked texture or vertex/material colors
- posterize/quantize to a target filament palette
- merge tiny islands
- write grouped OBJ/MTL and 3MF colorgroup

Exit condition:

- provider-baked cowgirl/duck output opens as a manageable palette instead of a texture dump.

Current implementation:

- `convert_provider_baked_model_to_assets(...)`
- CLI: `python -m color3dconverter.cli convert-provider-bake`
- report field: `conversion_lane: provider_baked_repaired_same_mesh`
- report field: `provider_bake_assessment`

### Phase 3: Repaired Transfer Assessment Gate

Attach an explicit assessment to every local repaired-transfer report.

Gate signals:

- target face count
- palette size
- connected component count
- tiny island count
- largest connected region share

Exit condition:

- noisy repaired transfers are marked `needs_review` and never become the default without a separate lane chooser.

### Phase 4: Lane Chooser

Compare candidate lanes:

- same-mesh oracle
- provider-bake conversion
- local repaired transfer

Prefer the first lane that is visually/structurally good enough. Do not prefer repaired geometry merely because it is repaired.

Exit condition:

- DuckAgent records chosen lane, rejected lane summaries, and operator-facing reason codes.

Current implementation:

- `choose_conversion_lane(...)`
- CLI: `python -m color3dconverter.cli choose-lane`
- report field: `mode: propose_only`
- report fields: `selected_lane`, `rejected_lanes`, `selection_policy`
- current policy order: same-mesh production, provider-bake, repaired-transfer
- current behavior: report-only; no asset mutation or publishing

### Phase 5: DuckAgent Integration

Wire the lane chooser into the concept-to-print workflow.

Required artifacts:

- chosen Bambu asset paths
- rejected lane reports
- preview comparison board
- recommendation if all lanes fail

Exit condition:

- the user can approve a concept knowing which conversion lane won and why.

## Review Gates

- `duck-change-planner`: before adding the DuckAgent integration contract.
- `duck-data-model-governance`: before adding new run artifact fields or report schemas.
- `duck-reliability-review`: before any scheduled/automatic lane chooser runs.
- `duck-ship-review`: before committing integration changes.

Testing gates:

- unit tests for repaired-transfer assessment thresholds
- regression test that known noisy repaired-transfer metrics return `needs_review`
- smoke run on one real DuckAgent concept artifact
- visual comparison board retained in the run output

Acceptance gates:

- repaired geometry is watertight
- repaired geometry has one printable body
- grouped OBJ/MTL exists with a practical filament palette
- source/export drift is below the configured threshold
- region fragmentation stays below the tiny-island and component-count thresholds

Rollout mode:

- Phase 1 and Phase 2: local/evaluation only
- Phase 3: report-only guardrail
- Phase 4: propose-only lane chooser
- Phase 5: operator-approved production lane

## Risks

- Provider bake may preserve visual texture but still produce too many tiny color islands for multicolor filament printing.
- Bambu import behavior differs between vertex colors, MTL colors, texture maps, and 3MF packages.
- Repaired meshes can be extremely large; running local transfer blindly can waste minutes and still produce bad output.
- Source concept art with muddy colors can defeat every conversion lane.

## Recommended First Slice

1. Keep local repaired transfer available as an evaluation bridge.
2. Add `repaired_transfer_assessment` to every repaired-transfer report.
3. Add a provider-bake oracle lane that consumes existing 3D AI Studio repaired/baked artifacts before making any new API calls.
4. Run same-mesh, provider-bake, and local repaired-transfer side by side on the bold cowgirl artifact.
5. Only after that, wire DuckAgent to choose a preferred printable lane.
