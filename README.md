# 3dcolorconverter

Convert textured 3D models into cleaner, printable color regions for Bambu-style multicolor workflows.

The package focuses on one problem:

`textured mesh -> cleaner printable paint regions -> grouped export assets`

This repo was carved out of two internal codebases:
- [3dcolor](/Users/philtullai/ai-agents/3dcolor), which explored texture baking, region segmentation, and grouped OBJ exports
- [duckAgent](/Users/philtullai/ai-agents/duckAgent), which added standards-based `colorgroup` 3MF export, validation probes, and operator-facing preview tooling

## What It Does Today

`3dcolorconverter` currently supports:
- textured OBJ input with `MTL` + `map_Kd` texture resolution
- textured GLB input with embedded texture extraction
- OBJ ZIP input for packaged OBJ/MTL/texture bundles
- texture-region segmentation with morphological cleanup
- geometry-aware transferred-color conversion for repaired-geometry workflows
- grouped OBJ/MTL export
- standards-based face-color 3MF export using `colorgroup`
- export previews, comparison artifacts, and validation reports

## Supported Inputs

| Input | Status | Notes |
| --- | --- | --- |
| `.obj` + `.mtl` + texture | Supported | Primary explicit mesh path |
| `.zip` / `.objzip` bundle | Supported | Extracts the first textured OBJ bundle found |
| `.glb` | Supported | Reads embedded texture and mesh data |
| already-loaded mesh payload | Supported in Python API | Used by DuckAgent for repaired-geometry flows |

## Quick Start

Convert a textured OBJ, OBJ ZIP, or GLB into grouped OBJ/MTL + face-color 3MF:

```bash
python -m color3dconverter.cli convert-model \
  /path/to/model.glb \
  --regions 8 \
  --out-dir /path/to/output
```

Run the production same-mesh converter. This uses the stable `legacy_fast_face_labels` core, tries a fixed candidate set, compares each candidate against an internally rendered source preview, and only marks the result production-ready when the drift stays under the threshold:

```bash
PYTHONPATH=/Users/philtullai/ai-agents/3dcolorconverter/src \
python -m color3dconverter.cli convert-production \
  /path/to/model.glb \
  --out-dir /path/to/output \
  --quality-threshold 0.02
```

Run the repaired-geometry transfer lane when you already have a textured source model plus a repaired target mesh. The target mesh can be an untextured OBJ; the source model supplies the color regions:

```bash
PYTHONPATH=/Users/philtullai/ai-agents/3dcolorconverter/src \
python -m color3dconverter.cli convert-repaired-transfer \
  /path/to/source.glb \
  /path/to/repaired_target.obj \
  --out-dir /path/to/output \
  --max-colors 12 \
  --strategy legacy_fast_face_labels
```

Convert a packaged OBJ ZIP:

```bash
python -m color3dconverter.cli convert-model \
  /path/to/model_bundle.zip \
  --regions 6 \
  --out-dir /path/to/output
```

Run a fixed-strategy ablation study across multiple real source variants:

```bash
PYTHONPATH=/Users/philtullai/ai-agents/3dcolorconverter/src \
python -m color3dconverter.cli real-ablation \
  --config /Users/philtullai/ai-agents/duckAgent/creative_agent/runtime/examples/bold_cowgirl_ablation.json \
  --out-dir /tmp/bold_cowgirl_ablation
```

Optional overrides let you keep the same config but swap the converter profile:

```bash
PYTHONPATH=/Users/philtullai/ai-agents/3dcolorconverter/src \
python -m color3dconverter.cli real-ablation \
  --config /Users/philtullai/ai-agents/duckAgent/creative_agent/runtime/examples/bold_cowgirl_ablation.json \
  --strategy legacy_fast_face_labels \
  --regions 8 \
  --out-dir /tmp/bold_cowgirl_ablation
```

Run a bounded iterative search that keeps exploring candidates until the metric target is hit or the search stalls:

```bash
PYTHONPATH=/Users/philtullai/ai-agents/3dcolorconverter/src \
python -m color3dconverter.cli iterative-search \
  --config /Users/philtullai/ai-agents/duckAgent/creative_agent/runtime/examples/cowgirl_original_iterative_search.json \
  --out-dir /tmp/cowgirl_original_iterative_search
```

Run a cross-case search that only rewards candidates which stay strong across multiple real assets:

```bash
PYTHONPATH=/Users/philtullai/ai-agents/3dcolorconverter/src \
python -m color3dconverter.cli cross-case-search \
  --config /Users/philtullai/ai-agents/duckAgent/creative_agent/runtime/examples/same_mesh_cross_case_search.json \
  --out-dir /tmp/same_mesh_cross_case_search
```

Run the compact acceptance check for the current best same-mesh rule (`posterize_4` + `legacy_fast_face_labels`):

```bash
PYTHONPATH=/Users/philtullai/ai-agents/3dcolorconverter/src \
python -m color3dconverter.cli cross-case-search \
  --config /Users/philtullai/ai-agents/duckAgent/creative_agent/runtime/examples/same_mesh_posterize4_acceptance.json \
  --out-dir /tmp/same_mesh_posterize4_acceptance
```

Train a repaired-geometry shading model from provider source/target pairs:

```bash
PYTHONPATH=/Users/philtullai/ai-agents/3dcolorconverter/src \
python -m color3dconverter.cli train-shading-model \
  --config /Users/philtullai/ai-agents/3dcolorconverter/examples/eight_pair_shading_model_config.json \
  --out-model /tmp/eight_pair_direct_rgb_model_et.pkl \
  --model-kind et \
  --target-kind direct_rgb \
  --sample-size 10000
```

Two larger repaired configs are also available:
- duck-focused: [/Users/philtullai/ai-agents/3dcolorconverter/examples/duck_ten_shading_model_config.json](/Users/philtullai/ai-agents/3dcolorconverter/examples/duck_ten_shading_model_config.json)
- broader figurine mix: [/Users/philtullai/ai-agents/3dcolorconverter/examples/all_twelve_shading_model_config.json](/Users/philtullai/ai-agents/3dcolorconverter/examples/all_twelve_shading_model_config.json)

Bundle multiple repaired shading models into a weighted ensemble:

```bash
PYTHONPATH=/Users/philtullai/ai-agents/3dcolorconverter/src \
python -m color3dconverter.cli bundle-shading-models \
  --out-model /tmp/repaired_ensemble.pkl \
  --model-path /tmp/duck10_et.pkl \
  --model-path /tmp/all12_et.pkl \
  --weights 0.5 0.5
```

Apply a trained repaired shading model to a raw textured source plus repaired target geometry:

```bash
PYTHONPATH=/Users/philtullai/ai-agents/3dcolorconverter/src \
python -m color3dconverter.cli convert-shading-model \
  /Users/philtullai/Downloads/Screenshot_2026_04_02_at_85543_Captain_America_themed_rubber_duck_figurine_Prism_30_Multi_Image_da2242c4.glb \
  /Users/philtullai/Downloads/Repaired_Captain_America_themed_rubber_duck_figuri_Mesh_Repair_04ef643e.obj \
  --model-path /tmp/eight_pair_direct_rgb_model_et.pkl \
  --alignment-json /private/tmp/captain_provider_oracle_v1/alignment_summary.json \
  --out-obj /tmp/eight_pair_predicted.obj
```

## Output Bundle

A typical conversion writes:
- `region_materials.obj`
- `region_materials.mtl`
- `region_materials_vertex_color.obj`
- `region_colorgroup.3mf`
- `region_preview.png`
- `palette_swatches.png`
- `palette.csv`
- `conversion_report.json`

Validation bundles can also include:
- `local_bambu_source_comparison.png`
- `local_bambu_validation_report.json`
- `local_bambu_validation_report.md`

Real ablation bundles also include:
- `ablation_summary.json`
- `ablation_summary.md`
- `ablation_board.png`

Iterative search bundles also include:
- `iterative_summary.json`
- `iterative_summary.md`
- `round_*/`

Cross-case search bundles also include:
- `cross_case_summary.json`
- `cross_case_summary.md`
- `cross_case_best_board.png`

Production conversion bundles also include:
- `production_report.json`
- `selected/`
- `_candidates/`

Shading-model training bundles also include:
- `<model>.pkl`
- `<model>.pkl.json`

Shading-model repaired conversions also include:
- `<output>.obj`
- `<output>.json`

## Python API

```python
from color3dconverter import convert_model_to_color_assets, run_production_conversion

report = convert_model_to_color_assets(
    "/path/to/model.glb",
    out_dir="/path/to/output",
    n_regions=8,
    object_name="Cowgirl Duck",
)

print(report["threemf_path"])
print(report["report_path"])

production = run_production_conversion(
    "/path/to/model.glb",
    out_dir="/path/to/output_production",
    quality_threshold=0.02,
)

print(production["ready_for_production"])
print(production["selected_candidate"]["selected_dir"])
```

## Production Scope

The production wrapper is intentionally narrow:
- same-mesh textured input only
- supported formats: `.obj`, `.zip` / `.objzip`, `.glb`
- internal candidate set:
  - `baseline_r16`
  - `posterize4_r16`
  - `posterize4_r8`
  - `baseline_r5`
- fixed conversion core: `legacy_fast_face_labels`

It does **not** claim production readiness for:
- repaired-geometry color transfer
- cross-topology baking
- semantic part transfer between different meshes

The repaired shading-model lane is currently a research path:
- it uses provider-pair supervision
- the current best shared recipe is an ExtraTrees direct-RGB model trained on multiple aligned provider pairs with 10k samples per pair
- it is promising for raw textured mesh -> repaired colored OBJ reproduction
- it is not yet claimed as production-ready

The `convert-repaired-transfer` command is the explicit evaluation bridge for that harder lane:
- it writes grouped OBJ/MTL and colorgroup 3MF assets from source art plus repaired target geometry
- it records `conversion_lane: repaired_geometry_region_transfer` in the report
- it is still gated by visual review and benchmark evidence before DuckAgent should treat it as automatic production output

## How DuckAgent Uses It

DuckAgent calls this package for:
- single-image GLB conversion
- repaired GLB conversion
- repaired-geometry plus transferred-color conversion
- local validation bundles for Bambu review

DuckAgent still owns:
- run orchestration
- operator UI
- approval workflows
- choosing which source asset to feed into the converter

## Development

Install in editable mode:

```bash
pip install -e .
```

Run tests:

```bash
pytest
```

## Roadmap

Near-term priorities:
- stronger paint-zone extraction for character parts like hat, body, and beak
- multipart export experiments
- more importer probes and example assets
- README examples that demonstrate repaired-geometry transfer workflows

## Repo Docs

- [Examples](/Users/philtullai/ai-agents/3dcolorconverter/examples/README.md)
- [Implementation Plan](/Users/philtullai/ai-agents/3dcolorconverter/docs/IMPLEMENTATION_PLAN.md)
- [DuckAgent Integration Plan](/Users/philtullai/ai-agents/3dcolorconverter/docs/DUCKAGENT_INTEGRATION_PLAN.md)
- [Provenance](/Users/philtullai/ai-agents/3dcolorconverter/docs/PROVENANCE.md)
- [AGENTS](/Users/philtullai/ai-agents/3dcolorconverter/AGENTS.md)
