# paint-to-print-3d

[![CI](https://github.com/tntpsu/paint-to-print-3d/actions/workflows/ci.yml/badge.svg)](https://github.com/tntpsu/paint-to-print-3d/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Convert textured OBJ/GLB models into cleaner Bambu-friendly multicolor 3D print assets.

The package focuses on one practical workflow:

`textured mesh -> cleaner printable paint regions -> grouped OBJ/MTL + 3MF colorgroup assets`

It is meant for makers and developers experimenting with AI-generated or texture-painted models that need to become more practical multicolor print inputs.

The project/package name is `paint-to-print-3d`; the Python import package remains `color3dconverter` for compatibility. The preferred CLI entry point is `paint-to-print-3d`, with `3dcolorconverter` kept as a legacy alias for now.

## Why It Exists

Many AI or texture-painted 3D models look good on screen but import poorly into slicers: too many tiny color islands, unclear material groups, or color data trapped in textures. This project turns that artwork into inspectable print assets with reports that explain whether the output is actually ready to try in Bambu Studio.

## What It Does Today

`paint-to-print-3d` currently supports:
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
PYTHONPATH=src python -m color3dconverter.cli convert-production \
  /path/to/model.glb \
  --out-dir /path/to/output \
  --quality-threshold 0.02
```

Run the repaired-geometry transfer lane when you already have a textured source model plus a repaired target mesh. The target mesh can be an untextured OBJ; the source model supplies the color regions:

```bash
PYTHONPATH=src python -m color3dconverter.cli convert-repaired-transfer \
  /path/to/source.glb \
  /path/to/repaired_target.obj \
  --out-dir /path/to/output \
  --max-colors 12 \
  --strategy legacy_fast_face_labels
```

Run the end-to-end repaired production lane. This repairs the source geometry locally, smooths it, transfers printable paint regions, optionally writes a deterministic paint-region cleanup candidate, and only promotes that cleanup when it improves island/component counts while still passing validation:

```bash
PYTHONPATH=src python -m color3dconverter.cli convert-repaired-production \
  /path/to/source.glb \
  --out-dir /path/to/output \
  --max-colors 8 \
  --repair-backend voxel_marching_cubes \
  --repair-voxel-divisions 128 \
  --repair-smoothing-iterations 18
```

Build the DuckAgent handoff bundle. This wraps the repaired production lane and adds a stable manifest, QA board, Markdown summary, and readiness gates that DuckAgent can consume without knowing converter internals:

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

Convert a packaged OBJ ZIP:

```bash
python -m color3dconverter.cli convert-model \
  /path/to/model_bundle.zip \
  --regions 6 \
  --out-dir /path/to/output
```

Run a fixed-strategy ablation study across multiple real source variants:

```bash
PYTHONPATH=src python -m color3dconverter.cli real-ablation \
  --config examples/your_ablation_config.json \
  --out-dir /tmp/bold_cowgirl_ablation
```

Optional overrides let you keep the same config but swap the converter profile:

```bash
PYTHONPATH=src python -m color3dconverter.cli real-ablation \
  --config examples/your_ablation_config.json \
  --strategy legacy_fast_face_labels \
  --regions 8 \
  --out-dir /tmp/bold_cowgirl_ablation
```

Run a bounded iterative search that keeps exploring candidates until the metric target is hit or the search stalls:

```bash
PYTHONPATH=src python -m color3dconverter.cli iterative-search \
  --config examples/your_iterative_search_config.json \
  --out-dir /tmp/cowgirl_original_iterative_search
```

Run a cross-case search that only rewards candidates which stay strong across multiple real assets:

```bash
PYTHONPATH=src python -m color3dconverter.cli cross-case-search \
  --config examples/your_cross_case_search_config.json \
  --out-dir /tmp/same_mesh_cross_case_search
```

Run the compact acceptance check for the current best same-mesh rule (`posterize_4` + `legacy_fast_face_labels`):

```bash
PYTHONPATH=src python -m color3dconverter.cli cross-case-search \
  --config examples/your_acceptance_config.json \
  --out-dir /tmp/same_mesh_posterize4_acceptance
```

Train a repaired-geometry shading model from provider source/target pairs:

```bash
PYTHONPATH=src python -m color3dconverter.cli train-shading-model \
  --config examples/eight_pair_shading_model_config.json \
  --out-model /tmp/eight_pair_direct_rgb_model_et.pkl \
  --model-kind et \
  --target-kind direct_rgb \
  --sample-size 10000
```

Two larger repaired configs are also available:
- duck-focused: [examples/duck_ten_shading_model_config.json](examples/duck_ten_shading_model_config.json)
- broader figurine mix: [examples/all_twelve_shading_model_config.json](examples/all_twelve_shading_model_config.json)

Bundle multiple repaired shading models into a weighted ensemble:

```bash
PYTHONPATH=src python -m color3dconverter.cli bundle-shading-models \
  --out-model /tmp/repaired_ensemble.pkl \
  --model-path /tmp/duck10_et.pkl \
  --model-path /tmp/all12_et.pkl \
  --weights 0.5 0.5
```

Apply a trained repaired shading model to a raw textured source plus repaired target geometry:

```bash
PYTHONPATH=src python -m color3dconverter.cli convert-shading-model \
  /path/to/source.glb \
  /path/to/repaired_target.obj \
  --model-path /tmp/eight_pair_direct_rgb_model_et.pkl \
  --alignment-json /tmp/provider_oracle/alignment_summary.json \
  --out-obj /tmp/eight_pair_predicted.obj
```

Convert a 3D AI Studio repaired model that already has baked texture data into Bambu-friendly region assets:

```bash
PYTHONPATH=src python -m color3dconverter.cli convert-provider-bake \
  /path/to/3dai_repaired_model.glb \
  --repair-result-json /path/to/3dai_repair_result.json \
  --out-dir /tmp/provider_bake_regions \
  --regions 8 \
  --strategy blender_cleanup_face_labels
```

This is the provider-bake oracle lane. It treats the provider's baked/repaired textured mesh as the color source, then reduces it into grouped OBJ/MTL and 3MF colorgroup assets for Bambu inspection.

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

Repaired production bundles also include:
- `selected/acceptance_summary.json`
- `selected/paint_intent_report.json`
- `selected/paint_intent_report.md`
- `_repair_geometry/repaired_geometry.obj`
- `_cleanup_candidates/paint_region_cleanup/` when cleanup is triggered by noisy component/tiny-island counts

DuckAgent handoff bundles also include:
- `handoff_manifest.json`
- `handoff_qa_board.png`
- `handoff_summary.md`
- `source_preview.png`

Lane-choice bundles also include:
- `lane_choice_report.json`

Shading-model training bundles also include:
- `<model>.pkl`
- `<model>.pkl.json`

Shading-model repaired conversions also include:
- `<output>.obj`
- `<output>.json`

## Python API

```python
from color3dconverter import choose_conversion_lane, convert_model_to_color_assets, run_duckagent_handoff, run_production_conversion

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

choice = choose_conversion_lane(
    [
        "/path/to/output_production/production_report.json",
        "/path/to/provider_bake_regions/conversion_report.json",
        "/path/to/repaired_transfer/conversion_report.json",
    ],
    out_report="/path/to/lane_choice_report.json",
)

print(choice["status"])
print(choice["selected_lane"])

handoff = run_duckagent_handoff(
    "/path/to/model.glb",
    out_dir="/path/to/duckagent_run/paint_to_print",
    object_name="Monster Truck Duck",
    max_colors=8,
)

print(handoff["ready_for_duckagent_handoff"])
print(handoff["artifacts"]["qa_board_path"])
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

The `repair-then-bake` command is the stricter algorithmic acceptance path:
- it repairs the textured source mesh locally
- it bakes source colors onto the repaired geometry
- it writes a dense debug vertex-color OBJ
- it writes a Bambu grouped OBJ/MTL and 3MF under `bambu_printable/`
- it records `ready_for_bambu_print` and rejection reasons when geometry, color drift, or region fragmentation do not pass

The `choose-lane` command is the safe report-only selector:
- it reads same-mesh production, provider-bake, and repaired-transfer reports
- it chooses the first report-ready lane by policy order
- it preserves rejected-lane reasons for operator review
- it does not mutate assets or publish anything

## Integration Notes

Downstream automation can call this package for:
- single-image GLB conversion
- repaired GLB conversion
- repaired-geometry plus transferred-color conversion
- local validation bundles for Bambu review
- DuckAgent handoff manifests that expose stable artifact paths and readiness gates

The converter should stay focused on deterministic asset generation and reports. Calling systems should own:
- run orchestration
- operator UI
- approval workflows
- choosing which source asset to feed into the converter

## Development

Install in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
PYTHONPATH=src python -m pytest
```

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, pull request expectations, and compatibility issue guidance.

## Roadmap

Near-term priorities:
- stronger paint-zone extraction for character parts like hat, body, and beak
- multipart export experiments
- more importer probes and example assets
- README examples that demonstrate repaired-geometry transfer workflows
- baseline-vs-cleanup candidate selection for noisy paint-region cases before adding provider-backed AI cleanup
- keeping `AGENTS.md` and [AI Development Guide](docs/AI_DEVELOPMENT_GUIDE.md) current as this becomes a public AI-assisted repo

## Repo Docs

- [Examples](examples/README.md)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [Integration Plan](docs/DUCKAGENT_INTEGRATION_PLAN.md)
- [DuckAgent Handoff Contract](docs/DUCKAGENT_HANDOFF_CONTRACT.md)
- [Generalization Sample Results](docs/GENERALIZATION_SAMPLE_RESULTS.md)
- [AI Skin Cleanup Lane Plan](docs/AI_SKIN_CLEANUP_LANE_PLAN.md)
- [AI Development Guide](docs/AI_DEVELOPMENT_GUIDE.md)
- [Provenance](docs/PROVENANCE.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [Agent Instructions](AGENTS.md)
