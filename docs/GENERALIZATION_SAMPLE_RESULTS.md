# Generalization Sample Results

This file tracks real model checks for the repaired production lane so the converter does not overfit to one showcase asset.

## Current Lane

- Command: `convert-repaired-production`
- Repair backend: `voxel_marching_cubes`
- Voxel divisions: `128`
- Smoothing: Taubin, `18` iterations
- Transfer strategy: `geometry_transfer_blender_like_bake_duck_intent`
- Max colors: `8`

## Samples

| Sample | Status | Main Finding | Output |
| --- | --- | --- | --- |
| Captain America duck | Ready | Smoothing plus paint-intent cleanup preserves shield/star/white details while removing gray head-cap artifacts. | `outputs/captain_duck_smooth_duck_intent_v11/selected` |
| Polo duck | Ready | The policy now chooses the dominant body color instead of assuming the body is blue, so yellow duck plus blue shirt works. | `outputs/generalization_duck_polo_v2/selected` |
| Spa duck | Review recommended | Geometry is watertight and flat-bottomed, and towel/cucumber intent is visible, but tiny island count is too high for auto print-readiness. | `outputs/generalization_spa_duck_v1/selected` |
| Spa duck cleanup smoke | Review recommended | Deterministic cleanup reduced components from 2,615 to 270 and tiny islands from 2,518 to 237, but correctly did not promote because the tiny-island gate is still 96. | `outputs/generalization_spa_duck_cleanup_v1/_cleanup_candidates/paint_region_cleanup` |

## Lessons

- The body-color policy must be semantic, not color-specific. A blue-body assumption works for Captain America but is wrong for standard yellow ducks with blue clothes.
- Light neutral colors should be treated as intentional details unless they become a large gray/tan artifact over a body/head zone.
- Tiny-island count is a strong early warning for noisy provider texture or over-detailed towel/fabric surfaces.
- Deterministic island absorption can greatly improve noisy paint regions, but it should stay fail-closed until the candidate meets the same tiny-island and Bambu validation gates as the baseline.
- Passing Bambu format validation is necessary but not sufficient. The paint-intent report should gate operator trust because fragmented material groups can still be painful to print.

## Next Benchmark Candidates

- Lacrosse duck for helmet color/detail preservation.
- Dapper duck for dark clothing, hat, and high-contrast face details.
- Canoe duck for non-duck accessory geometry and wood colors.
