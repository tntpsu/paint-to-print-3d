from __future__ import annotations

import argparse
import json

from .benchmark import (
    run_benchmark_suite,
    run_cross_case_iterative_search,
    run_curved_transfer_experiments,
    run_iterative_real_case_search,
    run_real_case_ablation,
    run_surface_bake_experiments,
)
from .fixtures import list_benchmark_fixtures
from .pipeline import convert_model_to_color_assets, convert_repaired_color_transfer_to_assets
from .production import run_production_conversion
from .provider_oracle import run_provider_oracle_experiments
from .repair_then_bake import run_repair_then_bake_experiment
from .shading_model import bundle_shading_models, convert_with_shading_model, train_shading_model


def main() -> None:
    parser = argparse.ArgumentParser(description="3dcolorconverter")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert-obj", help="Convert a textured OBJ into region-based assets.")
    convert_parser.add_argument("obj_path")
    convert_parser.add_argument("--texture-path")
    convert_parser.add_argument("--out-dir")
    convert_parser.add_argument("--regions", type=int, default=5)
    convert_parser.add_argument("--strategy", choices=["region_first", "legacy_fast_face_labels", "legacy_corner_face_labels", "blender_like_bake_face_labels", "blender_cleanup_face_labels", "hue_vcm_cleanup_face_labels"], default="region_first")
    convert_parser.add_argument("--object-name")

    convert_model_parser = subparsers.add_parser("convert-model", help="Convert a textured OBJ, OBJ ZIP, or GLB into region-based assets.")
    convert_model_parser.add_argument("source_path")
    convert_model_parser.add_argument("--texture-path")
    convert_model_parser.add_argument("--out-dir")
    convert_model_parser.add_argument("--regions", type=int, default=5)
    convert_model_parser.add_argument("--strategy", choices=["region_first", "legacy_fast_face_labels", "legacy_corner_face_labels", "blender_like_bake_face_labels", "blender_cleanup_face_labels", "hue_vcm_cleanup_face_labels"], default="region_first")
    convert_model_parser.add_argument("--object-name")

    production_parser = subparsers.add_parser("convert-production", help="Run the production same-mesh converter with gated candidate selection.")
    production_parser.add_argument("source_path")
    production_parser.add_argument("--texture-path")
    production_parser.add_argument("--out-dir", required=True)
    production_parser.add_argument("--object-name")
    production_parser.add_argument("--quality-threshold", type=float, default=0.02)
    production_parser.add_argument("--no-fail-closed", action="store_true")

    repaired_parser = subparsers.add_parser("convert-repaired-transfer", help="Transfer colors from a textured source model onto repaired target geometry.")
    repaired_parser.add_argument("color_source_path")
    repaired_parser.add_argument("target_path")
    repaired_parser.add_argument("--source-texture-path")
    repaired_parser.add_argument("--target-texture-path")
    repaired_parser.add_argument("--out-dir", required=True)
    repaired_parser.add_argument("--max-colors", type=int, default=12)
    repaired_parser.add_argument(
        "--strategy",
        choices=[
            "legacy_fast_face_labels",
            "legacy_face_regions",
            "legacy_face_regions_graph",
            "legacy_corner_face_regions",
            "blender_like_bake_face_labels",
            "blender_like_bake_face_regions",
            "duck_semantic_parts",
            "duck_seeded_parts",
            "geometry_transfer_legacy_face_regions_graph",
            "geometry_transfer_legacy_corner_face_regions",
            "geometry_transfer_blender_like_bake_face_regions",
            "geometry_transfer_duck_semantic_parts",
            "geometry_transfer_duck_seeded_parts",
            "region_first",
        ],
        default="legacy_fast_face_labels",
    )
    repaired_parser.add_argument("--object-name")

    benchmark_parser = subparsers.add_parser("benchmark", help="Run the synthetic fixture benchmark ladder.")
    benchmark_parser.add_argument("--out-dir", required=True)
    benchmark_parser.add_argument("--fixtures", nargs="*")
    benchmark_parser.add_argument("--same-mesh-strategy", choices=["legacy_fast_face_labels", "legacy_corner_face_labels", "blender_like_bake_face_labels", "blender_cleanup_face_labels", "hue_vcm_cleanup_face_labels", "region_first"], default="legacy_fast_face_labels")
    benchmark_parser.add_argument("--repaired-strategy", choices=["geometry_transfer_legacy_face_regions", "geometry_transfer_blender_like_bake_face_regions", "region_first"], default="geometry_transfer_legacy_face_regions")
    benchmark_parser.add_argument("--list-fixtures", action="store_true")

    curved_parser = subparsers.add_parser("curved-transfer", help="Run controlled curved-surface transfer experiments.")
    curved_parser.add_argument("--out-dir", required=True)
    curved_parser.add_argument("--fixtures", nargs="*")
    curved_parser.add_argument(
        "--strategies",
        nargs="*",
        choices=[
            "geometry_transfer_texture_regions",
            "legacy_face_regions",
            "geometry_transfer_legacy_face_regions_graph",
            "geometry_transfer_blender_like_bake_face_regions",
        ],
    )

    surface_bake_parser = subparsers.add_parser("surface-bake", help="Run the surface-texture to vertex/corner color experiment ladder.")
    surface_bake_parser.add_argument("--out-dir", required=True)
    surface_bake_parser.add_argument("--experiments", nargs="*")

    real_ablation_parser = subparsers.add_parser("real-ablation", help="Run a fixed-strategy ablation study across real source variants.")
    real_ablation_parser.add_argument("--config", required=True)
    real_ablation_parser.add_argument("--out-dir")
    real_ablation_parser.add_argument("--strategy", choices=["legacy_fast_face_labels", "legacy_corner_face_labels", "blender_like_bake_face_labels", "blender_cleanup_face_labels", "hue_vcm_cleanup_face_labels"])
    real_ablation_parser.add_argument("--regions", type=int)

    iterative_parser = subparsers.add_parser("iterative-search", help="Run a bounded metric-driven search across source-prep and conversion candidates.")
    iterative_parser.add_argument("--config", required=True)
    iterative_parser.add_argument("--out-dir")

    cross_case_parser = subparsers.add_parser("cross-case-search", help="Run a bounded search that scores each candidate across multiple real cases.")
    cross_case_parser.add_argument("--config", required=True)
    cross_case_parser.add_argument("--out-dir")

    provider_parser = subparsers.add_parser("provider-oracle", help="Compare textured source baking against a provider-generated repaired vertex-color OBJ.")
    provider_parser.add_argument("source_path")
    provider_parser.add_argument("target_obj_path")
    provider_parser.add_argument("--out-dir", required=True)
    provider_parser.add_argument("--sample-size", type=int, default=5000)
    provider_parser.add_argument("--seed", type=int, default=42)
    provider_parser.add_argument("--export-best-full", action="store_true")
    provider_parser.add_argument("--alignment-json")

    repair_bake_parser = subparsers.add_parser("repair-then-bake", help="Repair a textured mesh, bake dense vertex colors onto the repaired mesh, and optionally compare against a provider repaired OBJ.")
    repair_bake_parser.add_argument("source_path")
    repair_bake_parser.add_argument("--out-dir", required=True)
    repair_bake_parser.add_argument("--provider-target-obj")
    repair_bake_parser.add_argument("--backend", action="append", choices=["trimesh_clean", "pymeshfix_core"])
    repair_bake_parser.add_argument("--bake-method", choices=["nearest_vertex", "nearest_surface_uv"], default="nearest_surface_uv")
    repair_bake_parser.add_argument("--sample-size", type=int, default=12000)
    repair_bake_parser.add_argument("--target-face-count", type=int, default=250000)
    repair_bake_parser.add_argument("--seed", type=int, default=42)

    shading_parser = subparsers.add_parser("train-shading-model", help="Train a shared repaired shading model from provider source/target pairs.")
    shading_parser.add_argument("--config", required=True)
    shading_parser.add_argument("--out-model", required=True)
    shading_parser.add_argument("--model-kind", choices=["rf", "ridge", "et", "et_router", "et_residual_router", "hgb", "mlp"], default="rf")
    shading_parser.add_argument("--target-kind", choices=["scalar", "direct_rgb", "residual_rgb", "channel_scale"], default="scalar")
    shading_parser.add_argument("--sample-size", type=int, default=5000)
    shading_parser.add_argument("--seed", type=int, default=42)

    shading_convert_parser = subparsers.add_parser("convert-shading-model", help="Apply a trained repaired shading model to a textured source and repaired target OBJ.")
    shading_convert_parser.add_argument("source_path")
    shading_convert_parser.add_argument("target_obj_path")
    shading_convert_parser.add_argument("--model-path", required=True)
    shading_convert_parser.add_argument("--out-obj", required=True)
    shading_convert_parser.add_argument("--alignment-json")

    shading_bundle_parser = subparsers.add_parser("bundle-shading-models", help="Bundle multiple repaired shading models into a weighted ensemble.")
    shading_bundle_parser.add_argument("--out-model", required=True)
    shading_bundle_parser.add_argument("--model-path", action="append", required=True)
    shading_bundle_parser.add_argument("--weights", nargs="*", type=float)

    args = parser.parse_args()
    if args.command == "convert-obj":
        report = convert_model_to_color_assets(
            args.obj_path,
            texture_path=args.texture_path,
            out_dir=args.out_dir,
            n_regions=args.regions,
            strategy=args.strategy,
            object_name=args.object_name,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "convert-production":
        report = run_production_conversion(
            args.source_path,
            texture_path=args.texture_path,
            out_dir=args.out_dir,
            object_name=args.object_name,
            quality_threshold=args.quality_threshold,
            fail_closed=not bool(args.no_fail_closed),
        )
        print(json.dumps(report, indent=2))
    elif args.command == "convert-repaired-transfer":
        report = convert_repaired_color_transfer_to_assets(
            args.color_source_path,
            args.target_path,
            color_source_texture_path=args.source_texture_path,
            target_texture_path=args.target_texture_path,
            out_dir=args.out_dir,
            max_colors=args.max_colors,
            strategy=args.strategy,
            object_name=args.object_name,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "convert-model":
        report = convert_model_to_color_assets(
            args.source_path,
            texture_path=args.texture_path,
            out_dir=args.out_dir,
            n_regions=args.regions,
            strategy=args.strategy,
            object_name=args.object_name,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "benchmark":
        if args.list_fixtures:
            print(json.dumps(list_benchmark_fixtures(), indent=2))
            return
        report = run_benchmark_suite(
            out_dir=args.out_dir,
            fixture_names=args.fixtures,
            same_mesh_strategy=args.same_mesh_strategy,
            repaired_strategy=args.repaired_strategy,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "curved-transfer":
        report = run_curved_transfer_experiments(
            out_dir=args.out_dir,
            fixture_names=args.fixtures,
            strategies=args.strategies,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "surface-bake":
        report = run_surface_bake_experiments(
            out_dir=args.out_dir,
            experiment_names=args.experiments,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "real-ablation":
        report = run_real_case_ablation(
            config_path=args.config,
            out_dir=args.out_dir,
            strategy=args.strategy,
            n_regions=args.regions,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "iterative-search":
        report = run_iterative_real_case_search(
            config_path=args.config,
            out_dir=args.out_dir,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "cross-case-search":
        report = run_cross_case_iterative_search(
            config_path=args.config,
            out_dir=args.out_dir,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "provider-oracle":
        alignment_summary = None
        if args.alignment_json:
            with open(args.alignment_json, "r", encoding="utf-8") as handle:
                alignment_summary = json.load(handle)
        report = run_provider_oracle_experiments(
            source_path=args.source_path,
            target_obj_path=args.target_obj_path,
            out_dir=args.out_dir,
            sample_size=args.sample_size,
            seed=args.seed,
            export_best_full=bool(args.export_best_full),
            alignment_summary=alignment_summary,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "repair-then-bake":
        report = run_repair_then_bake_experiment(
            source_path=args.source_path,
            out_dir=args.out_dir,
            provider_target_obj_path=args.provider_target_obj,
            repair_backends=args.backend,
            sample_size=args.sample_size,
            bake_method=args.bake_method,
            target_face_count=args.target_face_count,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "train-shading-model":
        with open(args.config, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        report = train_shading_model(
            pair_specs=list(config.get("pairs") or []),
            out_model_path=args.out_model,
            model_kind=args.model_kind,
            target_kind=args.target_kind,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "convert-shading-model":
        alignment_summary = None
        if args.alignment_json:
            with open(args.alignment_json, "r", encoding="utf-8") as handle:
                alignment_summary = json.load(handle)
        report = convert_with_shading_model(
            source_path=args.source_path,
            target_obj_path=args.target_obj_path,
            model_path=args.model_path,
            out_obj_path=args.out_obj,
            alignment_summary=alignment_summary,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "bundle-shading-models":
        report = bundle_shading_models(
            model_paths=list(args.model_path or []),
            out_model_path=args.out_model,
            weights=list(args.weights) if args.weights else None,
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
