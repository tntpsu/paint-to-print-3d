from .bake import bake_texture_to_vertex_colors, build_uv_island_mask, sample_texture_bilinear, seam_pad_texture
from .benchmark import choose_preferred_lane, run_benchmark_suite, run_curved_transfer_experiments, run_fixture_benchmark
from .face_regions import build_region_first_face_palette, sample_texture, transfer_vertex_colors_from_source
from .export_3mf import write_colorgroup_3mf
from .export_obj import write_bambu_compatible_grouped_obj_with_mtl, write_grouped_obj_with_mtl
from .fixtures import BenchmarkFixture, list_benchmark_fixtures, load_benchmark_fixture
from .lane_chooser import choose_conversion_lane, normalize_lane_candidate
from .model_io import LoadedTexturedMesh, load_geometry_model, load_textured_glb, load_textured_model, load_textured_obj, load_textured_objzip
from .paint_cleanup import cleanup_paint_region_labels, paint_component_metrics
from .pipeline import (
    assess_repaired_transfer_candidate,
    assess_provider_bake_candidate,
    convert_color_transferred_mesh_to_assets,
    convert_face_colored_mesh_to_assets,
    convert_loaded_mesh_to_color_assets,
    convert_model_to_color_assets,
    convert_provider_baked_model_to_assets,
    convert_repaired_color_transfer_to_assets,
    convert_textured_obj_to_region_assets,
    write_face_color_mesh_to_assets,
    write_labeled_mesh_to_assets,
)
from .production import run_production_conversion, run_repaired_production_conversion
from .provider_oracle import run_provider_oracle_experiments
from .repair_then_bake import run_repair_then_bake_experiment
from .regions import assign_faces_to_texture_regions, build_texture_regions
from .shading_model import convert_with_shading_model, load_shading_model, sample_provider_pair_shading_data, train_shading_model
from .validation import validate_bambu_material_bundle, write_bambu_validation_bundle, write_source_export_comparison

__all__ = [
    "LoadedTexturedMesh",
    "BenchmarkFixture",
    "bake_texture_to_vertex_colors",
    "build_region_first_face_palette",
    "build_uv_island_mask",
    "choose_preferred_lane",
    "convert_color_transferred_mesh_to_assets",
    "convert_face_colored_mesh_to_assets",
    "choose_conversion_lane",
    "assign_faces_to_texture_regions",
    "assess_provider_bake_candidate",
    "assess_repaired_transfer_candidate",
    "build_texture_regions",
    "convert_loaded_mesh_to_color_assets",
    "convert_model_to_color_assets",
    "convert_provider_baked_model_to_assets",
    "convert_repaired_color_transfer_to_assets",
    "convert_with_shading_model",
    "convert_textured_obj_to_region_assets",
    "cleanup_paint_region_labels",
    "normalize_lane_candidate",
    "paint_component_metrics",
    "run_production_conversion",
    "run_repaired_production_conversion",
    "list_benchmark_fixtures",
    "load_benchmark_fixture",
    "load_geometry_model",
    "load_textured_glb",
    "load_textured_model",
    "load_textured_obj",
    "load_textured_objzip",
    "run_benchmark_suite",
    "run_curved_transfer_experiments",
    "run_fixture_benchmark",
    "run_provider_oracle_experiments",
    "run_repair_then_bake_experiment",
    "load_shading_model",
    "sample_provider_pair_shading_data",
    "sample_texture",
    "sample_texture_bilinear",
    "seam_pad_texture",
    "train_shading_model",
    "transfer_vertex_colors_from_source",
    "validate_bambu_material_bundle",
    "write_bambu_validation_bundle",
    "write_bambu_compatible_grouped_obj_with_mtl",
    "write_colorgroup_3mf",
    "write_face_color_mesh_to_assets",
    "write_labeled_mesh_to_assets",
    "write_grouped_obj_with_mtl",
    "write_source_export_comparison",
]
