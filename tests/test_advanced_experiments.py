from __future__ import annotations

from pathlib import Path

import numpy as np

from color3dconverter.advanced_experiments import run_repaired_transfer_experiment_suite
from color3dconverter.fixtures import build_six_color_cube_fixture, build_smiley_cube_fixture


def test_advanced_experiment_suite_runs_on_six_color_cube(tmp_path: Path) -> None:
    fixture = build_six_color_cube_fixture()
    assert fixture.repaired_mesh is not None
    summary = run_repaired_transfer_experiment_suite(
        target_loaded=fixture.repaired_mesh,
        color_source_loaded=fixture.same_mesh,
        out_dir=tmp_path / "suite",
        source_preview_path=None,
        probe_exports=[],
        max_colors=fixture.suggested_regions,
        source_mode="synthetic",
        simplify_applied=False,
    )
    experiment_names = {item["experiment_name"] for item in summary["experiments"]}
    assert "uv_label_transfer" in experiment_names
    assert "multiview_projection_transfer" in experiment_names
    assert "closest_face_projection_transfer" in experiment_names
    assert summary["best_repaired_experiment"] is not None
    assert Path(summary["same_mesh_seeded_parts"]).exists()
    assert Path(summary["same_mesh_high_contrast"]).exists()


def test_advanced_experiment_suite_generates_multipart_manifest(tmp_path: Path) -> None:
    fixture = build_smiley_cube_fixture()
    assert fixture.repaired_mesh is not None
    summary = run_repaired_transfer_experiment_suite(
        target_loaded=fixture.repaired_mesh,
        color_source_loaded=fixture.same_mesh,
        out_dir=tmp_path / "suite_smiley",
        source_preview_path=None,
        probe_exports=[],
        max_colors=fixture.suggested_regions,
        source_mode="synthetic",
        simplify_applied=False,
    )
    manifest_path = summary["multipart_manifest_path"]
    assert manifest_path is not None
    assert Path(manifest_path).exists()
    blocked_names = {item["name"] for item in summary["blocked_experiments"]}
    assert {"manual_part_mask_oracle", "template_retopology_experiment", "blender_oracle_comparison"} <= blocked_names
