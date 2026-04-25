from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from color3dconverter.benchmark import (
    choose_preferred_lane,
    run_benchmark_suite,
    run_cross_case_iterative_search,
    run_curved_transfer_experiments,
    run_fixture_benchmark,
    run_iterative_real_case_search,
    run_real_case_ablation,
    run_surface_bake_experiments,
)
from color3dconverter.fixtures import load_benchmark_fixture


def test_run_fixture_benchmark_on_cube(tmp_path: Path) -> None:
    fixture = load_benchmark_fixture("six_color_cube")
    summary = run_fixture_benchmark(fixture, out_dir=tmp_path)

    assert summary["fixture_name"] == "six_color_cube"
    assert len(summary["lanes"]) == 2
    assert summary["preferred_lane"] is not None

    same_mesh = next(item for item in summary["lanes"] if item["lane"] == "same_mesh")
    assert same_mesh["face_accuracy"] >= 0.99
    assert Path(same_mesh["report_path"]).exists()
    assert Path(same_mesh["comparison_path"]).exists()


def test_choose_preferred_lane_prefers_higher_score(tmp_path: Path) -> None:
    fixture = load_benchmark_fixture("six_color_cube")
    summary = run_fixture_benchmark(fixture, out_dir=tmp_path / "fixture")
    preferred = choose_preferred_lane([])
    assert preferred is None
    preferred = choose_preferred_lane(
        [
            type("Lane", (), lane)()
            for lane in summary["lanes"]
        ]
    )
    assert preferred is not None


def test_run_benchmark_suite_subset(tmp_path: Path) -> None:
    suite = run_benchmark_suite(out_dir=tmp_path, fixture_names=["six_color_cube", "smiley_cube"])
    assert suite["fixture_count"] == 2
    assert len(suite["fixtures"]) == 2
    assert (tmp_path / "suite_summary.json").exists()


def test_run_curved_transfer_experiments_subset(tmp_path: Path) -> None:
    suite = run_curved_transfer_experiments(
        out_dir=tmp_path,
        fixture_names=["banded_sphere"],
        strategies=["legacy_face_regions", "geometry_transfer_legacy_face_regions_graph"],
    )
    assert suite["fixture_count"] == 1
    assert suite["fixtures"] == ["banded_sphere"]
    assert (tmp_path / "curved_transfer_suite.json").exists()


def test_run_surface_bake_experiments_subset(tmp_path: Path) -> None:
    suite = run_surface_bake_experiments(
        out_dir=tmp_path,
        experiment_names=[
            "01_seam_split_quad_collapsed_nearest",
            "03_seam_split_quad_corner_bilinear",
        ],
    )
    assert suite["experiment_count"] == 2
    assert (tmp_path / "surface_bake_suite.json").exists()


def test_run_real_case_ablation(tmp_path: Path) -> None:
    texture_path = tmp_path / "texture.png"
    Image.new("RGB", (2, 2), (255, 210, 0)).save(texture_path)
    mtl_path = tmp_path / "sample.mtl"
    mtl_path.write_text("newmtl Material\nmap_Kd texture.png\n", encoding="utf-8")
    obj_path = tmp_path / "sample.obj"
    obj_path.write_text(
        "\n".join(
            [
                "mtllib sample.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "vt 0 1",
                "vt 1 1",
                "vt 0 0",
                "usemtl Material",
                "f 1/1 2/2 3/3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    preview_path = tmp_path / "source_preview.png"
    Image.new("RGB", (64, 64), (240, 200, 20)).save(preview_path)
    config_path = tmp_path / "ablation.json"
    config_path.write_text(
        json.dumps(
            {
                "case_name": "toy_case",
                "source_preview_path": str(preview_path),
                "source_mode": "toy_case",
                "strategy": "legacy_fast_face_labels",
                "n_regions": 2,
                "variants": [
                    {"label": "v1", "source_path": str(obj_path)},
                    {"label": "v2", "source_path": str(obj_path)},
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = run_real_case_ablation(config_path=config_path, out_dir=tmp_path / "out")
    assert summary["case_name"] == "toy_case"
    assert summary["variant_count"] == 2
    assert Path(summary["results"][0]["report_path"]).exists()
    assert Path(summary["board_path"]).exists()


def test_run_real_case_ablation_with_texture_transform(tmp_path: Path) -> None:
    texture_path = tmp_path / "texture.png"
    image = Image.new("RGB", (2, 2), (180, 120, 80))
    image.putpixel((1, 1), (220, 200, 40))
    image.save(texture_path)
    mtl_path = tmp_path / "sample.mtl"
    mtl_path.write_text("newmtl Material\nmap_Kd texture.png\n", encoding="utf-8")
    obj_path = tmp_path / "sample.obj"
    obj_path.write_text(
        "\n".join(
            [
                "mtllib sample.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "vt 0 1",
                "vt 1 1",
                "vt 0 0",
                "usemtl Material",
                "f 1/1 2/2 3/3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    preview_path = tmp_path / "source_preview.png"
    Image.new("RGB", (64, 64), (200, 140, 80)).save(preview_path)
    config_path = tmp_path / "ablation_transform.json"
    config_path.write_text(
        json.dumps(
            {
                "case_name": "toy_transform_case",
                "source_preview_path": str(preview_path),
                "strategy": "legacy_fast_face_labels",
                "n_regions": 2,
                "variants": [
                    {
                        "label": "baseline",
                        "source_path": str(obj_path)
                    },
                    {
                        "label": "posterized",
                        "source_path": str(obj_path),
                        "texture_transform": {
                            "posterize_levels": 2,
                            "brightness": 0.02,
                            "contrast": 0.1,
                            "saturation": 1.1,
                            "value": 1.05
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = run_real_case_ablation(config_path=config_path, out_dir=tmp_path / "out_transform")
    assert summary["variant_count"] == 2
    transformed_dir = Path(summary["results"][0]["report_path"]).parent
    assert (transformed_dir / "local_bambu_converter_report.json").exists()
    # One of the variants should have the generated transformed assets.
    assert any((Path(row["report_path"]).parent / "transformed_texture.png").exists() for row in summary["results"])


def test_run_iterative_real_case_search(tmp_path: Path) -> None:
    texture_path = tmp_path / "texture.png"
    image = Image.new("RGB", (2, 2), (180, 120, 80))
    image.putpixel((1, 0), (220, 200, 40))
    image.putpixel((0, 1), (150, 90, 60))
    image.putpixel((1, 1), (110, 60, 30))
    image.save(texture_path)
    mtl_path = tmp_path / "sample.mtl"
    mtl_path.write_text("newmtl Material\nmap_Kd texture.png\n", encoding="utf-8")
    obj_path = tmp_path / "sample.obj"
    obj_path.write_text(
        "\n".join(
            [
                "mtllib sample.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "vt 0 1",
                "vt 1 1",
                "vt 0 0",
                "usemtl Material",
                "f 1/1 2/2 3/3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    preview_path = tmp_path / "source_preview.png"
    Image.new("RGB", (64, 64), (200, 140, 80)).save(preview_path)
    config_path = tmp_path / "iterative.json"
    config_path.write_text(
        json.dumps(
            {
                "case_name": "iterative_toy_case",
                "source_path": str(obj_path),
                "source_preview_path": str(preview_path),
                "strategy": "legacy_fast_face_labels",
                "n_regions": 2,
                "target_value": 0.2,
                "improvement_epsilon": 0.001,
                "patience": 2,
                "max_iterations": 4,
                "base_candidate": {
                    "strategy": "legacy_fast_face_labels",
                    "n_regions": 2,
                },
                "search_space": {
                    "strategy": ["legacy_fast_face_labels"],
                    "n_regions": [2],
                    "texture_transform": {
                        "posterize_levels": [None, 2],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    summary = run_iterative_real_case_search(config_path=config_path, out_dir=tmp_path / "iterative_out")
    assert summary["case_name"] == "iterative_toy_case"
    assert summary["round_count"] >= 1
    assert summary["best_result"] is not None
    assert Path(summary["best_result"]["report_path"]).exists()
    assert (tmp_path / "iterative_out" / "iterative_summary.json").exists()


def test_run_cross_case_iterative_search(tmp_path: Path) -> None:
    texture_path = tmp_path / "texture.png"
    Image.new("RGB", (2, 2), (200, 140, 80)).save(texture_path)
    mtl_path = tmp_path / "sample.mtl"
    mtl_path.write_text("newmtl Material\nmap_Kd texture.png\n", encoding="utf-8")
    obj_path = tmp_path / "sample.obj"
    obj_path.write_text(
        "\n".join(
            [
                "mtllib sample.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "vt 0 1",
                "vt 1 1",
                "vt 0 0",
                "usemtl Material",
                "f 1/1 2/2 3/3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    preview_a = tmp_path / "preview_a.png"
    preview_b = tmp_path / "preview_b.png"
    Image.new("RGB", (64, 64), (200, 140, 80)).save(preview_a)
    Image.new("RGB", (64, 64), (190, 130, 70)).save(preview_b)
    config_path = tmp_path / "cross_case.json"
    config_path.write_text(
        json.dumps(
            {
                "suite_name": "toy_cross_case_suite",
                "strategy": "legacy_fast_face_labels",
                "n_regions": 2,
                "target_value": 1.0,
                "improvement_epsilon": 0.001,
                "patience": 2,
                "max_iterations": 3,
                "base_candidate": {
                    "strategy": "legacy_fast_face_labels",
                    "n_regions": 2
                },
                "search_space": {
                    "strategy": ["legacy_fast_face_labels"],
                    "n_regions": [2],
                    "texture_transform": {
                        "posterize_levels": [None, 2]
                    }
                },
                "cases": [
                    {
                        "case_name": "case_a",
                        "source_path": str(obj_path),
                        "source_preview_path": str(preview_a),
                        "target_value": 1.0
                    },
                    {
                        "case_name": "case_b",
                        "source_path": str(obj_path),
                        "source_preview_path": str(preview_b),
                        "target_value": 1.0
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    summary = run_cross_case_iterative_search(config_path=config_path, out_dir=tmp_path / "cross_case_out")
    assert summary["suite_name"] == "toy_cross_case_suite"
    assert summary["case_count"] == 2
    assert summary["best_result"] is not None
    assert len(summary["best_result"]["case_results"]) == 2
    assert Path(summary["best_result"]["case_results"][0]["report_path"]).exists()
    assert (tmp_path / "cross_case_out" / "cross_case_summary.json").exists()
