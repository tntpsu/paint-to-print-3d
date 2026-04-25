from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from color3dconverter.bake import sample_texture_bilinear
from color3dconverter.provider_oracle import run_provider_oracle_experiments


def test_run_provider_oracle_experiments_matches_same_geometry_vertex_colors(tmp_path: Path) -> None:
    texture_path = tmp_path / "texture.png"
    texture = Image.new("RGB", (2, 2), (0, 0, 0))
    texture.putpixel((0, 0), (255, 0, 0))
    texture.putpixel((1, 0), (0, 255, 0))
    texture.putpixel((0, 1), (0, 0, 255))
    texture.putpixel((1, 1), (255, 255, 0))
    texture.save(texture_path)

    source_mtl = tmp_path / "source.mtl"
    source_mtl.write_text("newmtl Material\nmap_Kd texture.png\n", encoding="utf-8")
    source_obj = tmp_path / "source.obj"
    source_obj.write_text(
        "\n".join(
            [
                "mtllib source.mtl",
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

    uv = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    vertex_colors = sample_texture_bilinear(np.array(texture, dtype=np.uint8), uv).astype(np.float32) / 255.0

    target_obj = tmp_path / "target.obj"
    target_obj.write_text(
        "\n".join(
            [
                "# vertex-color target",
                *(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}" for (x, y, z), (r, g, b) in zip(
                    [
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                    ],
                    vertex_colors.tolist(),
                )),
                "f 1 2 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_provider_oracle_experiments(
        source_path=source_obj,
        target_obj_path=target_obj,
        out_dir=tmp_path / "provider_oracle",
        sample_size=3,
        seed=7,
        variants=[
            {
                "label": "nearest_vertex_bilinear_vflip",
                "method": "nearest_vertex",
                "sampling_mode": "bilinear",
                "uv_flip_y": True,
                "candidate_count": 1,
                "pad_pixels": 0,
            }
        ],
        export_best_full=True,
    )

    assert summary["best_result"] is not None
    assert summary["best_result"]["label"] == "nearest_vertex_bilinear_vflip"
    assert summary["best_result"]["mean_abs_total"] == 0.0
    assert summary["best_result"]["fraction_within_8"] == 1.0
    predicted_obj = Path(summary["best_full_result"]["predicted_obj_path"])
    assert predicted_obj.exists()
    assert (tmp_path / "provider_oracle" / "alignment_summary.json").exists()


def test_run_provider_oracle_experiments_weighted_surface_uv_matches_same_geometry(tmp_path: Path) -> None:
    texture_path = tmp_path / "texture_weighted.png"
    texture = Image.new("RGB", (2, 2), (0, 0, 0))
    texture.putpixel((0, 0), (255, 0, 0))
    texture.putpixel((1, 0), (0, 255, 0))
    texture.putpixel((0, 1), (0, 0, 255))
    texture.putpixel((1, 1), (255, 255, 0))
    texture.save(texture_path)

    source_mtl = tmp_path / "weighted.mtl"
    source_mtl.write_text("newmtl Material\nmap_Kd texture_weighted.png\n", encoding="utf-8")
    source_obj = tmp_path / "weighted.obj"
    source_obj.write_text(
        "\n".join(
            [
                "mtllib weighted.mtl",
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

    uv = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    vertex_colors = sample_texture_bilinear(np.array(texture, dtype=np.uint8), uv).astype(np.float32) / 255.0

    target_obj = tmp_path / "weighted_target.obj"
    target_obj.write_text(
        "\n".join(
            [
                "# weighted target",
                *(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}" for (x, y, z), (r, g, b) in zip(
                    [
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                    ],
                    vertex_colors.tolist(),
                )),
                "f 1 2 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_provider_oracle_experiments(
        source_path=source_obj,
        target_obj_path=target_obj,
        out_dir=tmp_path / "provider_oracle_weighted",
        sample_size=3,
        seed=11,
        variants=[
            {
                "label": "weighted_surface_uv",
                "method": "weighted_surface_uv",
                "sampling_mode": "bilinear",
                "uv_flip_y": True,
                "candidate_count": 3,
                "pad_pixels": 0,
                "distance_power": 2.0,
                "normal_power": 0.0,
                "smooth_neighbors": 0,
                "smooth_blend": 0.0,
            }
        ],
    )

    assert summary["best_result"] is not None
    assert summary["best_result"]["label"] == "weighted_surface_uv"
    assert summary["best_result"]["mean_abs_total"] == 0.0


def test_run_provider_oracle_experiments_oracle_shaded_surface_uv_ridge_reduces_shaded_error(tmp_path: Path) -> None:
    texture_path = tmp_path / "texture_oracle_shaded.png"
    texture = Image.new("RGB", (2, 2), (0, 0, 0))
    texture.putpixel((0, 0), (255, 0, 0))
    texture.putpixel((1, 0), (0, 255, 0))
    texture.putpixel((0, 1), (0, 0, 255))
    texture.putpixel((1, 1), (255, 255, 0))
    texture.save(texture_path)

    source_mtl = tmp_path / "oracle_shaded.mtl"
    source_mtl.write_text("newmtl Material\nmap_Kd texture_oracle_shaded.png\n", encoding="utf-8")
    source_obj = tmp_path / "oracle_shaded.obj"
    source_obj.write_text(
        "\n".join(
            [
                "mtllib oracle_shaded.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 1 1 0",
                "vt 0 1",
                "vt 1 1",
                "vt 0 0",
                "vt 1 0",
                "usemtl Material",
                "f 1/1 2/2 3/3",
                "f 2/2 4/4 3/3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    base_vertex_colors = sample_texture_bilinear(
        np.array(texture, dtype=np.uint8),
        np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    ).astype(np.float32)
    shade = np.array([0.5, 0.75, 1.25, 1.5], dtype=np.float32)
    target_vertex_colors = np.clip(np.rint(base_vertex_colors * shade[:, None]), 0, 255).astype(np.uint8) / 255.0

    target_obj = tmp_path / "oracle_shaded_target.obj"
    target_obj.write_text(
        "\n".join(
            [
                "# oracle shaded target",
                *(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}" for (x, y, z), (r, g, b) in zip(
                    [
                        (0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (1.0, 1.0, 0.0),
                    ],
                    target_vertex_colors.tolist(),
                )),
                "f 1 2 3",
                "f 2 4 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_provider_oracle_experiments(
        source_path=source_obj,
        target_obj_path=target_obj,
        out_dir=tmp_path / "provider_oracle_oracle_shaded",
        sample_size=4,
        seed=13,
        variants=[
            {
                "label": "oracle_shaded_surface_uv_ridge",
                "method": "oracle_shaded_surface_uv_ridge",
                "sampling_mode": "bilinear",
                "uv_flip_y": True,
                "candidate_count": 4,
                "pad_pixels": 0,
                "shading_folds": 2,
            }
        ],
    )

    assert summary["best_result"] is not None
    assert summary["best_result"]["label"] == "oracle_shaded_surface_uv_ridge"
    assert summary["best_result"]["mean_abs_total"] < summary["best_result"]["base_mean_abs_total"]
