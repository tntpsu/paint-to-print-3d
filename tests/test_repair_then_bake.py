from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from color3dconverter.bake import sample_texture_bilinear
from color3dconverter.repair_then_bake import run_repair_then_bake_experiment


def test_run_repair_then_bake_experiment_writes_outputs(tmp_path: Path) -> None:
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
    provider_obj = tmp_path / "provider.obj"
    provider_obj.write_text(
        "\n".join(
            [
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

    report = run_repair_then_bake_experiment(
        source_path=source_obj,
        out_dir=tmp_path / "repair_then_bake",
        provider_target_obj_path=provider_obj,
        repair_backends=["trimesh_clean"],
        sample_size=3,
        seed=5,
    )

    assert report["best_result"] is not None
    assert report["best_result"]["backend"] == "trimesh_clean"
    assert Path(report["best_result"]["predicted_obj_path"]).exists()
    assert Path(report["best_result"]["preview_path"]).exists()
    assert Path(report["best_result"]["board_path"]).exists()
    assert report["best_result"]["provider_color_metrics"]["mean_abs_total"] == 0.0
