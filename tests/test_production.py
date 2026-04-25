from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from color3dconverter.production import run_production_conversion, run_repaired_production_conversion


def _write_toy_textured_obj(root: Path) -> Path:
    texture_path = root / "texture.png"
    image = Image.new("RGB", (4, 4), (230, 190, 40))
    image.putpixel((3, 0), (190, 40, 40))
    image.putpixel((0, 3), (40, 90, 190))
    image.save(texture_path)
    (root / "sample.mtl").write_text("newmtl Material\nmap_Kd texture.png\n", encoding="utf-8")
    obj_path = root / "sample.obj"
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
    return obj_path


def _write_watertight_tetra_obj(root: Path) -> Path:
    texture_path = root / "tetra_texture.png"
    image = Image.new("RGB", (2, 2), (40, 90, 190))
    image.putpixel((1, 0), (230, 190, 40))
    image.putpixel((0, 1), (190, 40, 40))
    image.save(texture_path)
    (root / "tetra.mtl").write_text("newmtl Material\nmap_Kd tetra_texture.png\n", encoding="utf-8")
    obj_path = root / "tetra.obj"
    obj_path.write_text(
        "\n".join(
            [
                "mtllib tetra.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "vt 0 0",
                "vt 1 0",
                "vt 0 1",
                "vt 1 1",
                "usemtl Material",
                "f 1/1 3/3 2/2",
                "f 1/1 2/2 4/4",
                "f 2/2 3/3 4/4",
                "f 3/3 1/1 4/4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return obj_path


def test_run_production_conversion_writes_selected_artifacts(tmp_path: Path) -> None:
    obj_path = _write_toy_textured_obj(tmp_path)

    report = run_production_conversion(
        obj_path,
        out_dir=tmp_path / "production_out",
        quality_threshold=0.2,
    )

    assert report["ready_for_production"] is True
    production_report_path = Path(report["production_report_path"])
    assert production_report_path.exists()
    selected_dir = Path(report["selected_candidate"]["selected_dir"])
    assert selected_dir.exists()
    assert (selected_dir / "conversion_report.json").exists()
    assert (selected_dir / "region_preview.png").exists()
    assert (selected_dir / "source_export_comparison.png").exists()
    saved = json.loads(production_report_path.read_text(encoding="utf-8"))
    assert saved["status"] == "ready"
    assert len(saved["candidates"]) == 4


def test_run_production_conversion_fail_closed_rejects_when_threshold_is_impossible(tmp_path: Path) -> None:
    obj_path = _write_toy_textured_obj(tmp_path)

    report = run_production_conversion(
        obj_path,
        out_dir=tmp_path / "production_reject",
        quality_threshold=-1.0,
        fail_closed=True,
    )

    assert report["ready_for_production"] is False
    assert report["status"] == "rejected"
    assert "did not meet" in report["message"]
    assert Path(report["selected_candidate"]["selected_dir"]).exists()


def test_run_repaired_production_conversion_writes_validated_assets(tmp_path: Path) -> None:
    obj_path = _write_watertight_tetra_obj(tmp_path)

    report = run_repaired_production_conversion(
        obj_path,
        out_dir=tmp_path / "repaired_production",
        object_name="Repaired Tetra Duck",
        repair_backend="trimesh_clean",
        target_face_count=None,
        max_colors=4,
    )

    assert report["ready_for_production"] is True
    assert report["status"] == "ready"
    assert report["transfer_strategy"] == "geometry_transfer_blender_like_bake_duck_intent"
    assert report["repair_smoothing_iterations"] == 0
    assert report["repair_voxel_divisions"] is None
    assert Path(report["acceptance_summary_path"]).exists()
    assert Path(report["production_report_path"]).exists()
    assert Path(report["paint_intent_report_path"]).exists()
    assert Path(report["paint_intent_markdown_path"]).exists()
    assert Path(report["obj_path"]).exists()
    assert Path(report["mtl_path"]).exists()
    assert Path(report["threemf_path"]).exists()
    assert report["bambu_material_validation"]["ready_for_bambu"] is True
    assert report["bambu_material_validation"]["obj_topology"]["is_watertight"] is True
    assert report["bambu_material_validation"]["threemf_colorgroup"]["triangle_count"] == 4
    paint_intent = json.loads(Path(report["paint_intent_report_path"]).read_text(encoding="utf-8"))
    assert paint_intent["summary"]["palette_size"] == report["palette_size"]
    assert "bottom_flatness" in paint_intent["geometry"]
