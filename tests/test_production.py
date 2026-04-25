from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from color3dconverter.production import run_production_conversion


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
