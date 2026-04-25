from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

import numpy as np

from color3dconverter.model_io import LoadedTexturedMesh
from color3dconverter.pipeline import (
    convert_color_transferred_mesh_to_assets,
    convert_loaded_mesh_to_color_assets,
    convert_model_to_color_assets,
    convert_repaired_color_transfer_to_assets,
    convert_textured_obj_to_region_assets,
)
from color3dconverter.validation import write_bambu_validation_bundle


def test_convert_textured_obj_to_region_assets(tmp_path: Path) -> None:
    obj_path = tmp_path / "sample.obj"
    mtl_path = tmp_path / "sample.mtl"
    texture_path = tmp_path / "texture.png"

    texture = Image.new("RGB", (2, 2), (255, 210, 0))
    texture.putpixel((1, 1), (180, 30, 30))
    texture.save(texture_path)

    mtl_path.write_text(
        "\n".join(
            [
                "newmtl Material",
                f"map_Kd {texture_path.name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    obj_path.write_text(
        "\n".join(
            [
                f"mtllib {mtl_path.name}",
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

    report = convert_textured_obj_to_region_assets(obj_path, out_dir=tmp_path / "out", n_regions=2)

    assert Path(report["obj_path"]).exists()
    assert Path(report["mtl_path"]).exists()
    assert Path(report["threemf_path"]).exists()
    assert Path(report["report_path"]).exists()
    assert report["region_count"] == 2
    assert report["obj_export_style"] == "bambu_simple"
    obj_text = Path(report["obj_path"]).read_text(encoding="utf-8")
    mtl_text = Path(report["mtl_path"]).read_text(encoding="utf-8")
    assert "usemtl mat_1" in obj_text
    assert "newmtl mat_1" in mtl_text
    assert "Material_01" not in obj_text


def test_convert_model_to_color_assets_accepts_objzip(tmp_path: Path) -> None:
    obj_path = tmp_path / "sample.obj"
    mtl_path = tmp_path / "sample.mtl"
    texture_path = tmp_path / "texture.png"
    zip_path = tmp_path / "sample_obj_bundle.zip"

    texture = Image.new("RGB", (2, 2), (255, 210, 0))
    texture.putpixel((1, 1), (180, 30, 30))
    texture.save(texture_path)

    mtl_path.write_text(
        "\n".join(
            [
                "newmtl Material",
                f"map_Kd {texture_path.name}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    obj_path.write_text(
        "\n".join(
            [
                f"mtllib {mtl_path.name}",
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
    with ZipFile(zip_path, "w") as archive:
        archive.write(obj_path, arcname=obj_path.name)
        archive.write(mtl_path, arcname=mtl_path.name)
        archive.write(texture_path, arcname=texture_path.name)

    report = convert_model_to_color_assets(zip_path, out_dir=tmp_path / "out", n_regions=2, object_name="OBJ ZIP Duck")

    assert Path(report["obj_path"]).exists()
    assert Path(report["threemf_path"]).exists()
    assert report["source_format"] == "objzip"


def test_convert_loaded_mesh_to_color_assets(tmp_path: Path) -> None:
    loaded = LoadedTexturedMesh(
        mesh=None,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        texcoords=np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        texture_rgb=np.array(
            [
                [[255, 210, 0], [255, 210, 0]],
                [[180, 30, 30], [180, 30, 30]],
            ],
            dtype=np.uint8,
        ),
        source_path=tmp_path / "synthetic.glb",
        texture_path=None,
        source_format="glb",
    )

    report = convert_loaded_mesh_to_color_assets(loaded, out_dir=tmp_path / "out", n_regions=2, object_name="Synthetic Duck")

    assert Path(report["obj_path"]).exists()
    assert Path(report["threemf_path"]).exists()
    assert Path(report["preview_path"]).exists()
    assert Path(report["palette_csv_path"]).exists()
    assert Path(report["vertex_color_obj_path"]).exists()
    assert report["source_format"] == "glb"
    assert report["face_count"] == 1


def test_convert_loaded_mesh_to_color_assets_legacy_fast_strategy(tmp_path: Path) -> None:
    loaded = LoadedTexturedMesh(
        mesh=None,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        texcoords=np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        texture_rgb=np.array(
            [
                [[255, 210, 0], [255, 210, 0]],
                [[180, 30, 30], [180, 30, 30]],
            ],
            dtype=np.uint8,
        ),
        source_path=tmp_path / "synthetic.obj",
        texture_path=None,
        source_format="obj",
    )

    report = convert_loaded_mesh_to_color_assets(
        loaded,
        out_dir=tmp_path / "legacy_out",
        n_regions=2,
        strategy="legacy_fast_face_labels",
        object_name="Legacy Synthetic Duck",
    )

    assert report["strategy"] == "legacy_fast_face_labels"
    assert report["legacy_fast_path"] is True
    assert report["obj_export_style"] == "bambu_simple"
    assert Path(report["obj_path"]).exists()
    assert Path(report["mtl_path"]).exists()
    obj_text = Path(report["obj_path"]).read_text(encoding="utf-8")
    assert "usemtl mat_1" in obj_text


def test_convert_loaded_mesh_to_color_assets_blender_like_bake_strategy(tmp_path: Path) -> None:
    loaded = LoadedTexturedMesh(
        mesh=None,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        texcoords=np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        texture_rgb=np.array(
            [
                [[255, 210, 0], [255, 210, 0]],
                [[180, 30, 30], [180, 30, 30]],
            ],
            dtype=np.uint8,
        ),
        source_path=tmp_path / "synthetic.glb",
        texture_path=None,
        source_format="glb",
    )

    report = convert_loaded_mesh_to_color_assets(
        loaded,
        out_dir=tmp_path / "bake_out",
        n_regions=2,
        strategy="blender_like_bake_face_labels",
        object_name="Baked Synthetic Duck",
    )

    assert report["strategy"] == "blender_like_bake_face_labels"
    assert report["blender_like_bake"] is True
    assert report["bake_metadata"]["sampling_mode"] == "bilinear"
    assert Path(report["obj_path"]).exists()


def test_convert_loaded_mesh_to_color_assets_legacy_corner_strategy(tmp_path: Path) -> None:
    loaded = LoadedTexturedMesh(
        mesh=None,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        texcoords=np.array(
            [
                [0.0, 1.0],
                [0.45, 1.0],
                [0.45, 0.0],
                [0.55, 1.0],
                [1.0, 0.0],
                [0.55, 0.0],
            ],
            dtype=np.float32,
        ),
        texture_rgb=np.array(
            [
                [[255, 0, 0], [255, 0, 0], [0, 0, 255], [0, 0, 255]],
                [[255, 0, 0], [255, 0, 0], [0, 0, 255], [0, 0, 255]],
            ],
            dtype=np.uint8,
        ),
        source_path=tmp_path / "seam.glb",
        texture_path=None,
        source_format="glb",
    )
    report = convert_loaded_mesh_to_color_assets(
        loaded,
        out_dir=tmp_path / "corner_out",
        n_regions=2,
        strategy="legacy_corner_face_labels",
        object_name="Corner Duck",
    )
    assert report["strategy"] == "legacy_corner_face_labels"
    assert report["legacy_corner_path"] is True
    assert Path(report["obj_path"]).exists()


def test_convert_loaded_mesh_to_color_assets_blender_cleanup_strategy(tmp_path: Path) -> None:
    loaded = LoadedTexturedMesh(
        mesh=None,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        texcoords=np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        texture_rgb=np.array(
            [
                [[255, 210, 0], [255, 210, 0]],
                [[180, 30, 30], [180, 30, 30]],
            ],
            dtype=np.uint8,
        ),
        source_path=tmp_path / "cleanup.glb",
        texture_path=None,
        source_format="glb",
    )
    report = convert_loaded_mesh_to_color_assets(
        loaded,
        out_dir=tmp_path / "cleanup_out",
        n_regions=3,
        strategy="blender_cleanup_face_labels",
        object_name="Cleanup Duck",
    )
    assert report["strategy"] == "blender_cleanup_face_labels"
    assert report["blender_cleanup_path"] is True
    assert report["corner_bake_metadata"]["sampling_mode"] == "bilinear"
    assert Path(report["obj_path"]).exists()
    assert Path(report["threemf_path"]).exists()


def test_convert_loaded_mesh_to_color_assets_hue_vcm_cleanup_strategy(tmp_path: Path) -> None:
    loaded = LoadedTexturedMesh(
        mesh=None,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        texcoords=np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        texture_rgb=np.array(
            [
                [[255, 210, 0], [255, 210, 0]],
                [[180, 30, 30], [180, 30, 30]],
            ],
            dtype=np.uint8,
        ),
        source_path=tmp_path / "hue_vcm.glb",
        texture_path=None,
        source_format="glb",
    )
    report = convert_loaded_mesh_to_color_assets(
        loaded,
        out_dir=tmp_path / "hue_vcm_out",
        n_regions=3,
        strategy="hue_vcm_cleanup_face_labels",
        object_name="HUE VCM Duck",
    )
    assert report["strategy"] == "hue_vcm_cleanup_face_labels"
    assert report["hue_vcm_cleanup_path"] is True
    assert report["corner_bake_metadata"]["sampling_mode"] == "bilinear"
    assert Path(report["obj_path"]).exists()
    assert Path(report["threemf_path"]).exists()


def test_convert_color_transferred_mesh_to_assets(tmp_path: Path) -> None:
    target = LoadedTexturedMesh(
        mesh=None,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        texcoords=np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        texture_rgb=np.array([[[255, 255, 255]]], dtype=np.uint8),
        source_path=tmp_path / "target.glb",
        texture_path=None,
        source_format="glb",
    )
    color_source = LoadedTexturedMesh(
        mesh=None,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        texcoords=np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        texture_rgb=np.array(
            [
                [[255, 0, 0], [255, 0, 0]],
                [[255, 0, 0], [255, 0, 0]],
            ],
            dtype=np.uint8,
        ),
        source_path=tmp_path / "source.glb",
        texture_path=None,
        source_format="glb",
    )

    report = convert_color_transferred_mesh_to_assets(
        target_loaded=target,
        color_source_loaded=color_source,
        out_dir=tmp_path / "out",
        max_colors=8,
        object_name="Transferred Duck",
    )

    assert Path(report["obj_path"]).exists()
    assert Path(report["threemf_path"]).exists()
    assert Path(report["preview_path"]).exists()
    assert report["strategy"] == "geometry_transfer_texture_regions"
    assert report["color_transfer_applied"] is True
    assert report["color_source_path"] == str(color_source.source_path)


def test_convert_color_transferred_mesh_to_assets_legacy_fast_strategy(tmp_path: Path) -> None:
    target = LoadedTexturedMesh(
        mesh=None,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        texcoords=np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        texture_rgb=np.array([[[255, 255, 255]]], dtype=np.uint8),
        source_path=tmp_path / "target.glb",
        texture_path=None,
        source_format="glb",
    )
    color_source = LoadedTexturedMesh(
        mesh=None,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        texcoords=np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        texture_rgb=np.array(
            [
                [[255, 0, 0], [255, 0, 0]],
                [[255, 0, 0], [255, 0, 0]],
            ],
            dtype=np.uint8,
        ),
        source_path=tmp_path / "source.glb",
        texture_path=None,
        source_format="glb",
    )

    report = convert_color_transferred_mesh_to_assets(
        target_loaded=target,
        color_source_loaded=color_source,
        out_dir=tmp_path / "out_legacy",
        max_colors=8,
        strategy="legacy_fast_face_labels",
        object_name="Transferred Duck Legacy",
    )

    assert report["strategy"] == "geometry_transfer_legacy_face_regions"
    assert report["legacy_fast_path"] is True
    assert report["region_transfer_mode"] == "connected_face_regions"
    assert report["source_region_model"] == "legacy_fast_face_regions"
    assert set(report["duck_part_anchor_labels"].keys()) == {"body", "bandana", "hat", "beak"}
    assert report["color_transfer_applied"] is True
    assert Path(report["obj_path"]).exists()
    assert Path(report["threemf_path"]).exists()


def test_convert_repaired_color_transfer_accepts_untextured_target_obj(tmp_path: Path) -> None:
    source_obj = tmp_path / "source.obj"
    source_mtl = tmp_path / "source.mtl"
    source_texture = tmp_path / "source_texture.png"
    target_obj = tmp_path / "target.obj"

    texture = Image.new("RGB", (2, 2), (255, 0, 0))
    texture.save(source_texture)
    source_mtl.write_text("newmtl Material\nmap_Kd source_texture.png\n", encoding="utf-8")
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
    target_obj.write_text(
        "\n".join(
            [
                "v 0 0 0.05",
                "v 1 0 0.05",
                "v 0 1 0.05",
                "f 1 2 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = convert_repaired_color_transfer_to_assets(
        source_obj,
        target_obj,
        out_dir=tmp_path / "repaired_transfer",
        max_colors=4,
        strategy="legacy_fast_face_labels",
        object_name="Repaired Transfer Duck",
    )

    assert report["conversion_lane"] == "repaired_geometry_region_transfer"
    assert report["target_path"] == str(target_obj.resolve())
    assert report["target_texture_path"] is None
    assert Path(report["obj_path"]).exists()
    assert Path(report["threemf_path"]).exists()
    saved = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert saved["conversion_lane"] == "repaired_geometry_region_transfer"


def test_convert_color_transferred_mesh_to_assets_blender_like_bake_strategy(tmp_path: Path) -> None:
    target = LoadedTexturedMesh(
        mesh=None,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        texcoords=np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        texture_rgb=np.array([[[255, 255, 255]]], dtype=np.uint8),
        source_path=tmp_path / "target.glb",
        texture_path=None,
        source_format="glb",
    )
    color_source = LoadedTexturedMesh(
        mesh=None,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        texcoords=np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        texture_rgb=np.array(
            [
                [[255, 0, 0], [255, 0, 0]],
                [[255, 0, 0], [255, 0, 0]],
            ],
            dtype=np.uint8,
        ),
        source_path=tmp_path / "source.glb",
        texture_path=None,
        source_format="glb",
    )

    report = convert_color_transferred_mesh_to_assets(
        target_loaded=target,
        color_source_loaded=color_source,
        out_dir=tmp_path / "out_bake_transfer",
        max_colors=8,
        strategy="blender_like_bake_face_labels",
        object_name="Transferred Duck Baked",
    )

    assert report["strategy"] == "geometry_transfer_blender_like_bake_face_regions"
    assert report["blender_like_bake"] is True
    assert report["region_transfer_mode"] == "connected_face_regions"


def test_convert_color_transferred_mesh_to_assets_semantic_parts_strategy(tmp_path: Path) -> None:
    target = LoadedTexturedMesh(
        mesh=None,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        texcoords=np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        texture_rgb=np.array([[[255, 255, 255]]], dtype=np.uint8),
        source_path=tmp_path / "target.glb",
        texture_path=None,
        source_format="glb",
    )
    color_source = LoadedTexturedMesh(
        mesh=None,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        texcoords=np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        texture_rgb=np.array(
            [
                [[242, 210, 74], [242, 210, 74]],
                [[242, 210, 74], [242, 210, 74]],
            ],
            dtype=np.uint8,
        ),
        source_path=tmp_path / "source.glb",
        texture_path=None,
        source_format="glb",
    )

    report = convert_color_transferred_mesh_to_assets(
        target_loaded=target,
        color_source_loaded=color_source,
        out_dir=tmp_path / "out_semantic_transfer",
        max_colors=5,
        strategy="geometry_transfer_duck_semantic_parts",
        object_name="Transferred Duck Semantic",
    )

    assert report["strategy"] == "geometry_transfer_duck_semantic_parts"
    assert report["region_transfer_mode"] == "duck_semantic_parts"
    assert "semantic_part_ids" in report
    assert report["semantic_part_ids"] == {"body": 0, "bandana": 1, "hat": 2, "beak": 3, "accent": 4}
    assert set(report["duck_part_anchor_labels"].keys()) == {"body", "bandana", "hat", "beak"}
    assert report["color_transfer_applied"] is True
    assert report["corner_bake_metadata"]["sampling_mode"] == "nearest"
    assert Path(report["obj_path"]).exists()
    assert Path(report["threemf_path"]).exists()


def test_convert_color_transferred_mesh_to_assets_seeded_parts_strategy(tmp_path: Path) -> None:
    target = LoadedTexturedMesh(
        mesh=None,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        texcoords=np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        texture_rgb=np.array([[[255, 255, 255]]], dtype=np.uint8),
        source_path=tmp_path / "target.glb",
        texture_path=None,
        source_format="glb",
    )
    color_source = LoadedTexturedMesh(
        mesh=None,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        texcoords=np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        texture_rgb=np.array(
            [
                [[242, 210, 74], [242, 210, 74]],
                [[242, 210, 74], [242, 210, 74]],
            ],
            dtype=np.uint8,
        ),
        source_path=tmp_path / "source.glb",
        texture_path=None,
        source_format="glb",
    )

    report = convert_color_transferred_mesh_to_assets(
        target_loaded=target,
        color_source_loaded=color_source,
        out_dir=tmp_path / "out_seeded_transfer",
        max_colors=7,
        strategy="geometry_transfer_duck_seeded_parts",
        object_name="Transferred Duck Seeded",
    )

    assert report["strategy"] == "geometry_transfer_duck_seeded_parts"
    assert report["region_transfer_mode"] == "duck_seeded_parts"
    assert report["semantic_part_ids"] == {
        "body": 0,
        "bandana": 1,
        "hat_brim": 2,
        "hat_crown": 3,
        "beak": 4,
        "boots": 5,
        "accent": 6,
    }
    assert report["corner_bake_metadata"]["sampling_mode"] == "nearest"
    assert Path(report["obj_path"]).exists()
    assert Path(report["threemf_path"]).exists()


def test_write_bambu_validation_bundle(tmp_path: Path) -> None:
    source_preview = tmp_path / "source.png"
    export_preview = tmp_path / "export.png"
    source_image = Image.new("RGB", (128, 128), (210, 160, 90))
    export_image = Image.new("RGB", (128, 128), (200, 150, 84))
    source_image.save(source_preview)
    export_image.save(export_preview)

    threemf_path = tmp_path / "model.3mf"
    obj_path = tmp_path / "model.obj"
    threemf_path.write_bytes(b"fake-3mf")
    obj_path.write_text("# fake obj\n", encoding="utf-8")

    bundle = write_bambu_validation_bundle(
        output_dir=tmp_path,
        source_preview_path=source_preview,
        export_preview_path=export_preview,
        threemf_path=threemf_path,
        obj_path=obj_path,
        probe_exports=[{"label": "4-color probe", "path": str(tmp_path / "probe.3mf")}],
        source_mode="single_image",
    )

    assert bundle is not None
    assert Path(bundle["comparison_path"]).exists()
    assert Path(bundle["validation_report_path"]).exists()
    assert Path(bundle["validation_markdown_path"]).exists()
    assert bundle["assessment"] in {"close", "moderate_drift", "poor_match"}
