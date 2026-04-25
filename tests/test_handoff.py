from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from color3dconverter.handoff import HANDOFF_SCHEMA_VERSION, run_duckagent_handoff


def _write_subdivided_box_obj(root: Path, *, divisions: int = 3) -> Path:
    texture_path = root / "duck_texture.png"
    texture = Image.new("RGB", (8, 8), (240, 205, 55))
    for x in range(4, 8):
        for y in range(0, 4):
            texture.putpixel((x, y), (35, 95, 210))
    for x in range(5, 8):
        for y in range(5, 8):
            texture.putpixel((x, y), (230, 95, 35))
    texture.save(texture_path)
    (root / "box.mtl").write_text("newmtl Material\nmap_Kd duck_texture.png\n", encoding="utf-8")

    vertices: list[tuple[float, float, float]] = []
    vertex_indexes: dict[tuple[float, float, float], int] = {}
    texcoords: list[tuple[float, float]] = []
    faces: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []

    def vertex_index(coord: tuple[float, float, float]) -> int:
        key = tuple(round(value, 6) for value in coord)
        if key not in vertex_indexes:
            vertex_indexes[key] = len(vertices) + 1
            vertices.append(key)
        return vertex_indexes[key]

    def texcoord_index(uv: tuple[float, float]) -> int:
        texcoords.append(uv)
        return len(texcoords)

    def add_quad(corners: list[tuple[float, float, float]], uv: list[tuple[float, float]]) -> None:
        refs = [(vertex_index(coord), texcoord_index(tex)) for coord, tex in zip(corners, uv, strict=True)]
        faces.append((refs[0], refs[1], refs[2]))
        faces.append((refs[0], refs[2], refs[3]))

    def add_face(axis: str, value: float) -> None:
        for i in range(divisions):
            for j in range(divisions):
                a = i / divisions
                b = j / divisions
                c = (i + 1) / divisions
                d = (j + 1) / divisions
                if axis == "y0":
                    corners = [(a, value, b), (c, value, b), (c, value, d), (a, value, d)]
                elif axis == "y1":
                    corners = [(a, value, b), (a, value, d), (c, value, d), (c, value, b)]
                elif axis == "x0":
                    corners = [(value, a, b), (value, a, d), (value, c, d), (value, c, b)]
                elif axis == "x1":
                    corners = [(value, a, b), (value, c, b), (value, c, d), (value, a, d)]
                elif axis == "z0":
                    corners = [(a, b, value), (a, d, value), (c, d, value), (c, b, value)]
                else:
                    corners = [(a, b, value), (c, b, value), (c, d, value), (a, d, value)]
                add_quad(corners, [(a, b), (c, b), (c, d), (a, d)])

    for face_axis, face_value in (
        ("y0", 0.0),
        ("y1", 1.0),
        ("x0", 0.0),
        ("x1", 1.0),
        ("z0", 0.0),
        ("z1", 1.0),
    ):
        add_face(face_axis, face_value)

    lines = ["mtllib box.mtl"]
    lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    lines.extend(f"vt {u:.6f} {v:.6f}" for u, v in texcoords)
    lines.append("usemtl Material")
    for face in faces:
        lines.append("f " + " ".join(f"{vertex}/{texcoord}" for vertex, texcoord in face))

    obj_path = root / "box.obj"
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return obj_path


def test_run_duckagent_handoff_writes_manifest_and_qa_board(tmp_path: Path) -> None:
    obj_path = _write_subdivided_box_obj(tmp_path)

    manifest = run_duckagent_handoff(
        obj_path,
        out_dir=tmp_path / "handoff",
        object_name="Box Duck Fixture",
        repair_backend="trimesh_clean",
        target_face_count=None,
        max_colors=4,
        min_colors=1,
        repair_smoothing_iterations=0,
    )

    assert manifest["schema_version"] == HANDOFF_SCHEMA_VERSION
    assert manifest["status"] in {"ready", "review_required"}
    assert isinstance(manifest["ready_for_duckagent_handoff"], bool)
    assert manifest["duckagent_contract"]["read"] == "handoff_manifest.json"
    assert "bambu_3mf_path" in manifest["duckagent_contract"]["stable_artifact_keys"]
    assert Path(manifest["artifacts"]["handoff_manifest_path"]).exists()
    assert Path(manifest["artifacts"]["handoff_markdown_path"]).exists()
    assert Path(manifest["artifacts"]["qa_board_path"]).exists()
    assert Path(manifest["artifacts"]["bambu_3mf_path"]).exists()
    assert Path(manifest["artifacts"]["grouped_obj_path"]).exists()
    assert Path(manifest["artifacts"]["grouped_mtl_path"]).exists()
    assert Path(manifest["artifacts"]["source_preview_path"]).exists()

    saved = json.loads(Path(manifest["artifacts"]["handoff_manifest_path"]).read_text(encoding="utf-8"))
    assert saved["schema_version"] == HANDOFF_SCHEMA_VERSION
    assert {gate["id"] for gate in saved["gates"]} >= {
        "required_artifacts_exist",
        "bambu_material_bundle_valid",
        "flat_bottom_support_preserved",
        "qa_board_written",
    }
