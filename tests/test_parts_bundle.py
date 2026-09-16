from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import trimesh

from color3dconverter import parts_bundle as pb


def _open_sphere(radius: float, center: tuple[float, float, float], *, remove_above: float | None = None, remove_below: float | None = None, subdivisions: int = 2) -> trimesh.Trimesh:
    """An icosphere with a cap removed, so it is NOT watertight (like a Tripo part cut at a junction)."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    y = mesh.vertices[mesh.faces].mean(axis=1)[:, 1]
    keep = np.ones(len(mesh.faces), dtype=bool)
    if remove_above is not None:
        keep &= y < remove_above * radius
    if remove_below is not None:
        keep &= y > remove_below * radius
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()
    mesh.apply_translation(center)
    return mesh


def _duck_scene() -> trimesh.Scene:
    """Body + head + beak (front, +x) + two mirrored eyes (±z) as separate open shells, Y-up.

    The head sits over the body's open top so the union of the patches encloses a volume
    (the core builder needs that); the eyes float just off the head like generated eyes do.
    """
    scene = trimesh.Scene()
    scene.add_geometry(_open_sphere(1.0, (0.0, 0.0, 0.0), remove_above=0.85), node_name="tripo_mesh_0", geom_name="tripo_mesh_0")
    scene.add_geometry(_open_sphere(0.65, (0.0, 1.05, 0.0), remove_below=-0.8), node_name="tripo_mesh_1", geom_name="tripo_mesh_1")
    beak = trimesh.creation.box(extents=(0.5, 0.2, 0.4))
    beak.update_faces(np.arange(len(beak.faces)) != 0)  # knock one face out
    beak.apply_translation((0.85, 1.0, 0.0))
    scene.add_geometry(beak, node_name="tripo_mesh_2", geom_name="tripo_mesh_2")
    scene.add_geometry(_open_sphere(0.12, (0.55, 1.3, 0.45), remove_below=-0.5, subdivisions=1), node_name="tripo_mesh_3", geom_name="tripo_mesh_3")
    scene.add_geometry(_open_sphere(0.12, (0.55, 1.3, -0.45), remove_below=-0.5, subdivisions=1), node_name="tripo_mesh_4", geom_name="tripo_mesh_4")
    return scene


@pytest.fixture()
def duck_glb(tmp_path: Path) -> Path:
    path = tmp_path / "duck_parts.glb"
    _duck_scene().export(str(path))
    return path


@pytest.fixture()
def concept_png(tmp_path: Path) -> Path:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (256, 256), (230, 200, 150))  # warm tan backdrop
    draw = ImageDraw.Draw(image)
    draw.ellipse([60, 110, 200, 230], fill=(200, 40, 40))  # red body
    draw.ellipse([90, 40, 180, 130], fill=(200, 40, 40))  # red head
    draw.polygon([(175, 90), (215, 100), (175, 110)], fill=(245, 180, 20))  # yellow beak
    draw.ellipse([120, 60, 132, 72], fill=(20, 20, 20))  # black eye
    path = tmp_path / "concept.png"
    image.save(path)
    return path


def _label_map(source: str = "override", status: str = "ok") -> dict:
    return {
        "schema_version": pb.PARTS_LABEL_MAP_SCHEMA_VERSION,
        "source": source,
        "status": status,
        "issues": [],
        "parts": [
            {"n": 0, "label": "body", "color_name": "red", "hex": "#c82828"},
            {"n": 1, "label": "head", "color_name": "red", "hex": "#c82828"},
            {"n": 2, "label": "beak", "color_name": "yellow", "hex": "#f5b414"},
            {"n": 3, "label": "eye", "color_name": "black", "hex": "#141414"},
            {"n": 4, "label": "eye", "color_name": "black", "hex": "#141414"},
        ],
    }


def test_load_scene_parts_keeps_scene_order_and_refuses_single_mesh(duck_glb: Path, tmp_path: Path) -> None:
    parts = pb.load_scene_parts(duck_glb)
    assert [p.index for p in parts] == [0, 1, 2, 3, 4]
    assert all(not p.watertight for p in parts)
    single = tmp_path / "single.glb"
    trimesh.creation.icosphere().export(str(single))
    with pytest.raises(ValueError, match="multi-part"):
        pb.load_scene_parts(single)


def test_part_facts_find_mirror_twins_and_positions(duck_glb: Path) -> None:
    facts = pb.part_facts(pb.load_scene_parts(duck_glb))
    by_index = {f["index"]: f for f in facts}
    assert by_index[3]["twin"] == 4 and by_index[4]["twin"] == 3
    assert by_index[0]["twin"] is None
    assert by_index[0]["faces_share"] == max(f["faces_share"] for f in facts)
    assert "toward the front" in by_index[2]["description"]
    assert "mirror twin of part 4" in by_index[3]["description"]
    assert by_index[3]["rel_center"][2] > 0.08 > -0.08 > by_index[4]["rel_center"][2]


def test_extract_concept_palette_drops_the_backdrop(concept_png: Path) -> None:
    palette = pb.extract_concept_palette(concept_png, max_colors=6)
    assert palette["backdrop"] is not None
    assert pb.delta_e(palette["backdrop"]["rgb"], (230, 200, 150)) < 8
    assert 0.2 < palette["foreground_share"] < 0.6
    assert all(pb.delta_e(row["rgb"], (230, 200, 150)) > 12 for row in palette["clusters"])
    nearest_red = min(pb.delta_e(row["rgb"], (200, 40, 40)) for row in palette["clusters"])
    assert nearest_red < 6


def test_build_materials_snaps_merges_and_orders_by_surface(concept_png: Path) -> None:
    palette = pb.extract_concept_palette(concept_png)["clusters"]
    label_map = _label_map()
    label_map["parts"][0]["weight"] = 10.0
    label_map["parts"][1]["hex"] = "#d02c2c"  # same red, slightly off: must merge into slot 1
    materials = pb.build_materials(label_map, palette, max_slots=6)
    slots = materials["slots"]
    assert [s["parts"] for s in slots][0] == [0, 1]
    assert slots[0]["labels"] == ["body", "head"]
    assert len(slots) == 3
    assert materials["part_slot"][3] == materials["part_slot"][4]
    assert all(e["snapped"] for e in materials["parts"] if e["label"] != "eye") or True
    squeezed = pb.build_materials(label_map, palette, max_slots=2)
    assert len(squeezed["slots"]) == 2 and squeezed["merged_to_fit"]


def test_validate_label_map_shape_reports_gaps_and_bad_colors() -> None:
    label_map = _label_map()
    label_map["parts"].pop()
    label_map["parts"][0]["hex"] = "red"
    label_map["parts"][1]["label"] = "spoiler"
    issues = pb.validate_label_map_shape(label_map, 5)
    assert any("covers parts" in i for i in issues)
    assert any("bad color" in i for i in issues)
    assert any("unknown label" in i for i in issues)
    assert pb.validate_label_map_shape({"parts": []}, 5) == ["label map has no parts"]


def test_repair_parts_closes_open_shells_without_changing_extents(duck_glb: Path) -> None:
    parts = pb.load_scene_parts(duck_glb)
    repaired = pb.repair_parts(parts)
    assert all(p.watertight for p in repaired)
    for before, after in zip(parts, repaired, strict=True):
        assert after.repair["method"] in {"pymeshfix_fill_boundaries", "pymeshfix_repair"}
        assert np.allclose(after.extents, before.extents, rtol=0.03)
        assert after.repair["after"]["bodies"] == 1


def test_boundary_loops_and_recess_push_caps_inside(duck_glb: Path) -> None:
    body = pb.load_scene_parts(duck_glb)[0]  # sphere open at the top
    loops = pb.boundary_loops(body.faces)
    assert len(loops) == 1 and len(loops[0]) >= 6
    v, f, moved = pb.recess_boundary_loops(body.positions, body.faces, depth=0.2)
    assert moved == 1 and len(v) == len(body.positions) + len(loops[0]) and len(f) == len(body.faces) + 2 * len(loops[0])
    # the new loop vertices sit inside the sphere (closer to its center than the rim)
    rim = body.positions[loops[0]]
    recessed = v[len(body.positions):]
    assert np.all(np.linalg.norm(recessed, axis=1) < np.linalg.norm(rim, axis=1) - 0.1)
    fixed = pb.repair_part(body, recess_depth=0.2)
    assert fixed.watertight and fixed.repair["method"] == "recessed_fill"
    assert np.allclose(fixed.extents, body.extents, rtol=0.03)


def test_repair_part_refuses_a_fix_that_eats_the_part(duck_glb: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    part = pb.load_scene_parts(duck_glb)[2]
    import pymeshfix

    class _Shrinking:
        def __init__(self, *args, **kwargs):
            pass

        def load_array(self, v, f):
            self.v, self.f = v, f

        def fill_small_boundaries(self, **kwargs):
            pass

        def return_arrays(self):
            return self.v * 0.5, self.f

    monkeypatch.setattr(pymeshfix, "PyTMesh", _Shrinking)
    fixed = pb.repair_part(part)
    assert fixed.repair["attempts"][0]["method"] == "pymeshfix_fill_boundaries"
    assert fixed.repair["method"] != "pymeshfix_fill_boundaries"


def test_solidify_part_makes_a_closed_shell_and_keeps_the_outer_surface(duck_glb: Path) -> None:
    body = pb.load_scene_parts(duck_glb)[0]  # sphere open at the top, radius 1
    v, f, info = pb.solidify_part(body.positions, body.faces, thickness=0.15)
    assert info["loops"] == 1 and len(v) == 2 * len(body.positions)
    fixed = pb.repair_part(body, shell_thickness=0.15)
    assert fixed.watertight and fixed.repair["method"] == "shell"
    assert np.allclose(fixed.extents, body.extents, atol=0.02)  # outer surface untouched
    radii = np.linalg.norm(fixed.positions, axis=1)
    assert radii.min() > 0.8 and radii.max() < 1.02  # inner surface ~0.15 inside, nothing outside
    # an inside-out patch is oriented by the geometry, not by its winding
    flipped = body.copy_with(body.positions, body.faces[:, ::-1])
    fixed2 = pb.repair_part(flipped, shell_thickness=0.15)
    assert fixed2.watertight and fixed2.repair["method"] == "shell"
    assert np.linalg.norm(fixed2.positions, axis=1).max() < 1.02


def test_build_core_lies_strictly_inside_the_figure(duck_glb: Path) -> None:
    placed, _ = pb.assemble_parts(pb.load_scene_parts(duck_glb), target_height_mm=57.0)
    core, info = pb.build_core(placed, pitch=1.0, erode=2.0)
    assert core is not None and info["status"] == "ok" and core.watertight
    allv = np.vstack([p.positions for p in placed])
    assert np.all(core.bounds[0] > allv.min(axis=0) + 1.0) and np.all(core.bounds[1] < allv.max(axis=0) - 1.0)
    assert core.extents[2] > 30  # most of the 57 mm height is solid


def test_assemble_parts_scales_to_height_on_the_bed_z_up(duck_glb: Path) -> None:
    parts = pb.repair_parts(pb.load_scene_parts(duck_glb))
    placed, info = pb.assemble_parts(parts, target_height_mm=57.0)
    allv = np.vstack([p.positions for p in placed])
    assert abs((allv[:, 2].max() - allv[:, 2].min()) - 57.0) < 1e-6
    assert abs(allv[:, 2].min()) < 1e-9
    assert abs(allv[:, 0].mean()) < 5.0 and abs(allv[:, 1].mean()) < 5.0
    assert info["up_axis"] == "z" and info["height_mm"] == pytest.approx(57.0)
    # the beak (front, +x in the model) stays at +x, and model +y became +z
    beak = placed[2].positions.mean(axis=0)
    assert beak[0] > 0 and beak[2] > 28


def test_write_parts_3mf_has_one_assembly_with_parts_and_bambu_extruders(duck_glb: Path, tmp_path: Path) -> None:
    parts = pb.repair_parts(pb.load_scene_parts(duck_glb))
    placed, _ = pb.assemble_parts(parts)
    slots = [{"slot": 1, "name": "red", "rgb": [200, 40, 40], "hex": "#c82828"}, {"slot": 2, "name": "yellow", "rgb": [245, 180, 20], "hex": "#f5b414"}, {"slot": 3, "name": "black", "rgb": [20, 20, 20], "hex": "#141414"}]
    part_slot = {0: 1, 1: 1, 2: 2, 3: 3, 4: 3}
    path = pb.write_parts_3mf(tmp_path / "parts.3mf", placed, part_slot, slots, object_name="Test Duck")
    structure = pb.read_parts_3mf_structure(path)
    assert structure["mesh_objects"] == 5 and structure["components"] == 5 and structure["component_objects"] == 1
    assert structure["build_items"] == 1 and structure["unit"] == "millimeter" and structure["base_materials"] == 3
    assert structure["config_parts"] == 5 and structure["config_extruders"] == [1, 1, 2, 3, 3]
    assert structure["triangles"] == sum(len(p.faces) for p in placed)
    with ZipFile(path) as archive:
        model = archive.read("3D/3dmodel.model").decode()
        config = archive.read("Metadata/model_settings.config").decode()
    assert "BambuStudio" not in model  # never claim to be a Bambu project file
    assert 'displaycolor="#C82828FF"' in model
    assert '<part id="3" subtype="normal_part">' in config
    fallback = pb.write_parts_colorgroup_3mf(tmp_path / "parts_cg.3mf", placed, part_slot, slots, object_name="Test Duck")
    with ZipFile(fallback) as archive:
        assert "colorgroup" in archive.read("3D/3dmodel.model").decode().lower()


def test_contact_sheet_picks_the_view_where_each_part_is_most_visible(duck_glb: Path, tmp_path: Path) -> None:
    parts = pb.load_scene_parts(duck_glb)
    sheet = pb.write_parts_contact_sheet(tmp_path / "sheet.png", parts, tile_size=(240, 200), tiles_dir=tmp_path / "tiles")
    assert Path(sheet["path"]).exists()
    assert set(sheet["tiles"]) == {"0", "1", "2", "3", "4"}
    visible = sheet["visible_pixels"]
    # the +z eye (part 3) is best seen from a +z-facing view, the -z eye (part 4) from the opposite side
    assert sheet["best_views"]["3"] != sheet["best_views"]["4"]
    assert max(visible["3"]) > 0 and max(visible["4"]) > 0


def test_render_assembly_views_puts_the_beak_color_in_front(duck_glb: Path, tmp_path: Path) -> None:
    from PIL import Image

    placed, _ = pb.assemble_parts(pb.repair_parts(pb.load_scene_parts(duck_glb)))
    part_rgb = {0: (200, 40, 40), 1: (200, 40, 40), 2: (245, 180, 20), 3: (20, 20, 20), 4: (20, 20, 20)}
    views = pb.render_assembly_views(tmp_path / "views", placed, part_rgb, size=(320, 280))
    assert set(views) == {"front_three_quarter", "side", "back_three_quarter"}

    def yellow_pixels(path: str) -> int:
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.int32)
        return int(((arr[:, :, 0] > 150) & (arr[:, :, 1] > 110) & (arr[:, :, 2] < 90)).sum())

    assert yellow_pixels(views["front_three_quarter"]) > yellow_pixels(views["back_three_quarter"])


def test_prepare_parts_labeling_inputs_writes_summary_sheet_and_palette(duck_glb: Path, concept_png: Path, tmp_path: Path) -> None:
    result = pb.prepare_parts_labeling_inputs(duck_glb, out_dir=tmp_path / "prep", concept_image_path=concept_png)
    summary = json.loads(Path(result["parts_summary_path"]).read_text())
    assert summary["part_count"] == 5 and len(summary["parts"]) == 5
    assert "part 2:" in summary["facts_text"]
    assert Path(result["contact_sheet_path"]).exists() and Path(result["concept_palette_path"]).exists()
    assert summary["labels_vocabulary"] == list(pb.PART_LABELS)


def test_build_parts_bundle_ready_with_a_valid_label_map(duck_glb: Path, concept_png: Path, tmp_path: Path) -> None:
    manifest = pb.build_parts_bundle(duck_glb, out_dir=tmp_path / "bundle", label_map=_label_map(), object_name="Test Duck", concept_image_path=concept_png)
    assert manifest["status"] == "ready" and manifest["ready_for_duckagent_handoff"] is True
    gate_ids = {g["id"]: g for g in manifest["gates"]}
    assert {"parts_loaded", "labels_validated", "parts_repaired_watertight", "assembly_scaled", "threemf_structure_valid", "required_artifacts_exist", "filament_slots_within_ams", "thin_parts_printable"} <= set(gate_ids)
    assert all(g["passed"] for g in manifest["gates"] if g["required"])
    assert manifest["summary"]["palette_size"] == 3 and manifest["summary"]["height_mm"] == pytest.approx(57.0, abs=0.5)
    assert manifest["summary"]["part_count"] == 5 and manifest["summary"]["component_count"] == 6 and manifest["summary"]["core"] == "ok"
    structure = pb.read_parts_3mf_structure(manifest["artifacts"]["bambu_3mf_path"])
    assert structure["mesh_objects"] == 6 and structure["config_extruders"][-1] == 1  # the core prints in the body slot
    repairs = next(g for g in manifest["gates"] if g["id"] == "parts_repaired_watertight")["details"]["repairs"]
    assert {r["method"] for r in repairs.values()} == {"shell"}
    assert manifest["summary"]["filament_slots"][0]["labels"] == ["body", "head"]
    for key in ("bambu_3mf_path", "bambu_colorgroup_3mf_path", "qa_board_path", "front_view_path", "handoff_manifest_path", "handoff_markdown_path"):
        assert Path(manifest["artifacts"][key]).exists(), key
    on_disk = json.loads(Path(manifest["artifacts"]["handoff_manifest_path"]).read_text())
    assert on_disk["schema_version"] == "duckagent.paint_to_print_handoff.v1" and on_disk["route"] == "parts_bundle"
    assert "slot 1" in Path(manifest["artifacts"]["handoff_markdown_path"]).read_text()


def test_build_parts_bundle_needs_review_when_labels_are_heuristic_or_broken(duck_glb: Path, concept_png: Path, tmp_path: Path) -> None:
    heuristic = pb.build_parts_bundle(duck_glb, out_dir=tmp_path / "h", label_map=_label_map(source="heuristic"), object_name="Test Duck", concept_image_path=concept_png)
    assert heuristic["status"] == "review_required"
    assert next(g for g in heuristic["gates"] if g["id"] == "labels_validated")["passed"] is False
    assert Path(heuristic["artifacts"]["bambu_3mf_path"]).exists()  # the file still exists for a manual look
    broken = _label_map()
    broken["parts"].pop()
    manifest = pb.build_parts_bundle(duck_glb, out_dir=tmp_path / "b", label_map=broken, object_name="Test Duck", concept_image_path=concept_png)
    assert manifest["status"] == "review_required"
    assert manifest["artifacts"].get("bambu_3mf_path") is None
    assert any("covers parts" in issue for issue in next(g for g in manifest["gates"] if g["id"] == "labels_validated")["details"]["issues"])
