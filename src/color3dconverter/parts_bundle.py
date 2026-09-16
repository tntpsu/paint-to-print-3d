"""Parts route: a part-aware model (one mesh per semantic part) → one flat color per part → printable multi-part 3MF.

The texture route quantizes a baked texture into color regions. The parts route skips
texture entirely: a part-aware generator (Tripo 3.1 ``generate_parts``) already splits
the figure into body, head, beak, eyes, wings, ...; each part gets exactly one material,
so the print has clean color boundaries and no mottling.

This module is deterministic and offline. It knows nothing about vision models: the
part labels and colors arrive as a *label map* (written by duckAgent's labeler or by
hand) and everything else is geometry:

1. ``load_scene_parts``           GLB scene → welded parts (each part one connected shell)
2. ``part_facts``                 size share, position, mirror twins → text facts for a labeler
3. ``prepare_parts_labeling_inputs``  contact sheet + per-part tiles + concept palette for the labeler
4. ``repair_parts``               close every part into a watertight solid (pymeshfix)
5. ``assemble_parts``             Y-up model units → Z-up millimeters at the target height, on the bed
6. ``build_materials``            label-map colors → ≤ N filament slots (snapped to the concept image)
7. ``write_parts_3mf``            one assembly object with N component parts + Bambu per-part extruders
8. ``render_assembly_views``      three shaded views + QA board
9. ``build_parts_bundle``         all of the above → handoff manifest with gates
"""
from __future__ import annotations

import io
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

PARTS_BUNDLE_SCHEMA_VERSION = "duckagent.parts_bundle.v1"
PARTS_LABEL_MAP_SCHEMA_VERSION = "duckagent.parts_label_map.v1"
HANDOFF_SCHEMA_VERSION = "duckagent.paint_to_print_handoff.v1"
PART_LABELS = (
    "body", "head", "helmet", "hair", "face", "beak", "eye", "visor", "wing", "tail",
    "pattern", "chest", "collar", "horn", "ear", "hat", "prop", "trim", "other",
)
DEFAULT_TARGET_HEIGHT_MM = 57.0
DEFAULT_MAX_SLOTS = 6
DEFAULT_AMS_SLOTS = 4
MIN_PRINTABLE_EXTENT_MM = 1.2
DEFAULT_RECESS_MM = 1.0
DEFAULT_SHELL_MM = 2.0
DEFAULT_CORE_ERODE_MM = 1.2
CONTACT_SHEET_VIEWS = ((-32.0, 20.0), (32.0, 20.0), (-100.0, 20.0), (100.0, 20.0), (148.0, 20.0), (212.0, 20.0))
ASSEMBLY_VIEWS = (("front_three_quarter", -32.0, 18.0), ("side", -90.0, 12.0), ("back_three_quarter", 148.0, 18.0))
HIGHLIGHT_RGB = (255, 0, 220)
BACKGROUND_RGB = (244, 240, 233)
CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


@dataclass
class ScenePart:
    index: int
    name: str
    positions: np.ndarray
    faces: np.ndarray
    watertight: bool = False
    repair: dict[str, Any] = field(default_factory=dict)

    @property
    def bounds(self) -> np.ndarray:
        return np.vstack([self.positions.min(axis=0), self.positions.max(axis=0)])

    @property
    def extents(self) -> np.ndarray:
        return self.bounds[1] - self.bounds[0]

    def copy_with(self, positions: np.ndarray, faces: np.ndarray, **changes: Any) -> "ScenePart":
        return ScenePart(index=self.index, name=self.name, positions=np.asarray(positions, dtype=np.float64), faces=np.asarray(faces, dtype=np.int64), watertight=bool(changes.get("watertight", self.watertight)), repair=dict(changes.get("repair", self.repair)))


# --------------------------------------------------------------------------- loading + facts


def _boundary_stats(faces: np.ndarray) -> tuple[int, int]:
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum()), int((counts > 2).sum())


def load_scene_parts(glb_path: str | Path, *, merge_vertices: bool = True) -> list[ScenePart]:
    """Load a part-aware GLB as a list of parts in scene order, transforms applied, seams welded.

    Part-aware generators write each part as many unwelded patches; welding coincident
    vertices (ignoring normals/uvs) turns every part back into one connected shell.
    A single-mesh file is refused: this route needs parts.
    """
    import trimesh

    loaded = trimesh.load(str(glb_path))
    if isinstance(loaded, trimesh.Scene):
        meshes = loaded.dump(concatenate=False)
    else:
        meshes = [loaded]
    if len(meshes) < 2:
        raise ValueError(f"{glb_path} has {len(meshes)} mesh; the parts route needs a multi-part model (generate_parts)")
    named = [(str((mesh.metadata or {}).get("name") or f"part_{i}"), mesh) for i, mesh in enumerate(meshes)]
    suffixes = [re.search(r"(\d+)$", name) for name, _ in named]
    if all(suffixes):
        named.sort(key=lambda item: int(re.search(r"(\d+)$", item[0]).group(1)))  # type: ignore[union-attr]
    parts: list[ScenePart] = []
    for index, (name, mesh) in enumerate(named):
        mesh = mesh.copy()
        mesh.remove_unreferenced_vertices()
        if merge_vertices:
            mesh.merge_vertices(merge_tex=True, merge_norm=True)
            mesh.remove_unreferenced_vertices()
        parts.append(ScenePart(index=index, name=name, positions=np.asarray(mesh.vertices, dtype=np.float64), faces=np.asarray(mesh.faces, dtype=np.int64), watertight=bool(mesh.is_watertight)))
    return parts


def detect_mirror_pairs(facts: list[dict[str, Any]], *, tolerance: float = 0.06, min_offset: float = 0.08) -> dict[int, int]:
    """Parts whose centers mirror across the left/right plane with similar size are twins."""
    pairs: dict[int, int] = {}
    for f in facts:
        cz = f["rel_center"][2]
        if abs(cz) < min_offset:
            continue
        for g in facts:
            if g["index"] == f["index"] or abs(g["rel_center"][2] + cz) > tolerance:
                continue
            same_place = np.linalg.norm(np.asarray(g["rel_center"][:2]) - np.asarray(f["rel_center"][:2])) < tolerance
            same_size = 0.6 < g["faces"] / max(f["faces"], 1) < 1.7
            if same_place and same_size:
                pairs[f["index"]] = g["index"]
    return pairs


def describe_part(fact: dict[str, Any]) -> str:
    x, y, z = fact["rel_center"]
    ex = fact["rel_extents"]
    side = "centered left-right" if abs(z) < 0.08 else ("on the model's left side" if z > 0 else "on the model's right side")
    front_back = "toward the front" if x > 0.12 else ("toward the back" if x < -0.12 else "mid-length")
    height = "high up" if y > 0.15 else ("low down" if y < -0.15 else "mid-height")
    size = f"{fact['faces_share'] * 100:.0f}% of the surface, spans {ex[0] * 100:.0f}% of the length, {ex[1] * 100:.0f}% of the height, {ex[2] * 100:.0f}% of the width"
    twin = f", mirror twin of part {fact['twin']}" if fact.get("twin") is not None else ""
    return f"part {fact['index']}: {size}; {front_back}, {height}, {side}{twin}"


def part_facts(parts: list[ScenePart]) -> list[dict[str, Any]]:
    """Geometry facts per part in the model's own frame (x = front, y = up, z = left/right).

    Relative centers run -0.5..0.5 of the assembled extents; the text description is
    what a labeler reads next to the contact sheet.
    """
    allv = np.vstack([p.positions for p in parts])
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    ext = np.maximum(hi - lo, 1e-9)
    mid = (lo + hi) / 2.0
    total_faces = sum(len(p.faces) for p in parts)
    facts: list[dict[str, Any]] = []
    for p in parts:
        center = (p.bounds.mean(axis=0) - mid) / ext
        boundary, nonmanifold = _boundary_stats(p.faces)
        facts.append({
            "index": p.index,
            "name": p.name,
            "faces": int(len(p.faces)),
            "vertices": int(len(p.positions)),
            "faces_share": float(len(p.faces) / max(total_faces, 1)),
            "rel_center": [float(v) for v in center],
            "rel_extents": [float(v) for v in (p.extents / ext)],
            "extents": [float(v) for v in p.extents],
            "boundary_edges": boundary,
            "nonmanifold_edges": nonmanifold,
            "watertight": bool(p.watertight),
            "twin": None,
        })
    for index, twin in detect_mirror_pairs(facts).items():
        facts[index]["twin"] = twin
    for f in facts:
        f["description"] = describe_part(f)
    return facts


# --------------------------------------------------------------------------- concept palette


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    from skimage.color import rgb2lab

    arr = np.asarray(rgb, dtype=np.float64).reshape(-1, 1, 3) / 255.0
    return rgb2lab(arr).reshape(-1, 3)


def delta_e(a: Any, b: Any) -> float:
    return float(np.linalg.norm(_rgb_to_lab(np.asarray(a, dtype=np.float64)) - _rgb_to_lab(np.asarray(b, dtype=np.float64))))


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = str(value or "").strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"not a #rrggbb color: {value!r}")
    return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb: Any) -> str:
    r, g, b = (int(v) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def extract_concept_palette(image_path: str | Path, *, max_colors: int = 12, working_size: int = 384, seed: int = 0) -> dict[str, Any]:
    """Cluster the concept render into candidate colors with the backdrop removed.

    The backdrop is whichever clusters dominate a 4% border band (plus clusters within a
    short RGB distance of them, which catches the soft ground shadow). Shading still splits
    one material into lit and shadowed clusters, so this is *evidence* for snapping a
    labeler's colors, not the material list itself.
    """
    from PIL import Image
    from sklearn.cluster import KMeans

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((working_size, working_size))
    px = np.asarray(image, dtype=np.float32) / 255.0
    h, w, _ = px.shape
    flat = px.reshape(-1, 3)
    k0 = max_colors + 3
    km = KMeans(n_clusters=k0, n_init=4, random_state=seed).fit(flat)
    labels = km.labels_.reshape(h, w)
    band = max(2, int(0.04 * min(h, w)))
    border = np.concatenate([labels[:band].ravel(), labels[-band:].ravel(), labels[:, :band].ravel(), labels[:, -band:].ravel()])
    share = np.bincount(border, minlength=k0) / border.size
    backdrop = {int(i) for i, s in enumerate(share) if s >= 0.15}
    centers = km.cluster_centers_
    for i in range(k0):
        if i not in backdrop and any(np.linalg.norm(centers[i] - centers[j]) < 0.10 for j in backdrop):
            backdrop.add(i)
    mask = ~np.isin(labels, sorted(backdrop))
    fg = flat[mask.ravel()]
    if len(fg) < 50:
        raise ValueError("concept image has no foreground after backdrop removal")
    k = min(max_colors, max(1, len(fg) // 50))
    km2 = KMeans(n_clusters=k, n_init=6, random_state=seed).fit(fg)
    weights = np.bincount(km2.labels_, minlength=k) / len(fg)
    rows = []
    for center, weight in sorted(zip(km2.cluster_centers_, weights, strict=False), key=lambda t: -t[1]):
        rgb = tuple(int(round(float(v) * 255)) for v in center)
        rows.append({"rgb": rgb, "hex": rgb_to_hex(rgb), "pixel_share": float(weight)})
    backdrop_rgb = tuple(int(round(float(v) * 255)) for v in centers[max(backdrop, key=lambda i: share[i])]) if backdrop else None
    return {
        "method": "kmeans_border_backdrop",
        "working_size": working_size,
        "foreground_share": float(mask.mean()),
        "backdrop": {"rgb": backdrop_rgb, "hex": rgb_to_hex(backdrop_rgb)} if backdrop_rgb else None,
        "clusters": rows,
    }


def snap_color(rgb: Any, palette_rows: list[dict[str, Any]], *, max_delta_e: float = 15.0) -> dict[str, Any]:
    """Nearest concept-image cluster to a labeler's color, kept only when it is close enough.

    Small light regions (cream horns) rarely get their own pixel cluster; a loose threshold
    would drag them onto the nearest tan, so the labeler's own color wins beyond 15 ΔE.
    """
    if not palette_rows:
        return {"rgb": tuple(int(v) for v in rgb), "snapped": False, "delta_e": None}
    best = min(palette_rows, key=lambda row: delta_e(row["rgb"], rgb))
    distance = delta_e(best["rgb"], rgb)
    if distance <= max_delta_e:
        return {"rgb": tuple(int(v) for v in best["rgb"]), "snapped": True, "delta_e": distance}
    return {"rgb": tuple(int(v) for v in rgb), "snapped": False, "delta_e": distance}


# --------------------------------------------------------------------------- label map + materials


def validate_label_map_shape(label_map: dict[str, Any], part_count: int) -> list[str]:
    """Structural checks only (ids, labels, colors); semantic checks live with the labeler."""
    issues: list[str] = []
    rows = label_map.get("parts") if isinstance(label_map, dict) else None
    if not isinstance(rows, list) or not rows:
        return ["label map has no parts"]
    seen = sorted(int(r.get("n", -1)) for r in rows if isinstance(r, dict))
    if seen != list(range(part_count)):
        issues.append(f"label map covers parts {seen}, model has parts 0..{part_count - 1}")
    for r in rows:
        if not isinstance(r, dict):
            issues.append("label map row is not an object")
            continue
        if str(r.get("label")) not in PART_LABELS:
            issues.append(f"part {r.get('n')}: unknown label {r.get('label')!r}")
        try:
            hex_to_rgb(str(r.get("hex")))
        except ValueError:
            issues.append(f"part {r.get('n')}: bad color {r.get('hex')!r}")
    return issues


def build_materials(label_map: dict[str, Any], palette_rows: list[dict[str, Any]] | None, *, merge_delta_e: float = 12.0, max_slots: int = DEFAULT_MAX_SLOTS) -> dict[str, Any]:
    """Turn per-part colors into filament slots: snap to the concept image, merge near-identical colors.

    Slots are ordered by covered surface (largest first) so slot 1 is the body color.
    When more distinct colors than slots exist, the closest pair merges until it fits.
    """
    rows = sorted((r for r in label_map.get("parts", []) if isinstance(r, dict)), key=lambda r: int(r.get("n", 0)))
    entries = []
    for r in rows:
        want = hex_to_rgb(str(r.get("hex")))
        snapped = snap_color(want, palette_rows or [])
        entries.append({"n": int(r["n"]), "label": str(r.get("label")), "color_name": str(r.get("color_name") or ""), "requested_rgb": want, "rgb": snapped["rgb"], "snapped": snapped["snapped"], "delta_e": snapped["delta_e"], "weight": float(r.get("weight", 1.0))})
    materials: list[dict[str, Any]] = []
    for e in entries:
        hit = next((m for m in materials if delta_e(m["rgb"], e["rgb"]) < merge_delta_e), None)
        if hit is None:
            hit = {"name": e["color_name"] or rgb_to_hex(e["rgb"]), "rgb": tuple(e["rgb"]), "parts": [], "weight": 0.0}
            materials.append(hit)
        hit["parts"].append(e["n"])
        hit["weight"] += e["weight"]
    merged_pairs: list[str] = []
    while len(materials) > max_slots:
        best = min(((a, b) for a in materials for b in materials if a is not b), key=lambda ab: delta_e(ab[0]["rgb"], ab[1]["rgb"]))
        keep, drop = (best if best[0]["weight"] >= best[1]["weight"] else (best[1], best[0]))
        keep["parts"].extend(drop["parts"])
        keep["weight"] += drop["weight"]
        merged_pairs.append(f"{drop['name']} → {keep['name']}")
        materials.remove(drop)
    materials.sort(key=lambda m: -m["weight"])
    slots = []
    part_slot: dict[int, int] = {}
    for slot_index, m in enumerate(materials, start=1):
        for n in sorted(m["parts"]):
            part_slot[n] = slot_index
        labels = sorted({e["label"] for e in entries if e["n"] in m["parts"]})
        slots.append({"slot": slot_index, "name": m["name"], "rgb": [int(v) for v in m["rgb"]], "hex": rgb_to_hex(m["rgb"]), "parts": sorted(m["parts"]), "labels": labels})
    return {"slots": slots, "part_slot": part_slot, "parts": entries, "merged_to_fit": merged_pairs}


# --------------------------------------------------------------------------- repair


def _mesh_report(positions: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    import trimesh

    mesh = trimesh.Trimesh(positions, faces, process=False)
    boundary, nonmanifold = _boundary_stats(np.asarray(faces, dtype=np.int64))
    bodies = len(mesh.split(only_watertight=False)) if len(faces) else 0
    return {"faces": int(len(faces)), "vertices": int(len(positions)), "watertight": bool(mesh.is_watertight), "bodies": int(bodies), "boundary_edges": boundary, "nonmanifold_edges": nonmanifold, "volume": float(mesh.volume) if mesh.is_watertight else None}


def _extents_delta(before: ScenePart, positions: np.ndarray, *, allowance: float = 0.0) -> float:
    """Largest relative change of the bounding box, after forgiving ``allowance`` (absolute) per axis.

    A recessed cap legitimately deepens a surface patch by the recess depth, so that much
    growth is not a defect; anything beyond it (a fix that ate or ballooned the part) is.
    """
    after = np.asarray(positions).max(axis=0) - np.asarray(positions).min(axis=0)
    change = np.maximum(np.abs(after - before.extents) - allowance, 0.0)
    return float(np.max(change / np.maximum(before.extents, 1e-9)))


def boundary_loops(faces: np.ndarray) -> list[list[int]]:
    """Closed boundary loops as ordered vertex cycles, following each open edge in its face's winding."""
    faces = np.asarray(faces, dtype=np.int64)
    directed = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    undirected = np.sort(directed, axis=1)
    _, first, counts = np.unique(undirected, axis=0, return_index=True, return_counts=True)
    open_edges = directed[first[counts == 1]]
    nxt: dict[int, int] = {}
    for a, b in open_edges.tolist():
        nxt.setdefault(int(a), int(b))
    loops: list[list[int]] = []
    seen: set[int] = set()
    for start in list(nxt):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        cur = nxt[start]
        while cur != start and cur not in seen and cur in nxt:
            loop.append(cur)
            seen.add(cur)
            cur = nxt[cur]
        if cur == start and len(loop) >= 3:
            loops.append(loop)
    return loops


def recess_boundary_loops(positions: np.ndarray, faces: np.ndarray, *, depth: float, center: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Extrude every open loop inward by ``depth`` (along the inward vertex normals) with a side wall.

    A part cut off a closed figure is an open patch; capping its loops flat puts the cap on
    the surface of the neighbouring part (a ring-shaped visor trim gets a membrane across
    the visor). Pushing the loop inward first leaves the cap ``depth`` inside the model,
    where the other part covers it. Returns the extended mesh (still open at the recessed
    loops) and the number of loops moved.

    Generated parts do not come with reliable winding, so the face winding is made
    consistent first and "inward" is the side of the surface facing ``center`` (the whole
    figure's center; the part's own center when not given).
    """
    import trimesh

    positions = np.asarray(positions, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if not len(faces) or depth <= 0:
        return positions, faces, 0
    mesh = trimesh.Trimesh(positions, faces, process=False)
    trimesh.repair.fix_normals(mesh, multibody=True)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    loops = boundary_loops(faces)
    if not loops:
        return positions, faces, 0
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    loop_vertices = np.unique(np.concatenate([np.asarray(l) for l in loops]))
    # Which side is inside? The part's own box center is on the inner side of a bowl, a dome,
    # a ring or a band; when the part is too flat for that to be decisive, the figure's center decides.
    candidates = [(positions.min(axis=0) + positions.max(axis=0)) / 2.0]
    if center is not None:
        candidates.append(np.asarray(center, dtype=np.float64))
    agreement = []
    for reference in candidates:
        toward = reference - positions[loop_vertices]
        agreement.append(float(np.mean(np.sign((normals[loop_vertices] * toward).sum(axis=1)))))
    decisive = max(range(len(candidates)), key=lambda i: abs(agreement[i]))
    if agreement[decisive] > 0:
        normals = -normals  # winding was inside-out: flip so normals point away from the inside
    new_positions = [positions]
    new_faces = [faces]
    next_index = len(positions)
    for loop in loops:
        idx = np.asarray(loop, dtype=np.int64)
        inward = -normals[idx]
        norms = np.linalg.norm(inward, axis=1, keepdims=True)
        inward = np.where(norms > 1e-9, inward / np.maximum(norms, 1e-9), 0.0)
        copies = positions[idx] + inward * depth
        copy_index = np.arange(next_index, next_index + len(idx))
        next_index += len(idx)
        a = idx
        b = np.roll(idx, -1)
        a2 = copy_index
        b2 = np.roll(copy_index, -1)
        new_positions.append(copies)
        new_faces.append(np.column_stack([b, a, a2]))
        new_faces.append(np.column_stack([b, a2, b2]))
    return np.vstack(new_positions), np.vstack(new_faces), len(loops)


def _oriented_normals(positions: np.ndarray, faces: np.ndarray, *, center: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Consistently wound faces, outward vertex normals and the open loops of a surface patch.

    The winding is made consistent (never inverted by a volume guess: the patch is open), then
    "outward" is decided once for the whole patch by an area-weighted vote of the face normals
    against the direction away from the figure's center (the part's own box center when no
    figure center is given). A compact figure has every patch facing away from its center on
    average, including a concave dish, which fools any per-part center test.
    """
    import trimesh

    mesh = trimesh.Trimesh(positions, faces, process=False)
    trimesh.repair.fix_winding(mesh)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    loops = boundary_loops(faces)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    face_centers = positions[faces].mean(axis=1)
    reference = np.asarray(center, dtype=np.float64) if center is not None else (positions.min(axis=0) + positions.max(axis=0)) / 2.0
    away = face_centers - reference
    vote = float(np.sum(np.asarray(mesh.area_faces) * np.sign((face_normals * away).sum(axis=1))))
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    if vote < 0:
        normals = -normals  # the patch faces the inside: flip
        faces = faces[:, ::-1]
    return faces, normals, loops


def solidify_part(positions: np.ndarray, faces: np.ndarray, *, thickness: float, center: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Thicken a surface patch inward into a closed shell: outer surface, inner offset surface, side walls.

    This keeps the printed surface exactly where the generator put it and puts nothing in
    front of any neighbour: a ring stays a ring, a visor stays a dish. Flat caps across the
    open loops do not (they cut through concave dishes and coincide with the neighbour's cap).
    """
    positions = np.asarray(positions, dtype=np.float64)
    faces, normals, loops = _oriented_normals(positions, np.asarray(faces, dtype=np.int64), center=center)
    n = len(positions)
    inner = positions - normals * float(thickness)
    parts_faces = [faces, faces[:, ::-1] + n]
    for loop in loops:
        a = np.asarray(loop, dtype=np.int64)
        b = np.roll(a, -1)
        a2, b2 = a + n, b + n
        parts_faces.append(np.column_stack([b, a, a2]))
        parts_faces.append(np.column_stack([b, a2, b2]))
    return np.vstack([positions, inner]), np.vstack(parts_faces), {"thickness": float(thickness), "loops": len(loops)}


def shell_thickness_for(part: ScenePart, *, max_thickness: float, min_thickness: float = 0.8, fraction: float = 0.45) -> float:
    """Thin parts (eyes) get thin shells; everything else the standard shell."""
    return float(min(max_thickness, max(min_thickness, fraction * float(part.extents.min()))))


def build_core(parts: list[ScenePart], *, pitch: float = 0.6, erode: float = 1.2, samples_per_mm2: float = 60.0) -> tuple[ScenePart | None, dict[str, Any]]:
    """A solid strictly inside the figure, so the shells sit on a filled body and the print is not hollow.

    The union of all part surfaces is the figure's closed outer surface; sampling it densely
    into a voxel grid, filling the enclosed volume and eroding it by ``erode`` gives a solid
    that never reaches the surface. Marching cubes turns it back into a watertight mesh.
    """
    import trimesh
    from scipy import ndimage
    from skimage import measure

    positions, faces, _ = _concatenate(parts)
    mesh = trimesh.Trimesh(positions, faces, process=False)
    count = int(min(6_000_000, max(200_000, mesh.area * samples_per_mm2)))
    points, _ = trimesh.sample.sample_surface(mesh, count)
    lo = mesh.bounds[0] - 2 * pitch
    shape = np.ceil((mesh.bounds[1] - lo) / pitch).astype(int) + 3
    grid = np.zeros(tuple(int(v) for v in shape), dtype=bool)
    idx = np.floor((points - lo) / pitch).astype(int)
    idx = idx[np.all((idx >= 0) & (idx < shape), axis=1)]
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    grid = ndimage.binary_closing(grid, iterations=1)
    filled = ndimage.binary_fill_holes(grid)
    iterations = max(1, int(round(erode / pitch)))
    core = ndimage.binary_erosion(filled, iterations=iterations)
    info = {"pitch_mm": pitch, "erode_mm": iterations * pitch, "samples": count, "filled_voxels": int(filled.sum()), "core_voxels": int(core.sum())}
    if core.sum() < 8:
        return None, {**info, "status": "too_small"}
    verts, tris, _, _ = measure.marching_cubes(core.astype(np.float32), level=0.5, spacing=(pitch, pitch, pitch))
    core_mesh = trimesh.Trimesh(verts + lo, tris, process=True)
    core_mesh.remove_unreferenced_vertices()
    bodies = core_mesh.split(only_watertight=False)
    if len(bodies) > 1:
        core_mesh = max(bodies, key=lambda b: len(b.faces))
    inside = bool(np.all(core_mesh.bounds[0] >= mesh.bounds[0] - 1e-6) and np.all(core_mesh.bounds[1] <= mesh.bounds[1] + 1e-6))
    info.update({"status": "ok" if core_mesh.is_watertight and inside else "invalid", "faces": int(len(core_mesh.faces)), "watertight": bool(core_mesh.is_watertight), "inside_bounds": inside})
    if info["status"] != "ok":
        return None, info
    part = ScenePart(index=max(p.index for p in parts) + 1, name="core", positions=np.asarray(core_mesh.vertices, dtype=np.float64), faces=np.asarray(core_mesh.faces, dtype=np.int64), watertight=True, repair={"method": "voxel_core", "after": info})
    return part, info


def repair_part(part: ScenePart, *, shell_thickness: float = 0.0, recess_depth: float = 0.0, max_extents_delta: float = 0.03, center: np.ndarray | None = None) -> ScenePart:
    """Close one part into a watertight solid, trying the least invasive fix first.

    1. ``shell``: thicken the patch inward by ``shell_thickness`` (``solidify_part``), the
       construction that keeps the printed surface exact; leftover pinholes get a pymeshfix fill.
    2. recess the open loops inward by ``recess_depth`` and fill them flat.
    3. the same fill without the recess.
    4. pymeshfix full ``repair`` (joins components, removes self-intersections).
    A result that grows or shrinks the part by more than ``max_extents_delta`` (beyond what
    the method legitimately moves) is refused: a "repair" that eats a horn is worse than an
    open horn the operator can see.
    """
    import pymeshfix

    attempts: list[dict[str, Any]] = []
    positions = np.asarray(part.positions, dtype=np.float64)
    faces = np.asarray(part.faces, dtype=np.int64)
    if part.watertight and _mesh_report(positions, faces)["bodies"] == 1:
        return part.copy_with(positions, faces, watertight=True, repair={"method": "already_watertight", "attempts": attempts, "after": _mesh_report(positions, faces)})

    def _try(method: str, fn, *, allowance: float = 0.0) -> ScenePart | None:
        started = time.time()
        try:
            v, f = fn()
        except Exception as exc:  # noqa: BLE001 - the ladder records and moves on
            attempts.append({"method": method, "error": str(exc)[:200], "seconds": round(time.time() - started, 2)})
            return None
        v = np.asarray(v, dtype=np.float64)
        f = np.asarray(f, dtype=np.int64)
        report = _mesh_report(v, f)
        report.update({"method": method, "seconds": round(time.time() - started, 2), "extents_delta": _extents_delta(part, v, allowance=allowance)})
        attempts.append(report)
        if report["watertight"] and report["bodies"] == 1 and report["extents_delta"] <= max_extents_delta:
            return part.copy_with(v, f, watertight=True, repair={"method": method, "attempts": attempts, "after": report})
        return None

    def _fill(v: np.ndarray, f: np.ndarray):
        tin = pymeshfix.PyTMesh()
        tin.load_array(np.ascontiguousarray(v, dtype=np.float64), np.ascontiguousarray(f, dtype=np.int32))
        tin.fill_small_boundaries(nbe=0, refine=False)
        out_v, out_f = tin.return_arrays()
        return _weld(out_v, out_f)

    def _shell():
        v, f, info = solidify_part(positions, faces, thickness=shell_thickness, center=center)
        v, f = _weld(v, f)
        if _boundary_stats(f)[0] > 0:
            v, f = _fill(v, f)
        return v, f

    def _recessed_fill():
        v, f, moved = recess_boundary_loops(positions, faces, depth=recess_depth, center=center)
        if moved == 0:
            raise RuntimeError("no closed boundary loops to recess")
        return _fill(v, f)

    def _plain_fill():
        return _fill(positions, faces)

    def _full_repair():
        fixer = pymeshfix.MeshFix(positions.copy(), faces.astype(np.int32).copy())
        fixer.repair(joincomp=True, remove_smallest_components=False)
        return _weld(np.asarray(fixer.points, dtype=np.float64), np.asarray(fixer.faces, dtype=np.int64))

    # a shell's inner surface may stick out by the thickness on either side of a curled lip
    # (2 x thickness per axis); a recess may shrink a rounded band from both edges (2 x depth)
    ladder = ([("shell", _shell, shell_thickness * 2.1)] if shell_thickness > 0 else []) + ([("recessed_fill", _recessed_fill, recess_depth * 2.1)] if recess_depth > 0 else []) + [("pymeshfix_fill_boundaries", _plain_fill, 0.0), ("pymeshfix_repair", _full_repair, 0.0)]
    for method, fn, allowance in ladder:
        fixed = _try(method, fn, allowance=allowance)
        if fixed is not None:
            return fixed
    return part.copy_with(positions, faces, watertight=False, repair={"method": "failed", "attempts": attempts, "after": _mesh_report(positions, faces)})


def _weld(positions: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices and drop degenerate faces so watertightness is judged on shared edges."""
    import trimesh

    mesh = trimesh.Trimesh(np.asarray(positions, dtype=np.float64), np.asarray(faces, dtype=np.int64), process=True)
    mesh.remove_unreferenced_vertices()
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def repair_parts(parts: list[ScenePart], *, shell_thickness: float = 0.0, **kwargs: Any) -> list[ScenePart]:
    """Repair every part; shells and caps are oriented against the whole figure's center."""
    allv = np.vstack([p.positions for p in parts])
    kwargs.setdefault("center", (allv.min(axis=0) + allv.max(axis=0)) / 2.0)
    return [repair_part(p, shell_thickness=(shell_thickness_for(p, max_thickness=shell_thickness) if shell_thickness > 0 else 0.0), **kwargs) for p in parts]


# --------------------------------------------------------------------------- assembly (Z-up millimeters)

Y_UP_TO_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # (x, y, z) -> (x, -z, y)


def assemble_parts(parts: list[ScenePart], *, target_height_mm: float = DEFAULT_TARGET_HEIGHT_MM) -> tuple[list[ScenePart], dict[str, Any]]:
    """Rotate the Y-up model to Z-up, scale uniformly to the target height, stand it on z=0 centered in x/y.

    Already Z-up millimetre parts (a second pass after repair) are handled by ``place_parts``.
    """
    allv = np.vstack([p.positions for p in parts]) @ Y_UP_TO_Z_UP.T
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    height_units = float(hi[2] - lo[2])
    if height_units <= 0:
        raise ValueError("model has no height")
    scale = float(target_height_mm) / height_units
    shift = np.array([-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]])
    placed = [p.copy_with((p.positions @ Y_UP_TO_Z_UP.T + shift) * scale, p.faces) for p in parts]
    allmm = np.vstack([p.positions for p in placed])
    bounds = np.vstack([allmm.min(axis=0), allmm.max(axis=0)])
    info = {
        "target_height_mm": float(target_height_mm),
        "scale_mm_per_unit": scale,
        "height_mm": float(bounds[1][2] - bounds[0][2]),
        "bounds_mm": bounds.round(3).tolist(),
        "footprint_mm": [float(bounds[1][0] - bounds[0][0]), float(bounds[1][1] - bounds[0][1])],
        "up_axis": "z",
        "source_up_axis": "y",
    }
    return placed, info


def place_parts(parts: list[ScenePart], *, target_height_mm: float = DEFAULT_TARGET_HEIGHT_MM) -> tuple[list[ScenePart], dict[str, Any]]:
    """Re-seat already Z-up millimetre parts (after repair) on z=0, centered, and measure them.

    Repair never rescales, so the height is reported as measured rather than forced.
    """
    allv = np.vstack([p.positions for p in parts])
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    shift = np.array([-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]])
    placed = [p.copy_with(p.positions + shift, p.faces, watertight=p.watertight, repair=p.repair) for p in parts]
    allmm = np.vstack([p.positions for p in placed])
    bounds = np.vstack([allmm.min(axis=0), allmm.max(axis=0)])
    info = {
        "target_height_mm": float(target_height_mm),
        "height_mm": float(bounds[1][2] - bounds[0][2]),
        "bounds_mm": bounds.round(3).tolist(),
        "footprint_mm": [float(bounds[1][0] - bounds[0][0]), float(bounds[1][1] - bounds[0][1])],
        "up_axis": "z",
    }
    return placed, info


# --------------------------------------------------------------------------- rendering


def _project(positions: np.ndarray, faces: np.ndarray, *, yaw_deg: float, pitch_deg: float, size: tuple[int, int], up_axis: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    P = np.asarray(positions, dtype=np.float32)
    if up_axis == "z":
        P = P[:, [0, 2, 1]].copy()
        P[:, 2] *= -1.0
    center = (P.min(axis=0) + P.max(axis=0)) / 2.0
    span = float(np.max(P.max(axis=0) - P.min(axis=0)))
    N = (P - center) / max(span, 1e-6)
    ay, ax = math.radians(yaw_deg), math.radians(pitch_deg)
    ry = np.array([[math.cos(ay), 0.0, math.sin(ay)], [0.0, 1.0, 0.0], [-math.sin(ay), 0.0, math.cos(ay)]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(ax), -math.sin(ax)], [0.0, math.sin(ax), math.cos(ax)]], dtype=np.float32)
    T = N @ ry.T @ rx.T
    pr = T[:, :2].copy()
    pr[:, 1] *= -1.0
    w, h = size
    margin = 0.08 * min(w, h)
    k = min((w - 2 * margin), (h - 2 * margin)) * 0.9
    pr = pr * k + np.array([w / 2.0, h / 2.0], dtype=np.float32)
    fv = T[faces]
    depth = fv[:, :, 2].mean(axis=1)
    n = np.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0])
    ln = np.linalg.norm(n, axis=1)
    ok = ln > 1e-8
    n[ok] /= ln[ok][:, None]
    n[(n @ np.array([0.0, 0.0, 1.0], dtype=np.float32)) < 0.0] *= -1.0
    light = np.array([0.45, 0.55, 1.0], dtype=np.float32)
    light /= np.linalg.norm(light)
    lit = 0.42 + 0.58 * np.clip(n @ light, 0.0, 1.0)
    return pr[faces], depth, lit


def render_view(positions: np.ndarray, faces: np.ndarray, face_colors: np.ndarray, *, yaw_deg: float, pitch_deg: float = 20.0, size: tuple[int, int] = (960, 720), up_axis: str = "z", background: tuple[int, int, int] = BACKGROUND_RGB):
    """Painter's-order software render (PIL polygons), same shading as the texture route's preview."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, background)
    if len(positions) == 0 or len(faces) == 0:
        return image
    fp, depth, lit = _project(positions, faces, yaw_deg=yaw_deg, pitch_deg=pitch_deg, size=size, up_axis=up_axis)
    draw = ImageDraw.Draw(image)
    colors = np.asarray(face_colors, dtype=np.float32)
    polys = fp.tolist()
    for fi in np.argsort(depth).tolist():
        col = np.clip(np.rint(colors[fi] * lit[fi]), 0, 255).astype(np.uint8)
        draw.polygon([tuple(p) for p in polys[fi]], fill=tuple(int(v) for v in col))
    return image


def render_id_buffers(positions: np.ndarray, faces: np.ndarray, face_ids: np.ndarray, *, yaw_deg: float, pitch_deg: float, size: tuple[int, int], up_axis: str = "y") -> tuple[np.ndarray, np.ndarray]:
    """One lit greyscale buffer plus one part-id buffer per view; every part tile derives from them by pixel ops."""
    from PIL import Image, ImageDraw

    fp, depth, lit = _project(positions, faces, yaw_deg=yaw_deg, pitch_deg=pitch_deg, size=size, up_axis=up_axis)
    lit_img = Image.new("L", size, 0)
    id_img = Image.new("I", size, -1)
    lit_draw = ImageDraw.Draw(lit_img)
    id_draw = ImageDraw.Draw(id_img)
    polys = fp.tolist()
    for fi in np.argsort(depth).tolist():
        poly = [tuple(p) for p in polys[fi]]
        lit_draw.polygon(poly, fill=int(round(255 * float(lit[fi]))))
        id_draw.polygon(poly, fill=int(face_ids[fi]))
    return np.asarray(lit_img, dtype=np.float32) / 255.0, np.asarray(id_img, dtype=np.int32)


def _tile_from_buffers(lit: np.ndarray, ids: np.ndarray, part_index: int, *, highlight: tuple[int, int, int] = HIGHLIGHT_RGB):
    from PIL import Image

    h, w = ids.shape
    out = np.empty((h, w, 3), dtype=np.float32)
    out[:] = np.array(BACKGROUND_RGB, dtype=np.float32)
    drawn = ids >= 0
    out[drawn] = np.array([205.0, 205.0, 205.0], dtype=np.float32) * lit[drawn][:, None]
    mask = ids == part_index
    out[mask] = np.array(highlight, dtype=np.float32) * (0.55 + 0.45 * lit[mask])[:, None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _concatenate(parts: list[ScenePart]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.cumsum([0] + [len(p.positions) for p in parts])
    positions = np.vstack([p.positions for p in parts])
    faces = np.vstack([p.faces + offsets[i] for i, p in enumerate(parts)])
    ids = np.concatenate([np.full(len(p.faces), p.index, dtype=np.int32) for p in parts])
    return positions, faces, ids


def write_parts_contact_sheet(path: str | Path, parts: list[ScenePart], *, tile_size: tuple[int, int] = (420, 360), columns: int = 3, views: tuple[tuple[float, float], ...] = CONTACT_SHEET_VIEWS, tiles_dir: str | Path | None = None) -> dict[str, Any]:
    """Numbered contact sheet for a labeler: each part in magenta from the view where it is most visible.

    The opposite view is inset top-right so a part hidden from the front still reads.
    Per-part zoom tiles (for a targeted second look) go to ``tiles_dir`` when given.
    """
    from PIL import Image, ImageDraw

    positions, faces, ids = _concatenate(parts)
    buffers = [render_id_buffers(positions, faces, ids, yaw_deg=yaw, pitch_deg=pitch, size=tile_size, up_axis="y") for yaw, pitch in views]
    indices = [p.index for p in parts]
    visible = np.array([[int((idb == i).sum()) for i in indices] for _, idb in buffers])
    rows = max(1, math.ceil(len(parts) / columns))
    sheet = Image.new("RGB", (columns * tile_size[0], rows * tile_size[1]), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    best_views: dict[int, int] = {}
    tile_paths: dict[int, str] = {}
    for slot, part_index in enumerate(indices):
        v = int(np.argmax(visible[:, slot]))
        best_views[part_index] = v
        opposite = min((k for k in range(len(views)) if k != v), key=lambda k: abs(((views[k][0] - views[v][0] + 180.0) % 360.0) - 180.0), default=v)
        tile = _tile_from_buffers(*buffers[v], part_index)
        inset = _tile_from_buffers(*buffers[opposite], part_index).resize((tile_size[0] // 3, tile_size[1] // 3))
        tile.paste(inset, (tile_size[0] - tile_size[0] // 3 - 4, 4))
        x, y = (slot % columns) * tile_size[0], (slot // columns) * tile_size[1]
        sheet.paste(tile, (x, y))
        draw.rectangle([x + 6, y + 6, x + 96, y + 44], fill=(0, 0, 0))
        draw.text((x + 14, y + 14), f"part {part_index}", fill=(255, 255, 255))
        draw.rectangle([x, y, x + tile_size[0] - 1, y + tile_size[1] - 1], outline=(120, 120, 120))
        if tiles_dir is not None:
            lit, idb = buffers[v]
            zoom = _tile_from_buffers(lit, idb, part_index)
            ys, xs = np.nonzero(idb == part_index)
            if len(xs):
                pad = 90
                zoom = zoom.crop((max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad), min(tile_size[0], int(xs.max()) + pad), min(tile_size[1], int(ys.max()) + pad)))
            tile_path = Path(tiles_dir) / f"part_{part_index}.png"
            tile_path.parent.mkdir(parents=True, exist_ok=True)
            zoom.save(tile_path)
            tile_paths[part_index] = str(tile_path)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return {"path": str(out), "best_views": {str(k): v for k, v in best_views.items()}, "visible_pixels": {str(i): [int(v) for v in visible[:, s]] for s, i in enumerate(indices)}, "tiles": {str(k): v for k, v in tile_paths.items()}, "highlight_rgb": list(HIGHLIGHT_RGB)}


def render_assembly_views(out_dir: str | Path, parts: list[ScenePart], part_rgb: dict[int, Any], *, views: tuple[tuple[str, float, float], ...] = ASSEMBLY_VIEWS, up_axis: str = "z", size: tuple[int, int] = (720, 640)) -> dict[str, str]:
    positions, faces, ids = _concatenate(parts)
    face_colors = np.vstack([np.tile(np.asarray(part_rgb[p.index], dtype=np.float32), (len(p.faces), 1)) for p in parts])
    del ids
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, yaw, pitch in views:
        image = render_view(positions, faces, face_colors, yaw_deg=yaw, pitch_deg=pitch, size=size, up_axis=up_axis)
        target = out / f"view_{name}.png"
        image.save(target)
        paths[name] = str(target)
    return paths


def write_parts_qa_board(path: str | Path, *, concept_image_path: str | Path | None, view_paths: dict[str, str], slots: list[dict[str, Any]], parts_rows: list[dict[str, Any]], status: str, notes: list[str]) -> Path:
    """Concept beside the three model views, the slot legend and the gate notes: what the operator approves."""
    from PIL import Image, ImageDraw

    panel = (460, 400)
    tiles = []
    if concept_image_path and Path(concept_image_path).exists():
        img = Image.open(concept_image_path).convert("RGB")
        img.thumbnail(panel)
        tiles.append(("concept", img))
    for name, p in view_paths.items():
        img = Image.open(p).convert("RGB")
        img.thumbnail(panel)
        tiles.append((f"model: {name.replace('_', ' ')}", img))
    width = 20 + sum(t.width + 10 for _, t in tiles)
    legend_h = 44 * (len(slots) + 1) + 24 * (len(notes) + 1)
    board = Image.new("RGB", (max(width, 900), panel[1] + 60 + legend_h), (255, 255, 255))
    draw = ImageDraw.Draw(board)
    x = 10
    for title, t in tiles:
        draw.text((x, 8), title, fill=(0, 0, 0))
        board.paste(t, (x, 28))
        x += t.width + 10
    y = panel[1] + 44
    draw.text((10, y - 16), f"status: {status}", fill=(0, 0, 0))
    label_of = {int(r["n"]): str(r.get("label")) for r in parts_rows}
    for s in slots:
        draw.rectangle([10, y + 6, 50, y + 36], fill=tuple(s["rgb"]), outline=(0, 0, 0))
        parts_text = ", ".join(f"{n}:{label_of.get(n, '?')}" for n in s["parts"])
        draw.text((60, y + 14), f"slot {s['slot']}  {s['name']}  {s['hex']}  parts {parts_text}", fill=(0, 0, 0))
        y += 44
    for note in notes:
        draw.text((10, y + 4), f"- {note}"[:180], fill=(60, 60, 60))
        y += 24
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    board.save(out)
    return out


# --------------------------------------------------------------------------- 3MF


def _stream_mesh_xml(handle: io.TextIOBase, positions: np.ndarray, faces: np.ndarray) -> None:
    handle.write("<mesh><vertices>")
    pos = np.asarray(positions, dtype=np.float64)
    for start in range(0, len(pos), 20000):
        chunk = pos[start:start + 20000]
        handle.write("".join(f'<vertex x="{x:.4f}" y="{y:.4f}" z="{z:.4f}"/>' for x, y, z in chunk.tolist()))
    handle.write("</vertices><triangles>")
    tri = np.asarray(faces, dtype=np.int64)
    for start in range(0, len(tri), 20000):
        chunk = tri[start:start + 20000]
        handle.write("".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in chunk.tolist()))
    handle.write("</triangles></mesh>")


def _xml_attr(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def write_parts_3mf(output_path: str | Path, parts: list[ScenePart], part_slot: dict[int, int], slots: list[dict[str, Any]], *, object_name: str, include_bambu_config: bool = True) -> Path:
    """One assembly object whose components are the parts, plus Bambu's per-part extruder config.

    Standard 3MF core: mesh objects 1..N, assembly object N+1 with N identity components,
    one build item. Bambu Studio (and slicers built on the same importer) load that as ONE
    object with N parts, which is what a multi-color figure needs: arrange and scale keep
    the parts together. ``Metadata/model_settings.config`` carries the filament slot per
    part (``extruder``); part ids there equal the mesh object ids. Base materials give
    other tools a display color per part. Streamed, so half-million-triangle models do
    not build a DOM.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(parts, key=lambda p: p.index)
    mesh_ids = {p.index: i + 1 for i, p in enumerate(ordered)}
    assembly_id = len(ordered) + 1
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
        '<Default Extension="config" ContentType="text/xml"/></Types>'
    )
    relationships = (
        f'<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="{REL_NS}">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>'
    )
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        with archive.open("3D/3dmodel.model", "w") as raw, io.TextIOWrapper(raw, encoding="utf-8", newline="\n") as handle:
            handle.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            handle.write(f'<model unit="millimeter" xml:lang="en-US" xmlns="{CORE_NS}">')
            handle.write(f'<metadata name="Title">{_xml_attr(object_name)}</metadata>')
            handle.write("<resources>")
            handle.write('<basematerials id="1">')
            for s in slots:
                r, g, b = s["rgb"]
                handle.write(f'<base name="{_xml_attr(s["name"])}" displaycolor="#{r:02X}{g:02X}{b:02X}FF"/>')
            handle.write("</basematerials>")
            for p in ordered:
                pindex = max(0, int(part_slot.get(p.index, 1)) - 1)
                handle.write(f'<object id="{mesh_ids[p.index]}" type="model" name="{_xml_attr(p.name)}" pid="1" pindex="{pindex}">')
                _stream_mesh_xml(handle, p.positions, p.faces)
                handle.write("</object>")
            handle.write(f'<object id="{assembly_id}" type="model" name="{_xml_attr(object_name)}"><components>')
            for p in ordered:
                handle.write(f'<component objectid="{mesh_ids[p.index]}"/>')
            handle.write("</components></object></resources>")
            handle.write(f'<build><item objectid="{assembly_id}"/></build></model>')
        if include_bambu_config:
            lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<config>", f'  <object id="{assembly_id}">', f'    <metadata key="name" value="{_xml_attr(object_name)}"/>', '    <metadata key="extruder" value="1"/>']
            for p in ordered:
                lines += [f'    <part id="{mesh_ids[p.index]}" subtype="normal_part">', f'      <metadata key="name" value="{_xml_attr(p.name)}"/>', f'      <metadata key="extruder" value="{int(part_slot.get(p.index, 1))}"/>', "    </part>"]
            lines += ["  </object>", "  <plate>", '    <metadata key="plater_id" value="1"/>', "    <model_instance>", f'      <metadata key="object_id" value="{assembly_id}"/>', '      <metadata key="instance_id" value="0"/>', "    </model_instance>", "  </plate>", "</config>"]
            archive.writestr("Metadata/model_settings.config", "\n".join(lines))
    return path


def write_parts_colorgroup_3mf(output_path: str | Path, parts: list[ScenePart], part_slot: dict[int, int], slots: list[dict[str, Any]], *, object_name: str) -> Path:
    """Fallback single-mesh 3MF with per-triangle colors (the texture route's format)."""
    from .export_3mf import write_colorgroup_3mf

    positions, faces, ids = _concatenate(parts)
    palette = np.array([s["rgb"] for s in slots], dtype=np.uint8)
    face_palette = np.array([max(0, int(part_slot.get(int(i), 1)) - 1) for i in ids], dtype=np.int32)
    return write_colorgroup_3mf(output_path, positions.astype(np.float32), faces, palette, face_palette, object_name=object_name)


def read_parts_3mf_structure(threemf_path: str | Path) -> dict[str, Any]:
    """Cheap structural readback used as the export gate (streams the XML, no DOM)."""
    import xml.etree.ElementTree as ET

    counts = {"mesh_objects": 0, "component_objects": 0, "components": 0, "build_items": 0, "triangles": 0, "vertices": 0, "unit": None, "config_parts": 0, "config_extruders": [], "base_materials": 0}
    with ZipFile(threemf_path) as archive:
        names = archive.namelist()
        with archive.open("3D/3dmodel.model") as handle:
            in_components = False
            for event, elem in ET.iterparse(handle, events=("start", "end")):
                tag = elem.tag.split("}")[-1]
                if event == "start":
                    if tag == "model":
                        counts["unit"] = elem.get("unit")
                    elif tag == "components":
                        in_components = True
                        counts["component_objects"] += 1
                    elif tag == "component":
                        counts["components"] += 1
                    elif tag == "mesh":
                        counts["mesh_objects"] += 1
                    elif tag == "item":
                        counts["build_items"] += 1
                    elif tag == "base":
                        counts["base_materials"] += 1
                else:
                    if tag == "vertex":
                        counts["vertices"] += 1
                    elif tag == "triangle":
                        counts["triangles"] += 1
                    elif tag == "components":
                        in_components = False
                    if tag in {"vertex", "triangle"}:
                        elem.clear()
        if "Metadata/model_settings.config" in names:
            root = ET.fromstring(archive.read("Metadata/model_settings.config"))
            for part in root.iter("part"):
                counts["config_parts"] += 1
                for meta in part.findall("metadata"):
                    if meta.get("key") == "extruder":
                        counts["config_extruders"].append(int(meta.get("value") or 0))
    counts["entries"] = names
    return counts


# --------------------------------------------------------------------------- orchestration


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=_json_default))
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def prepare_parts_labeling_inputs(glb_path: str | Path, *, out_dir: str | Path, concept_image_path: str | Path | None = None, max_colors: int = 12) -> dict[str, Any]:
    """Phase 1 (seconds, no repair): everything a labeler needs to name the parts.

    Writes ``parts_summary.json`` (facts + descriptions), ``parts_contact_sheet.png``,
    ``parts_tiles/part_<n>.png`` and, when a concept image is given, ``concept_palette.json``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parts = load_scene_parts(glb_path)
    facts = part_facts(parts)
    sheet = write_parts_contact_sheet(out / "parts_contact_sheet.png", parts, tiles_dir=out / "parts_tiles")
    palette = extract_concept_palette(concept_image_path, max_colors=max_colors) if concept_image_path else None
    summary = {
        "schema_version": PARTS_BUNDLE_SCHEMA_VERSION,
        "source_path": str(glb_path),
        "concept_image_path": str(concept_image_path) if concept_image_path else None,
        "part_count": len(parts),
        "parts": facts,
        "facts_text": "\n".join(f["description"] for f in facts),
        "contact_sheet": sheet,
        "labels_vocabulary": list(PART_LABELS),
    }
    _write_json(out / "parts_summary.json", summary)
    if palette is not None:
        _write_json(out / "concept_palette.json", palette)
    return {"parts_summary_path": str(out / "parts_summary.json"), "contact_sheet_path": sheet["path"], "tiles": sheet["tiles"], "concept_palette_path": str(out / "concept_palette.json") if palette else None, "summary": summary, "palette": palette}


def _gate(gate_id: str, *, passed: bool, summary: str, required: bool = True, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"id": gate_id, "passed": bool(passed), "required": bool(required), "summary": summary, "details": details or {}}


def build_parts_bundle(
    glb_path: str | Path,
    *,
    out_dir: str | Path,
    label_map: dict[str, Any],
    object_name: str,
    concept_image_path: str | Path | None = None,
    target_height_mm: float = DEFAULT_TARGET_HEIGHT_MM,
    max_slots: int = DEFAULT_MAX_SLOTS,
    ams_slots: int = DEFAULT_AMS_SLOTS,
    merge_delta_e: float = 12.0,
    recess_mm: float = DEFAULT_RECESS_MM,
    shell_mm: float = DEFAULT_SHELL_MM,
    core: bool = True,
    render: bool = True,
) -> dict[str, Any]:
    """Phase 2: label map + model → repaired, scaled, colored multi-part 3MF with a handoff manifest.

    Never raises for a bad model: every problem becomes a failed gate the operator can read.
    ``ready_for_duckagent_handoff`` is true only when all required gates pass.
    """
    started = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gates: list[dict[str, Any]] = []
    notes: list[str] = []
    parts = load_scene_parts(glb_path)
    facts = part_facts(parts)
    gates.append(_gate("parts_loaded", passed=len(parts) >= 2, summary=f"{len(parts)} parts loaded from the model", details={"part_count": len(parts)}))

    shape_issues = validate_label_map_shape(label_map, len(parts))
    label_status = str(label_map.get("status") or ("ok" if not shape_issues else "invalid"))
    label_source = str(label_map.get("source") or "unknown")
    label_issues = list(label_map.get("issues") or []) + shape_issues
    labels_ok = not shape_issues and label_status == "ok" and label_source in {"vision", "override", "vision+override"}
    gates.append(_gate("labels_validated", passed=labels_ok, summary=(f"labels from {label_source}: {label_status}" + (f" ({len(label_issues)} issues)" if label_issues else "")), details={"source": label_source, "status": label_status, "issues": label_issues}))
    if shape_issues:
        manifest = _finish_manifest(out, object_name=object_name, glb_path=glb_path, concept_image_path=concept_image_path, gates=gates, summary={"part_count": len(parts)}, artifacts={}, notes=notes + shape_issues, started=started, label_map=label_map, materials=None, assembly=None, parts_rows=[])
        return manifest

    palette_rows = extract_concept_palette(concept_image_path)["clusters"] if concept_image_path else []
    materials = build_materials(label_map, palette_rows, merge_delta_e=merge_delta_e, max_slots=max_slots)
    slots = materials["slots"]
    part_slot = {int(k): int(v) for k, v in materials["part_slot"].items()}
    gates.append(_gate("filament_slots_within_ams", passed=len(slots) <= ams_slots, required=False, summary=f"{len(slots)} filament slots (AMS holds {ams_slots})", details={"slots": len(slots), "ams_slots": ams_slots, "merged_to_fit": materials["merged_to_fit"]}))
    if materials["merged_to_fit"]:
        notes.append("colors merged to fit the slot limit: " + "; ".join(materials["merged_to_fit"]))

    scaled, scale_info = assemble_parts(parts, target_height_mm=target_height_mm)
    repaired = repair_parts(scaled, shell_thickness=shell_mm, recess_depth=recess_mm)
    failed = [p for p in repaired if not p.watertight]
    gates.append(_gate("parts_repaired_watertight", passed=not failed, summary=(f"all {len(repaired)} parts closed into watertight solids" if not failed else f"{len(failed)} part(s) could not be closed: {[p.index for p in failed]}"), details={"shell_mm": shell_mm, "recess_mm": recess_mm, "repairs": {str(p.index): {k: v for k, v in p.repair.items() if k != "attempts"} for p in repaired}, "attempts": {str(p.index): p.repair.get("attempts") for p in repaired if p.repair.get("method") in {"failed", "pymeshfix_repair"}}, "failed": [p.index for p in failed]}))
    core_part, core_info = (build_core(scaled, erode=DEFAULT_CORE_ERODE_MM) if core else (None, {"status": "disabled"}))
    gates.append(_gate("core_filled", passed=core_part is not None, required=False, summary=(f"solid core built {core_info.get('erode_mm', 0):.1f} mm inside the surface ({core_info.get('faces')} faces)" if core_part else f"no core: {core_info.get('status')}"), details=core_info))

    placed, assembly = place_parts(repaired + ([core_part] if core_part else []), target_height_mm=target_height_mm)
    shells = [p for p in placed if p.name != "core"]
    core_placed = next((p for p in placed if p.name == "core"), None)
    assembly["scale_mm_per_unit"] = scale_info["scale_mm_per_unit"]
    assembly["source_up_axis"] = scale_info["source_up_axis"]
    gates.append(_gate("assembly_scaled", passed=abs(assembly["height_mm"] - target_height_mm) < 0.5 and abs(assembly["bounds_mm"][0][2]) < 0.01, summary=f"assembled at {assembly['height_mm']:.1f} mm tall, standing on z=0", details=assembly))
    thin = [{"index": p.index, "name": p.name, "min_extent_mm": float(p.extents.min())} for p in shells if float(p.extents.min()) < MIN_PRINTABLE_EXTENT_MM]
    gates.append(_gate("thin_parts_printable", passed=not thin, required=False, summary=("every part is at least %.1f mm in its thinnest direction" % MIN_PRINTABLE_EXTENT_MM if not thin else f"{len(thin)} part(s) thinner than {MIN_PRINTABLE_EXTENT_MM} mm"), details={"thin_parts": thin}))

    for p in shells:
        row = next((r for r in label_map["parts"] if int(r["n"]) == p.index), {})
        p.name = f"{p.index}_{row.get('label', 'part')}"
    if core_placed is not None:
        body_slot = slots[0]["slot"]
        part_slot[core_placed.index] = body_slot
        core_placed.name = f"{core_placed.index}_core"
    threemf = write_parts_3mf(out / "parts_bundle.3mf", placed, part_slot, slots, object_name=object_name)
    colorgroup = write_parts_colorgroup_3mf(out / "parts_bundle_colorgroup.3mf", placed, part_slot, slots, object_name=object_name)
    structure = read_parts_3mf_structure(threemf)
    structure_ok = structure["mesh_objects"] == len(placed) and structure["components"] == len(placed) and structure["build_items"] == 1 and structure["config_parts"] == len(placed) and structure["unit"] == "millimeter" and all(1 <= e <= len(slots) for e in structure["config_extruders"])
    gates.append(_gate("threemf_structure_valid", passed=structure_ok, summary=f"3MF: {structure['mesh_objects']} part meshes in one assembly ({len(shells)} shells{' + core' if core_placed else ''}), {len(slots)} slots", details={k: v for k, v in structure.items() if k != "entries"}))

    part_rgb = {p.index: slots[part_slot[p.index] - 1]["rgb"] for p in shells}
    view_paths = render_assembly_views(out / "views", shells, part_rgb) if render else {}
    parts_rows = [{**r, "slot": part_slot.get(int(r["n"]))} for r in sorted(label_map["parts"], key=lambda r: int(r["n"]))]
    status_preview = "ready" if all(g["passed"] for g in gates if g["required"]) else "review_required"
    qa_board = write_parts_qa_board(out / "handoff_qa_board.png", concept_image_path=concept_image_path, view_paths=view_paths, slots=slots, parts_rows=parts_rows, status=status_preview, notes=notes + [g["summary"] for g in gates if not g["passed"]]) if render else None
    artifacts = {
        "bambu_3mf_path": str(threemf),
        "bambu_colorgroup_3mf_path": str(colorgroup),
        "qa_board_path": str(qa_board) if qa_board else None,
        "front_view_path": view_paths.get("front_three_quarter"),
        "side_view_path": view_paths.get("side"),
        "back_view_path": view_paths.get("back_three_quarter"),
        "grouped_obj_path": None,
        "grouped_mtl_path": None,
    }
    summary = {
        "route": "parts_bundle",
        "part_count": len(shells),
        "palette_size": len(slots),
        "component_count": len(placed),
        "core": core_info.get("status"),
        "shell_mm": shell_mm,
        "tiny_island_count": len(thin),
        "height_mm": assembly["height_mm"],
        "scale_mm_per_unit": assembly["scale_mm_per_unit"],
        "footprint_mm": assembly["footprint_mm"],
        "filament_slots": slots,
        "label_source": label_source,
        "label_status": label_status,
        "faces": int(sum(len(p.faces) for p in placed)),
        "bottom_flatness_status": None,
        "bambu_validation_status": None,
    }
    return _finish_manifest(out, object_name=object_name, glb_path=glb_path, concept_image_path=concept_image_path, gates=gates, summary=summary, artifacts=artifacts, notes=notes, started=started, label_map=label_map, materials=materials, assembly=assembly, parts_rows=parts_rows)


def _finish_manifest(out: Path, *, object_name: str, glb_path: str | Path, concept_image_path: str | Path | None, gates: list[dict[str, Any]], summary: dict[str, Any], artifacts: dict[str, Any], notes: list[str], started: float, label_map: dict[str, Any], materials: dict[str, Any] | None, assembly: dict[str, Any] | None, parts_rows: list[dict[str, Any]]) -> dict[str, Any]:
    required_missing = [k for k in ("bambu_3mf_path", "qa_board_path") if not artifacts.get(k) or not Path(str(artifacts.get(k))).exists()]
    gates.append(_gate("required_artifacts_exist", passed=not required_missing, summary=("3MF and QA board written" if not required_missing else f"missing: {required_missing}"), details={"missing": required_missing}))
    ready = all(g["passed"] for g in gates if g["required"])
    manifest = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "route": "parts_bundle",
        "route_schema_version": PARTS_BUNDLE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready" if ready else "review_required",
        "ready_for_duckagent_handoff": ready,
        "source": {"path": str(glb_path), "concept_image_path": str(concept_image_path) if concept_image_path else None, "object_name": object_name},
        "summary": summary,
        "gates": gates,
        "artifacts": artifacts,
        "label_map": label_map,
        "materials": {k: v for k, v in (materials or {}).items() if k != "part_slot"} | ({"part_slot": {str(k): v for k, v in materials["part_slot"].items()}} if materials else {}),
        "assembly": assembly,
        "parts": parts_rows,
        "notes": notes,
        "seconds": round(time.time() - started, 1),
        "operator_next_action": ("Open parts_bundle.3mf in Bambu Studio: one object with the parts pre-assigned to filament slots; check the QA board first." if ready else "Review the failed gates on the QA board; fix the label map (parts_labels.json) or the model and rebuild."),
    }
    manifest_path = _write_json(out / "handoff_manifest.json", manifest)
    manifest["artifacts"]["handoff_manifest_path"] = str(manifest_path)
    markdown = _write_manifest_markdown(out / "handoff_summary.md", manifest)
    manifest["artifacts"]["handoff_markdown_path"] = str(markdown)
    _write_json(manifest_path, manifest)
    return manifest


def _write_manifest_markdown(path: Path, manifest: dict[str, Any]) -> Path:
    summary = manifest.get("summary") or {}
    lines = [f"# Parts bundle: {manifest['source'].get('object_name')}", "", f"Status: **{manifest['status']}** (ready for handoff: {manifest['ready_for_duckagent_handoff']})", ""]
    if summary:
        lines += [f"- Parts: {summary.get('part_count')} | filament slots: {summary.get('palette_size')} | height: {summary.get('height_mm', 0):.1f} mm | faces: {summary.get('faces')}", ""]
    for s in summary.get("filament_slots") or []:
        lines.append(f"- slot {s['slot']}: {s['name']} {s['hex']} → parts {s['parts']} ({', '.join(s['labels'])})")
    lines += ["", "## Gates", ""]
    for g in manifest["gates"]:
        lines.append(f"- {'PASS' if g['passed'] else 'FAIL'}{'' if g['required'] else ' (advisory)'} `{g['id']}`: {g['summary']}")
    if manifest.get("notes"):
        lines += ["", "## Notes", ""] + [f"- {n}" for n in manifest["notes"]]
    lines += ["", f"Next: {manifest['operator_next_action']}", ""]
    path.write_text("\n".join(lines))
    return path
