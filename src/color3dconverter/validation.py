from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile, is_zipfile
import xml.etree.ElementTree as ET


THREEMF_CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
THREEMF_MATERIAL_NS = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_grouped_obj_geometry_stats(obj_path: str | Path) -> dict[str, Any]:
    import numpy as np
    import trimesh

    path = Path(obj_path).expanduser().resolve()
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            parts = line.split()
            if len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("f "):
            face: list[int] = []
            for token in line.split()[1:]:
                face.append(int(token.split("/")[0]) - 1)
            if len(face) == 3:
                faces.append(face)

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    return {
        "path": str(path),
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "body_count": int(len(mesh.split(only_watertight=False))),
    }


def _count_mtl_materials(mtl_path: str | Path) -> int:
    path = Path(mtl_path).expanduser().resolve()
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("newmtl "))


def _read_3mf_colorgroup_stats(threemf_path: str | Path) -> dict[str, Any]:
    path = Path(threemf_path).expanduser().resolve()
    if not is_zipfile(path):
        return {"path": str(path), "zip_ok": False}

    with ZipFile(path, "r") as archive:
        model_bytes = archive.read("3D/3dmodel.model")

    root = ET.fromstring(model_bytes)
    namespaces = {"core": THREEMF_CORE_NS, "m": THREEMF_MATERIAL_NS}
    vertices = root.findall(".//core:vertices/core:vertex", namespaces)
    triangles = root.findall(".//core:triangles/core:triangle", namespaces)
    colors = root.findall(".//m:colorgroup/m:color", namespaces)
    triangle_color_indexes: list[int] = []
    for triangle in triangles:
        for key in ("p1", "p2", "p3"):
            if key in triangle.attrib:
                triangle_color_indexes.append(int(triangle.attrib[key]))
    return {
        "path": str(path),
        "zip_ok": True,
        "vertex_count": len(vertices),
        "triangle_count": len(triangles),
        "color_count": len(colors),
        "min_color_index": min(triangle_color_indexes) if triangle_color_indexes else None,
        "max_color_index": max(triangle_color_indexes) if triangle_color_indexes else None,
        "triangle_color_index_count": len(triangle_color_indexes),
    }


def validate_bambu_material_bundle(
    report: dict[str, Any],
    *,
    require_watertight: bool = True,
    require_single_body: bool = True,
    max_tiny_islands: int = 96,
    max_components_per_palette_color: int = 48,
) -> dict[str, Any]:
    """Validate the actual OBJ/MTL/3MF files Bambu Studio will import.

    Trimesh can split a material-grouped OBJ by material on import, so this reads
    the OBJ vertices/faces directly before measuring topology.
    """
    reasons: list[str] = []
    palette_size = _safe_int(report.get("palette_size") or report.get("region_count"), 1)
    vertex_count = _safe_int(report.get("vertex_count"))
    face_count = _safe_int(report.get("face_count"))
    component_count = _safe_int(report.get("component_count"))
    tiny_island_count = _safe_int(report.get("tiny_island_count"))

    obj_stats = _read_grouped_obj_geometry_stats(report["obj_path"])
    if obj_stats["vertex_count"] != vertex_count:
        reasons.append(f"OBJ vertex count {obj_stats['vertex_count']:,} does not match report {vertex_count:,}")
    if obj_stats["face_count"] != face_count:
        reasons.append(f"OBJ face count {obj_stats['face_count']:,} does not match report {face_count:,}")
    if require_watertight and obj_stats["is_watertight"] is not True:
        reasons.append("OBJ geometry is not watertight when parsed independent of material groups")
    if require_single_body and int(obj_stats["body_count"]) != 1:
        reasons.append(f"OBJ geometry has {obj_stats['body_count']} bodies instead of 1")

    material_count = _count_mtl_materials(report["mtl_path"])
    if material_count != palette_size:
        reasons.append(f"MTL material count {material_count} does not match palette size {palette_size}")

    threemf_stats = _read_3mf_colorgroup_stats(report["threemf_path"])
    if not threemf_stats.get("zip_ok"):
        reasons.append("3MF is not a valid zip package")
    else:
        if threemf_stats.get("vertex_count") != vertex_count:
            reasons.append(f"3MF vertex count {threemf_stats.get('vertex_count'):,} does not match report {vertex_count:,}")
        if threemf_stats.get("triangle_count") != face_count:
            reasons.append(f"3MF triangle count {threemf_stats.get('triangle_count'):,} does not match report {face_count:,}")
        if threemf_stats.get("color_count") != palette_size:
            reasons.append(f"3MF color count {threemf_stats.get('color_count')} does not match palette size {palette_size}")
        max_color_index = threemf_stats.get("max_color_index")
        if max_color_index is not None and int(max_color_index) >= palette_size:
            reasons.append(f"3MF references color index {max_color_index}, outside palette size {palette_size}")

    max_components = max(palette_size * int(max_components_per_palette_color), palette_size + 8)
    if component_count > max_components:
        reasons.append(f"connected component count {component_count:,} exceeds threshold {max_components:,}")
    if tiny_island_count > int(max_tiny_islands):
        reasons.append(f"tiny island count {tiny_island_count:,} exceeds threshold {int(max_tiny_islands):,}")

    ready = not reasons
    return {
        "status": "ready_for_bambu" if ready else "needs_review",
        "ready_for_bambu": ready,
        "reasons": reasons,
        "obj_topology": obj_stats,
        "mtl_material_count": material_count,
        "threemf_colorgroup": threemf_stats,
        "thresholds": {
            "max_tiny_islands": int(max_tiny_islands),
            "max_components_per_palette_color": int(max_components_per_palette_color),
            "require_watertight": bool(require_watertight),
            "require_single_body": bool(require_single_body),
        },
    }


def write_source_export_comparison(
    *,
    source_preview_path: str | Path,
    export_preview_path: str | Path,
    comparison_path: str | Path,
    source_mode: str | None = None,
    simplify_applied: bool = False,
    color_transfer_applied: bool = False,
) -> dict[str, Any]:
    from PIL import Image, ImageChops, ImageDraw, ImageOps, ImageStat

    source_preview = Path(source_preview_path).expanduser().resolve()
    export_preview = Path(export_preview_path).expanduser().resolve()
    output_path = Path(comparison_path).expanduser().resolve()
    if not source_preview.exists() or not export_preview.exists():
        raise FileNotFoundError("Source preview and export preview must both exist for comparison.")

    source_image = Image.open(source_preview).convert("RGB")
    export_image = Image.open(export_preview).convert("RGB")
    compare_size = (400, 400)
    source_fit = ImageOps.fit(source_image, compare_size, method=Image.Resampling.BICUBIC)
    export_fit = ImageOps.fit(export_image, compare_size, method=Image.Resampling.BICUBIC)

    diff = ImageChops.difference(source_fit, export_fit)
    diff_stats = ImageStat.Stat(diff)
    mean_abs_diff = round(sum(diff_stats.mean) / (len(diff_stats.mean) * 255.0), 4)
    if mean_abs_diff <= 0.09:
        assessment = "close"
        assessment_label = "Close match"
    elif mean_abs_diff <= 0.18:
        assessment = "moderate_drift"
        assessment_label = "Moderate drift"
    else:
        assessment = "poor_match"
        assessment_label = "Poor match"

    notes: list[str] = []
    if source_mode == "single_image":
        notes.append("The faceted look can come directly from the single-image source mesh when that upstream asset is already low poly.")
    if simplify_applied:
        notes.append("Geometry simplification was applied before export, so some drift may come from shape cleanup as well as palette reduction.")
    if color_transfer_applied:
        notes.append("This export uses repaired geometry with transferred source color, so the comparison is testing the reusable transfer path rather than direct texture-region labeling.")
    if assessment != "close":
        notes.append("The export still needs cleaner printable regions before it will feel trustworthy in Bambu Studio.")

    canvas = Image.new("RGB", (860, 470), (245, 241, 234))
    draw = ImageDraw.Draw(canvas)
    canvas.paste(source_fit, (20, 48))
    canvas.paste(export_fit, (440, 48))
    draw.text((20, 16), "Source Preview", fill=(28, 28, 28))
    draw.text((440, 16), "Local Bambu Export Preview", fill=(28, 28, 28))
    draw.text((20, 452), f"Assessment: {assessment_label}", fill=(28, 28, 28))
    draw.text((220, 452), f"Mean pixel drift: {mean_abs_diff:.3f}", fill=(28, 28, 28))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    return {
        "assessment": assessment,
        "assessment_label": assessment_label,
        "mean_pixel_drift": mean_abs_diff,
        "comparison_path": str(output_path),
        "source_preview_path": str(source_preview),
        "export_preview_path": str(export_preview),
        "summary": f"{assessment_label} between the source preview and the current Bambu export preview.",
        "notes": notes,
    }


def write_bambu_validation_bundle(
    *,
    output_dir: str | Path,
    source_preview_path: str | Path | None,
    export_preview_path: str | Path,
    threemf_path: str | Path,
    obj_path: str | Path,
    probe_exports: list[dict[str, Any]] | None = None,
    source_mode: str | None = None,
    simplify_applied: bool = False,
    color_transfer_applied: bool = False,
    comparison_filename: str = "local_bambu_source_comparison.png",
    report_filename: str = "local_bambu_validation_report.json",
    markdown_filename: str = "local_bambu_validation_report.md",
) -> dict[str, Any] | None:
    output_root = Path(output_dir).expanduser().resolve()
    source_preview_text = str(source_preview_path or "").strip()
    if not source_preview_text:
        return None
    source_preview = Path(source_preview_text).expanduser().resolve()
    export_preview = Path(export_preview_path).expanduser().resolve()
    if not source_preview.exists() or not export_preview.exists():
        return None

    comparison = write_source_export_comparison(
        source_preview_path=source_preview,
        export_preview_path=export_preview,
        comparison_path=output_root / comparison_filename,
        source_mode=source_mode,
        simplify_applied=simplify_applied,
        color_transfer_applied=color_transfer_applied,
    )
    report = {
        "status": "ok",
        "comparison": comparison,
        "threemf_path": str(Path(threemf_path).expanduser().resolve()),
        "obj_path": str(Path(obj_path).expanduser().resolve()),
        "probe_exports": probe_exports or [],
        "operator_checklist": [
            "Open the local 3MF in Bambu Studio first.",
            "Compare Bambu's imported color blocks against the source/export comparison image.",
            "If the mapping looks wrong, test the 4-color and 8-color probe files before changing the converter.",
            "Use the OBJ only as a fallback geometry/material inspection path.",
        ],
    }
    report_path = output_root / report_filename
    markdown_path = output_root / markdown_filename
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown = [
        "# Bambu Validation",
        "",
        f"- Assessment: {comparison['assessment_label']}",
        f"- Mean pixel drift: {comparison['mean_pixel_drift']:.3f}",
        f"- 3MF: {report['threemf_path']}",
        f"- OBJ: {report['obj_path']}",
        "",
        "## Checklist",
    ]
    markdown.extend([f"- {item}" for item in report["operator_checklist"]])
    if probe_exports:
        markdown.extend(["", "## Probe Files"])
        markdown.extend([f"- {item.get('label') or item.get('name') or 'Probe'}: {item.get('path') or ''}" for item in probe_exports])
    if comparison.get("notes"):
        markdown.extend(["", "## Notes"])
        markdown.extend([f"- {item}" for item in comparison["notes"]])
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return {
        **comparison,
        "validation_report_path": str(report_path),
        "validation_markdown_path": str(markdown_path),
    }
