from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .benchmark import _write_texture_source_preview
from .color_adjustments import posterize
from .export_obj_vertex_colors import write_obj_with_per_vertex_colors
from .model_io import LoadedTexturedMesh, load_textured_model
from .pipeline import convert_loaded_mesh_to_color_assets, convert_repaired_color_transfer_to_assets
from .repair_then_bake import _maybe_simplify_mesh, _mesh_stats, _repair_mesh
from .validation import validate_bambu_material_bundle, write_source_export_comparison


@dataclass(frozen=True)
class ProductionCandidate:
    label: str
    n_regions: int
    posterize_levels: int | None = None


DEFAULT_PRODUCTION_CANDIDATES: tuple[ProductionCandidate, ...] = (
    ProductionCandidate("baseline_r16", n_regions=16, posterize_levels=None),
    ProductionCandidate("posterize4_r16", n_regions=16, posterize_levels=4),
    ProductionCandidate("posterize4_r8", n_regions=8, posterize_levels=4),
    ProductionCandidate("baseline_r5", n_regions=5, posterize_levels=None),
)


def _posterize_texture(texture_rgb: np.ndarray, levels: int) -> np.ndarray:
    rgb = np.asarray(texture_rgb, dtype=np.float32) / 255.0
    adjusted = posterize(rgb, int(levels))
    return np.clip(np.rint(adjusted * 255.0), 0, 255).astype(np.uint8)


def _with_texture(loaded: LoadedTexturedMesh, texture_rgb: np.ndarray) -> LoadedTexturedMesh:
    return LoadedTexturedMesh(
        mesh=loaded.mesh,
        positions=np.asarray(loaded.positions, dtype=np.float32),
        faces=np.asarray(loaded.faces, dtype=np.int64),
        texcoords=np.asarray(loaded.texcoords, dtype=np.float32),
        texture_rgb=np.asarray(texture_rgb, dtype=np.uint8),
        source_path=loaded.source_path,
        texture_path=loaded.texture_path,
        source_format=loaded.source_format,
        normal_texture_rgb=None if loaded.normal_texture_rgb is None else np.asarray(loaded.normal_texture_rgb, dtype=np.uint8),
        orm_texture_rgb=None if loaded.orm_texture_rgb is None else np.asarray(loaded.orm_texture_rgb, dtype=np.uint8),
        base_color_factor=None if loaded.base_color_factor is None else np.asarray(loaded.base_color_factor, dtype=np.float32),
        metallic_factor=float(loaded.metallic_factor),
        roughness_factor=float(loaded.roughness_factor),
        normal_scale=float(loaded.normal_scale),
    )


def _replace_prefixes(value: Any, old_prefix: str, new_prefix: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_prefixes(item, old_prefix, new_prefix) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_prefixes(item, old_prefix, new_prefix) for item in value]
    if isinstance(value, str) and value.startswith(old_prefix):
        return new_prefix + value[len(old_prefix) :]
    return value


def _promote_candidate(candidate_dir: Path, selected_dir: Path) -> None:
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    shutil.copytree(candidate_dir, selected_dir)
    report_path = selected_dir / "conversion_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rewritten = _replace_prefixes(report, str(candidate_dir), str(selected_dir))
        report_path.write_text(json.dumps(rewritten, indent=2), encoding="utf-8")


def run_production_conversion(
    source_path: str | Path,
    *,
    texture_path: str | Path | None = None,
    out_dir: str | Path,
    object_name: str | None = None,
    quality_threshold: float = 0.02,
    fail_closed: bool = True,
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_textured_model(source_path, texture_path=texture_path)
    if loaded.source_format not in {"obj", "objzip", "glb"}:
        raise ValueError(f"Unsupported source format for production conversion: {loaded.source_format}")

    candidate_specs = [
        ProductionCandidate(**spec) for spec in candidates
    ] if candidates else list(DEFAULT_PRODUCTION_CANDIDATES)

    candidate_rows: list[dict[str, Any]] = []
    candidates_dir = output_dir / "_candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    for candidate in candidate_specs:
        candidate_dir = candidates_dir / candidate.label
        candidate_dir.mkdir(parents=True, exist_ok=True)

        candidate_loaded = loaded
        transformed_texture_path = None
        if candidate.posterize_levels is not None:
            transformed_texture = _posterize_texture(loaded.texture_rgb, candidate.posterize_levels)
            transformed_texture_path = candidate_dir / "transformed_texture.png"
            from PIL import Image

            Image.fromarray(transformed_texture, mode="RGB").save(transformed_texture_path)
            candidate_loaded = _with_texture(loaded, transformed_texture)

        source_preview_path = candidate_dir / "source_preview.png"
        _write_texture_source_preview(
            source_preview_path,
            positions=candidate_loaded.positions,
            faces=candidate_loaded.faces,
            texcoords=candidate_loaded.texcoords,
            texture_rgb=candidate_loaded.texture_rgb,
        )

        report = convert_loaded_mesh_to_color_assets(
            candidate_loaded,
            out_dir=candidate_dir,
            n_regions=int(candidate.n_regions),
            strategy="legacy_fast_face_labels",
            object_name=object_name,
        )
        comparison = write_source_export_comparison(
            source_preview_path=source_preview_path,
            export_preview_path=Path(report["preview_path"]),
            comparison_path=candidate_dir / "source_export_comparison.png",
            source_mode="same_mesh_production",
            simplify_applied=False,
            color_transfer_applied=False,
        )

        row = {
            "label": candidate.label,
            "strategy": "legacy_fast_face_labels",
            "n_regions": int(candidate.n_regions),
            "posterize_levels": candidate.posterize_levels,
            "source_preview_path": str(source_preview_path),
            "transformed_texture_path": str(transformed_texture_path) if transformed_texture_path else None,
            "report_path": str(candidate_dir / "conversion_report.json"),
            "export_preview_path": str(report["preview_path"]),
            "comparison_path": str(candidate_dir / "source_export_comparison.png"),
            "assessment": comparison["assessment"],
            "assessment_label": comparison["assessment_label"],
            "mean_pixel_drift": float(comparison["mean_pixel_drift"]),
        }
        candidate_rows.append(row)

    candidate_rows.sort(key=lambda item: (float(item["mean_pixel_drift"]), int(item["n_regions"])))
    best_candidate = candidate_rows[0]
    selected_dir = output_dir / "selected"
    best_candidate_dir = candidates_dir / str(best_candidate["label"])
    _promote_candidate(best_candidate_dir, selected_dir)

    ready_for_production = float(best_candidate["mean_pixel_drift"]) <= float(quality_threshold)
    status = "ready" if ready_for_production else "rejected"
    production_report = {
        "status": status,
        "ready_for_production": bool(ready_for_production),
        "quality_threshold": float(quality_threshold),
        "fail_closed": bool(fail_closed),
        "scope": {
            "same_mesh_only": True,
            "supported_source_formats": ["obj", "objzip", "glb"],
            "unsupported": [
                "automatic repaired-geometry transfer",
                "cross-topology color transfer",
            ],
        },
        "selected_candidate": {
            **best_candidate,
            "selected_dir": str(selected_dir),
        },
        "candidates": candidate_rows,
    }
    report_path = output_dir / "production_report.json"
    report_path.write_text(json.dumps(production_report, indent=2), encoding="utf-8")

    if fail_closed and not ready_for_production:
        return {
            **production_report,
            "message": "Best same-mesh candidate did not meet the production quality threshold.",
            "production_report_path": str(report_path),
        }

    return {
        **production_report,
        "message": "Production conversion completed.",
        "production_report_path": str(report_path),
    }


def _smooth_repaired_geometry(
    positions: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    iteration_count = max(0, int(iterations))
    if iteration_count == 0:
        return np.asarray(positions, dtype=np.float32), {
            "smoothing_applied": False,
            "iterations": 0,
        }
    mesh = trimesh.Trimesh(
        vertices=np.asarray(positions, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    before = _mesh_stats(np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int64))
    trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=iteration_count)
    trimesh.repair.fix_normals(mesh, multibody=True)
    after = _mesh_stats(np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int64))
    return np.asarray(mesh.vertices, dtype=np.float32), {
        "smoothing_applied": True,
        "method": "taubin",
        "iterations": iteration_count,
        "before": before,
        "after": after,
    }


def _bottom_flatness_report(positions: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    pos = np.asarray(positions, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    if len(pos) == 0 or len(face_array) == 0:
        return {
            "status": "unknown",
            "ready": False,
            "reason": "empty geometry",
        }
    min_y = float(pos[:, 1].min())
    height = float(max(pos[:, 1].max() - min_y, 1e-9))
    face_vertices = pos[face_array]
    face_y = face_vertices[:, :, 1]
    normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    near_plane_tolerance = max(height * 0.00001, 0.001)
    support_band_tolerance = height * 0.002
    exact_bottom_vertices = pos[pos[:, 1] <= min_y + near_plane_tolerance]
    exact_bottom_faces = (face_y <= min_y + near_plane_tolerance).all(axis=1)
    support_band_faces = (face_y <= min_y + support_band_tolerance).all(axis=1)
    downward_support_faces = support_band_faces & (np.abs(normals[:, 1]) >= 0.75)
    x_footprint = float(np.ptp(exact_bottom_vertices[:, 0])) if len(exact_bottom_vertices) else 0.0
    z_footprint = float(np.ptp(exact_bottom_vertices[:, 2])) if len(exact_bottom_vertices) else 0.0
    ready = bool(len(exact_bottom_vertices) >= 12 and int(exact_bottom_faces.sum()) >= 8 and x_footprint > height * 0.08 and z_footprint > height * 0.05)
    return {
        "status": "flat_support_ready" if ready else "needs_review",
        "ready": ready,
        "min_y": round(min_y, 6),
        "model_height": round(height, 6),
        "near_plane_tolerance": round(float(near_plane_tolerance), 6),
        "support_band_tolerance": round(float(support_band_tolerance), 6),
        "exact_bottom_vertex_count": int(len(exact_bottom_vertices)),
        "exact_bottom_face_count": int(exact_bottom_faces.sum()),
        "support_band_face_count": int(support_band_faces.sum()),
        "downward_support_face_count": int(downward_support_faces.sum()),
        "exact_bottom_x_footprint": round(x_footprint, 6),
        "exact_bottom_z_footprint": round(z_footprint, 6),
    }


def _build_paint_intent_report(
    *,
    conversion_report: dict[str, Any],
    repair_stats: dict[str, Any],
    smoothing_metadata: dict[str, Any],
    repaired_positions: np.ndarray,
    repaired_faces: np.ndarray,
    transfer_assessment: dict[str, Any],
    bambu_validation: dict[str, Any],
) -> dict[str, Any]:
    duck_intent = conversion_report.get("duck_color_intent") or {}
    rewrites = duck_intent.get("component_rewrites") or []
    rewrite_reason_counts = Counter(str(item.get("reason") or "unknown") for item in rewrites)
    palette = conversion_report.get("palette") or []
    light_neutral_faces = 0
    warm_faces = 0
    for row in palette:
        rgb = row.get("rgb") if isinstance(row, dict) else None
        face_count = int(row.get("face_count") or 0) if isinstance(row, dict) else 0
        if not isinstance(rgb, list) or len(rgb) < 3:
            continue
        color = np.asarray(rgb[:3], dtype=np.uint8)
        signals = {
            "neutral": max(0.0, 1.0 - (float(np.max(color) - np.min(color)) / 255.0) * 4.0),
            "luminance": float(0.2126 * color[0] / 255.0 + 0.7152 * color[1] / 255.0 + 0.0722 * color[2] / 255.0),
        }
        if signals["neutral"] >= 0.30 and signals["luminance"] >= 0.62:
            light_neutral_faces += face_count
        if signals["luminance"] < 0.68 and color[0] > color[2] and color[1] > color[2] * 0.8:
            warm_faces += face_count
    bottom_flatness = _bottom_flatness_report(repaired_positions, repaired_faces)
    ready = bool(
        transfer_assessment.get("ready_for_auto") is True
        and bambu_validation.get("ready_for_bambu") is True
        and bottom_flatness.get("ready") is True
    )
    risks: list[str] = []
    if int(conversion_report.get("tiny_island_count") or 0) > 96:
        risks.append("tiny island count is above the current auto threshold")
    if not bottom_flatness.get("ready"):
        risks.append("flat-bottom support plane needs review")
    if not rewrites and duck_intent:
        risks.append("duck color-intent policy found no cleanup opportunities")
    return {
        "status": "print_ready" if ready and not risks else "review_recommended",
        "ready_for_print_review": ready,
        "summary": {
            "palette_size": int(conversion_report.get("palette_size") or 0),
            "component_count": int(conversion_report.get("component_count") or 0),
            "tiny_island_count": int(conversion_report.get("tiny_island_count") or 0),
            "largest_component_share": conversion_report.get("largest_component_share"),
            "light_neutral_detail_faces": int(light_neutral_faces),
            "warm_beak_or_detail_faces": int(warm_faces),
            "semantic_reassigned_faces": int(duck_intent.get("reassigned_faces") or 0),
            "semantic_rewrite_reason_counts": dict(rewrite_reason_counts),
        },
        "geometry": {
            "repair_stats": repair_stats,
            "smoothing": smoothing_metadata,
            "bottom_flatness": bottom_flatness,
        },
        "color_intent": duck_intent,
        "transfer_assessment": transfer_assessment,
        "bambu_material_validation": bambu_validation,
        "risks": risks,
        "artifacts": {
            "preview_path": conversion_report.get("preview_path"),
            "obj_path": conversion_report.get("obj_path"),
            "mtl_path": conversion_report.get("mtl_path"),
            "threemf_path": conversion_report.get("threemf_path"),
            "palette_swatch_path": conversion_report.get("palette_swatch_path"),
        },
    }


def _write_paint_intent_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report.get("summary") or {}
    bottom = ((report.get("geometry") or {}).get("bottom_flatness") or {})
    artifacts = report.get("artifacts") or {}
    lines = [
        "# Paint Intent Report",
        "",
        f"- Status: {report.get('status')}",
        f"- Palette size: {summary.get('palette_size')}",
        f"- Component count: {summary.get('component_count')}",
        f"- Tiny islands: {summary.get('tiny_island_count')}",
        f"- Semantic reassigned faces: {summary.get('semantic_reassigned_faces')}",
        f"- Light neutral detail faces: {summary.get('light_neutral_detail_faces')}",
        f"- Bottom support: {bottom.get('status')} ({bottom.get('exact_bottom_face_count')} exact bottom faces)",
        f"- 3MF: {artifacts.get('threemf_path')}",
        f"- Preview: {artifacts.get('preview_path')}",
    ]
    risks = report.get("risks") or []
    if risks:
        lines.extend(["", "## Review Risks", ""])
        lines.extend(f"- {risk}" for risk in risks)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_repaired_production_conversion(
    source_path: str | Path,
    *,
    texture_path: str | Path | None = None,
    out_dir: str | Path,
    object_name: str | None = None,
    repair_backend: str = "voxel_marching_cubes",
    target_face_count: int | None = 250_000,
    max_colors: int = 8,
    transfer_strategy: str = "geometry_transfer_blender_like_bake_duck_intent",
    repair_smoothing_iterations: int | None = None,
    repair_voxel_divisions: int = 128,
    fail_closed: bool = True,
) -> dict[str, Any]:
    """Run the repaired-geometry production lane end to end.

    The lane is intentionally separate from same-mesh production: it repairs the
    source geometry first, then transfers larger printable paint regions from
    the original textured source onto that repaired shell.
    """
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_textured_model(source_path, texture_path=texture_path)
    repair_dir = output_dir / "_repair_geometry"
    selected_dir = output_dir / "selected"
    repair_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    repaired_positions, repaired_faces, repair_metadata = _repair_mesh(
        loaded.positions,
        loaded.faces,
        backend=repair_backend,
        voxel_divisions=repair_voxel_divisions,
    )
    repaired_positions, repaired_faces, simplify_metadata = _maybe_simplify_mesh(
        repaired_positions,
        repaired_faces,
        target_face_count=target_face_count,
    )
    smoothing_iterations = (
        12
        if repair_smoothing_iterations is None and repair_backend == "voxel_marching_cubes"
        else int(repair_smoothing_iterations or 0)
    )
    repaired_positions, smoothing_metadata = _smooth_repaired_geometry(
        repaired_positions,
        repaired_faces,
        iterations=smoothing_iterations,
    )
    repair_voxel_divisions_value = (
        int(repair_metadata.get("voxel_divisions", repair_voxel_divisions))
        if repair_backend == "voxel_marching_cubes"
        else None
    )
    repair_stats = _mesh_stats(repaired_positions, repaired_faces)
    neutral_vertex_colors = np.ones((len(repaired_positions), 3), dtype=np.float32)
    target_geometry_path = write_obj_with_per_vertex_colors(
        repair_dir / "repaired_geometry.obj",
        repaired_positions,
        repaired_faces,
        neutral_vertex_colors,
        object_name="RepairedProductionGeometry",
    )

    conversion_report = convert_repaired_color_transfer_to_assets(
        source_path,
        target_geometry_path,
        color_source_texture_path=texture_path,
        out_dir=selected_dir,
        max_colors=max_colors,
        target_face_count=None,
        strategy=transfer_strategy,
        object_name=object_name,
    )
    conversion_report.update(
        {
            "production_lane": "repaired_geometry_region_transfer",
            "repair_backend": repair_backend,
            "repair_metadata": repair_metadata,
            "repair_simplification": simplify_metadata,
            "repair_smoothing": smoothing_metadata,
            "repair_voxel_divisions": repair_voxel_divisions_value,
            "repair_stats": repair_stats,
            "intermediate_repaired_geometry_path": str(target_geometry_path),
        }
    )

    bambu_validation = validate_bambu_material_bundle(conversion_report)
    conversion_report["bambu_material_validation"] = bambu_validation

    transfer_assessment = conversion_report.get("repaired_transfer_assessment") or {}
    ready_for_production = (
        transfer_assessment.get("ready_for_auto") is True
        and bambu_validation.get("ready_for_bambu") is True
    )
    paint_intent_report = _build_paint_intent_report(
        conversion_report=conversion_report,
        repair_stats=repair_stats,
        smoothing_metadata=smoothing_metadata,
        repaired_positions=repaired_positions,
        repaired_faces=repaired_faces,
        transfer_assessment=transfer_assessment,
        bambu_validation=bambu_validation,
    )
    paint_intent_report_path = selected_dir / "paint_intent_report.json"
    paint_intent_markdown_path = selected_dir / "paint_intent_report.md"
    paint_intent_report_path.write_text(json.dumps(paint_intent_report, indent=2), encoding="utf-8")
    _write_paint_intent_markdown(paint_intent_markdown_path, paint_intent_report)
    conversion_report["paint_intent_report_path"] = str(paint_intent_report_path)
    conversion_report["paint_intent_markdown_path"] = str(paint_intent_markdown_path)
    conversion_report_path = Path(str(conversion_report["report_path"]))
    conversion_report_path.write_text(json.dumps(conversion_report, indent=2), encoding="utf-8")
    status = "ready" if ready_for_production else "rejected"
    acceptance_summary = {
        "status": status,
        "ready_for_production": bool(ready_for_production),
        "fail_closed": bool(fail_closed),
        "source_path": str(Path(source_path).expanduser().resolve()),
        "selected_dir": str(selected_dir),
        "object_name": object_name,
        "repair_backend": repair_backend,
        "target_face_count": None if target_face_count is None else int(target_face_count),
        "max_colors": int(max_colors),
        "transfer_strategy": transfer_strategy,
        "repair_smoothing_iterations": int(smoothing_iterations),
        "repair_voxel_divisions": repair_voxel_divisions_value,
        "intermediate_repaired_geometry_path": str(target_geometry_path),
        "conversion_report_path": str(conversion_report_path),
        "obj_path": conversion_report["obj_path"],
        "mtl_path": conversion_report["mtl_path"],
        "threemf_path": conversion_report["threemf_path"],
        "preview_path": conversion_report["preview_path"],
        "palette_size": conversion_report["palette_size"],
        "component_count": conversion_report["component_count"],
        "tiny_island_count": conversion_report["tiny_island_count"],
        "largest_component_share": conversion_report["largest_component_share"],
        "repair_stats": repair_stats,
        "repair_smoothing": smoothing_metadata,
        "target_geometry_stats": conversion_report.get("target_geometry_stats"),
        "paint_intent_report_path": str(paint_intent_report_path),
        "paint_intent_markdown_path": str(paint_intent_markdown_path),
        "paint_intent_report": paint_intent_report,
        "transfer_assessment": transfer_assessment,
        "bambu_material_validation": bambu_validation,
    }
    acceptance_path = selected_dir / "acceptance_summary.json"
    acceptance_path.write_text(json.dumps(acceptance_summary, indent=2), encoding="utf-8")

    production_report = {
        **acceptance_summary,
        "production_report_path": str(output_dir / "production_report.json"),
        "acceptance_summary_path": str(acceptance_path),
        "message": (
            "Repaired production conversion completed."
            if ready_for_production
            else "Repaired production conversion did not pass all production gates."
        ),
    }
    Path(production_report["production_report_path"]).write_text(json.dumps(production_report, indent=2), encoding="utf-8")

    if fail_closed and not ready_for_production:
        return production_report
    return production_report
