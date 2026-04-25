from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from pymeshfix import _meshfix
from scipy.spatial import cKDTree

from .benchmark import _write_face_color_preview, _write_texture_source_preview
from .export_obj_vertex_colors import write_obj_with_per_vertex_colors
from .face_regions import compute_face_normals, face_centroids
from .model_io import load_textured_model
from .pipeline import write_face_color_mesh_to_assets
from .provider_oracle import (
    ProviderOracleVariant,
    _align_source_to_target,
    _face_colors_from_vertex_colors,
    _load_target_vertex_color_obj,
    _normalize_points,
    _prepare_source_for_variant,
    _predict_variant_colors,
    _vertex_color_metrics,
)
from .surface_transfer import barycentric_weights, interpolate_triangle_colors
from .validation import write_source_export_comparison


DEFAULT_REPAIR_BACKENDS: tuple[str, ...] = (
    "trimesh_clean",
    "pymeshfix_core",
)
VOXEL_REPAIR_DIVISIONS = 96

DEFAULT_BAKE_VARIANT = ProviderOracleVariant(
    label="repair_then_bake_surface_uv",
    method="nearest_surface_uv",
    sampling_mode="bilinear",
    uv_flip_y=True,
    candidate_count=8,
    pad_pixels=4,
)
DEFAULT_PREVIEW_FACE_LIMIT = 60000
DEFAULT_ALIGNMENT_SAMPLE_SIZE = 2500
DEFAULT_TARGET_FACE_COUNT = 250000
DEFAULT_MAX_BAMBU_PIXEL_DRIFT = 0.06
DEFAULT_MAX_BAMBU_TINY_ISLANDS = 32
DEFAULT_MAX_BAMBU_COMPONENTS = 96


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mesh_stats(positions: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    mesh = trimesh.Trimesh(vertices=positions, faces=faces, process=False)
    face_count = int(len(np.asarray(faces)))
    return {
        "vertex_count": int(len(np.asarray(positions))),
        "face_count": face_count,
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "body_count": None if face_count > 200000 else int(len(mesh.split(only_watertight=False))),
    }


def _assess_bambu_print_ready(
    *,
    repair_stats: dict[str, Any],
    source_comparison: dict[str, Any],
    bambu_report: dict[str, Any],
    max_pixel_drift: float = DEFAULT_MAX_BAMBU_PIXEL_DRIFT,
    max_tiny_islands: int = DEFAULT_MAX_BAMBU_TINY_ISLANDS,
    max_components: int = DEFAULT_MAX_BAMBU_COMPONENTS,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not bool(repair_stats.get("is_watertight")):
        reasons.append("repaired geometry is not watertight")
    body_count = repair_stats.get("body_count")
    if body_count is None:
        reasons.append("repaired geometry body count was not measured")
    elif int(body_count) != 1:
        reasons.append(f"repaired geometry has {int(body_count)} bodies instead of 1")
    mean_pixel_drift = float(source_comparison.get("mean_pixel_drift") or 1e9)
    if mean_pixel_drift > float(max_pixel_drift):
        reasons.append(f"source/export drift {mean_pixel_drift:.4f} is above {float(max_pixel_drift):.4f}")
    tiny_islands = int(bambu_report.get("tiny_island_count") or 0)
    if tiny_islands > int(max_tiny_islands):
        reasons.append(f"printable region tiny island count {tiny_islands:,} is above {int(max_tiny_islands):,}")
    component_count = int(bambu_report.get("component_count") or 0)
    if component_count > int(max_components):
        reasons.append(f"printable region component count {component_count:,} is above {int(max_components):,}")

    ready = not reasons
    return {
        "status": "ready_for_bambu" if ready else "needs_algorithm_work",
        "ready_for_bambu_print": ready,
        "reasons": reasons,
        "thresholds": {
            "max_pixel_drift": float(max_pixel_drift),
            "max_tiny_islands": int(max_tiny_islands),
            "max_components": int(max_components),
        },
    }


def _bambu_candidate_sort_key(row: dict[str, Any]) -> tuple[int, int, float, int, int]:
    assessment = row.get("bambu_print_ready_assessment") or {}
    repair_stats = row.get("repair_stats") or {}
    source_comparison = row.get("source_comparison") or {}
    ready_penalty = 0 if assessment.get("ready_for_bambu_print") is True else 1
    geometry_penalty = 0 if repair_stats.get("is_watertight") is True and repair_stats.get("body_count") == 1 else 1
    drift = float(source_comparison.get("mean_pixel_drift") or 1e9)
    tiny_islands = int(row.get("bambu_printable_tiny_island_count") or 1e9)
    components = int(row.get("bambu_printable_component_count") or 1e9)
    return (ready_penalty, geometry_penalty, drift, tiny_islands, components)


def _subset_face_indexes(face_count: int, *, max_faces: int = DEFAULT_PREVIEW_FACE_LIMIT) -> np.ndarray:
    if face_count <= max_faces:
        return np.arange(face_count, dtype=np.int64)
    return np.linspace(0, face_count - 1, num=max_faces, dtype=np.int64)


def _write_subset_texture_preview(
    path: Path,
    *,
    positions: np.ndarray,
    faces: np.ndarray,
    texcoords: np.ndarray,
    texture_rgb: np.ndarray,
    max_faces: int = DEFAULT_PREVIEW_FACE_LIMIT,
) -> None:
    face_indexes = _subset_face_indexes(len(faces), max_faces=max_faces)
    _write_texture_source_preview(
        path,
        positions=positions,
        faces=np.asarray(faces, dtype=np.int64)[face_indexes],
        texcoords=texcoords,
        texture_rgb=texture_rgb,
    )


def _write_subset_face_preview(
    path: Path,
    *,
    positions: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
    max_faces: int = DEFAULT_PREVIEW_FACE_LIMIT,
) -> None:
    face_indexes = _subset_face_indexes(len(faces), max_faces=max_faces)
    _write_face_color_preview(
        path,
        positions,
        np.asarray(faces, dtype=np.int64)[face_indexes],
        np.asarray(face_colors, dtype=np.uint8)[face_indexes],
    )


def _clean_mesh(positions: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(positions, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh, multibody=True)
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int64)


def _repair_with_trimesh_clean(positions: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cleaned_positions, cleaned_faces = _clean_mesh(positions, faces)
    mesh = trimesh.Trimesh(vertices=cleaned_positions, faces=cleaned_faces, process=False)
    should_fill_holes = len(cleaned_faces) <= 300000
    filled = bool(mesh.fill_holes()) if should_fill_holes else False
    trimesh.repair.fix_normals(mesh, multibody=True)
    repaired_positions = np.asarray(mesh.vertices, dtype=np.float32)
    repaired_faces = np.asarray(mesh.faces, dtype=np.int64)
    return repaired_positions, repaired_faces, {
        "repair_backend": "trimesh_clean",
        "filled_holes": bool(filled),
        "attempted_fill_holes": bool(should_fill_holes),
    }


def _repair_with_pymeshfix_core(positions: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cleaned_positions, cleaned_faces = _clean_mesh(positions, faces)
    meshfix = _meshfix.PyTMesh()
    meshfix.set_quiet(True)
    meshfix.load_array(
        np.asarray(cleaned_positions, dtype=np.float64),
        np.asarray(cleaned_faces, dtype=np.int32),
    )
    boundaries_before = int(meshfix.n_boundaries)
    meshfix.fill_small_boundaries(0, True)
    meshfix.join_closest_components()
    meshfix.remove_smallest_components()
    meshfix.clean()
    repaired_positions, repaired_faces = meshfix.return_arrays()
    repaired_mesh = trimesh.Trimesh(vertices=repaired_positions, faces=repaired_faces, process=False)
    trimesh.repair.fix_normals(repaired_mesh, multibody=True)
    return (
        np.asarray(repaired_mesh.vertices, dtype=np.float32),
        np.asarray(repaired_mesh.faces, dtype=np.int64),
        {
            "repair_backend": "pymeshfix_core",
            "boundaries_before": boundaries_before,
            "boundaries_after": int(meshfix.n_boundaries),
        },
    )


def _repair_with_voxel_marching_cubes(
    positions: np.ndarray,
    faces: np.ndarray,
    *,
    voxel_divisions: int = VOXEL_REPAIR_DIVISIONS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cleaned_positions, cleaned_faces = _clean_mesh(positions, faces)
    mesh = trimesh.Trimesh(vertices=cleaned_positions, faces=cleaned_faces, process=False)
    extents = np.maximum(np.asarray(mesh.extents, dtype=np.float32), 1e-6)
    division_count = max(32, int(voxel_divisions))
    pitch = float(np.max(extents) / float(division_count))
    voxel_grid = mesh.voxelized(pitch)
    filled_grid = voxel_grid.fill(method="holes")
    repaired_mesh = filled_grid.marching_cubes
    trimesh.repair.fix_normals(repaired_mesh, multibody=True)
    return (
        np.asarray(repaired_mesh.vertices, dtype=np.float32),
        np.asarray(repaired_mesh.faces, dtype=np.int64),
        {
            "repair_backend": "voxel_marching_cubes",
            "voxel_divisions": int(division_count),
            "voxel_pitch": pitch,
            "voxel_shape": [int(value) for value in voxel_grid.shape],
            "filled_voxel_count": int(len(filled_grid.points)),
        },
    )


def _repair_mesh(
    positions: np.ndarray,
    faces: np.ndarray,
    *,
    backend: str,
    voxel_divisions: int = VOXEL_REPAIR_DIVISIONS,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if backend == "trimesh_clean":
        return _repair_with_trimesh_clean(positions, faces)
    if backend == "pymeshfix_core":
        return _repair_with_pymeshfix_core(positions, faces)
    if backend == "voxel_marching_cubes":
        return _repair_with_voxel_marching_cubes(positions, faces, voxel_divisions=voxel_divisions)
    raise ValueError(f"Unsupported repair backend: {backend}")


def _maybe_simplify_mesh(
    positions: np.ndarray,
    faces: np.ndarray,
    *,
    target_face_count: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if target_face_count is None:
        return np.asarray(positions, dtype=np.float32), np.asarray(faces, dtype=np.int64), {
            "simplification_applied": False,
            "target_face_count": None,
        }
    face_array = np.asarray(faces, dtype=np.int64)
    if len(face_array) <= int(target_face_count):
        return np.asarray(positions, dtype=np.float32), face_array, {
            "simplification_applied": False,
            "target_face_count": int(target_face_count),
        }
    mesh = trimesh.Trimesh(vertices=positions, faces=face_array, process=False)
    simplified = mesh.simplify_quadric_decimation(face_count=int(target_face_count))
    return (
        np.asarray(simplified.vertices, dtype=np.float32),
        np.asarray(simplified.faces, dtype=np.int64),
        {
            "simplification_applied": True,
            "target_face_count": int(target_face_count),
            "source_face_count": int(len(face_array)),
            "result_face_count": int(len(simplified.faces)),
        },
    )


def _sample_vertex_color_mesh_to_points(
    positions: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray,
    target_points: np.ndarray,
    *,
    candidate_count: int = 8,
) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.float32)
    face_array = np.asarray(faces, dtype=np.int64)
    colors = np.asarray(vertex_colors, dtype=np.float32)
    points = np.asarray(target_points, dtype=np.float32)
    if len(pos) == 0 or len(face_array) == 0 or len(points) == 0:
        return np.zeros((len(points), 3), dtype=np.uint8)
    triangles = pos[face_array]
    triangle_colors = colors[face_array]
    centroids = face_centroids(pos, face_array).astype(np.float32, copy=False)
    tree = cKDTree(centroids)
    k = min(max(int(candidate_count), 1), len(centroids))
    _, indexes = tree.query(points, k=k)
    if k == 1:
        indexes = np.asarray(indexes, dtype=np.int64)[:, None]
    predicted = np.zeros((len(points), 3), dtype=np.float32)
    for vertex_index, point in enumerate(points):
        candidate_faces = np.asarray(indexes[vertex_index], dtype=np.int64)
        candidate_triangles = triangles[candidate_faces]
        repeated = np.repeat(point[None, :], len(candidate_triangles), axis=0)
        closest_points = trimesh.triangles.closest_point(candidate_triangles, repeated)
        distances = np.linalg.norm(closest_points - repeated, axis=1)
        best_row = int(np.argmin(distances))
        source_face_index = int(candidate_faces[best_row])
        hit_point = closest_points[best_row]
        weights = barycentric_weights(hit_point, triangles[source_face_index])
        predicted[vertex_index] = interpolate_triangle_colors(triangle_colors[source_face_index], weights)
    return np.clip(np.rint(predicted), 0, 255).astype(np.uint8)


def _write_three_panel_board(
    *,
    source_preview_path: Path,
    ours_preview_path: Path,
    output_path: Path,
    provider_preview_path: Path | None = None,
) -> Path:
    source_image = Image.open(source_preview_path).convert("RGB")
    ours_image = Image.open(ours_preview_path).convert("RGB")
    provider_image = None if provider_preview_path is None else Image.open(provider_preview_path).convert("RGB")

    panel_width = 360
    panel_height = 270
    margin = 20
    gap = 16
    labels = ["Raw GLB Source", "Algorithmic Repair + Color"]
    images = [source_image, ours_image]
    if provider_image is not None:
        labels.append("Provider Repaired + Color")
        images.append(provider_image)

    width = margin * 2 + panel_width * len(images) + gap * (len(images) - 1)
    height = 340
    canvas = Image.new("RGB", (width, height), (245, 241, 234))
    draw = ImageDraw.Draw(canvas)
    x = margin
    for label, image in zip(labels, images, strict=False):
        fitted = image.resize((panel_width, panel_height))
        canvas.paste(fitted, (x, 42))
        draw.text((x, 14), label, fill=(28, 28, 28))
        x += panel_width + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def run_repair_then_bake_experiment(
    *,
    source_path: str | Path,
    out_dir: str | Path,
    provider_target_obj_path: str | Path | None = None,
    repair_backends: list[str] | None = None,
    sample_size: int = 12000,
    seed: int = 42,
    bake_variant: dict[str, Any] | None = None,
    bake_method: str | None = None,
    target_face_count: int | None = DEFAULT_TARGET_FACE_COUNT,
    max_colors: int = 8,
) -> dict[str, Any]:
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_loaded = load_textured_model(source_path)
    source_preview_path = output_dir / "source_preview.png"
    _write_subset_texture_preview(
        source_preview_path,
        positions=source_loaded.positions,
        faces=source_loaded.faces,
        texcoords=source_loaded.texcoords,
        texture_rgb=source_loaded.texture_rgb,
    )

    provider_positions = None
    provider_faces = None
    provider_vertex_colors = None
    provider_face_colors = None
    provider_preview_path = None
    if provider_target_obj_path:
        provider_positions, provider_faces, provider_vertex_colors = _load_target_vertex_color_obj(provider_target_obj_path)
        provider_face_colors = _face_colors_from_vertex_colors(provider_faces, provider_vertex_colors)
        provider_preview_path = output_dir / "provider_preview.png"
        _write_subset_face_preview(
            provider_preview_path,
            positions=provider_positions,
            faces=provider_faces,
            face_colors=provider_face_colors,
        )

    if bake_variant:
        variant = ProviderOracleVariant(**bake_variant)
    elif bake_method:
        variant = ProviderOracleVariant(
            label=f"repair_then_bake_{bake_method}",
            method=str(bake_method),
            sampling_mode=DEFAULT_BAKE_VARIANT.sampling_mode,
            uv_flip_y=DEFAULT_BAKE_VARIANT.uv_flip_y,
            candidate_count=DEFAULT_BAKE_VARIANT.candidate_count,
            pad_pixels=DEFAULT_BAKE_VARIANT.pad_pixels,
        )
    else:
        variant = DEFAULT_BAKE_VARIANT
    backend_rows: list[dict[str, Any]] = []
    backends = repair_backends or list(DEFAULT_REPAIR_BACKENDS)
    alignment_sample_size = min(DEFAULT_ALIGNMENT_SAMPLE_SIZE, max(512, int(sample_size)))

    for backend in backends:
        backend_dir = output_dir / backend
        backend_dir.mkdir(parents=True, exist_ok=True)
        try:
            repaired_positions, repaired_faces, repair_metadata = _repair_mesh(
                source_loaded.positions,
                source_loaded.faces,
                backend=backend,
            )
            repaired_positions, repaired_faces, simplify_metadata = _maybe_simplify_mesh(
                repaired_positions,
                repaired_faces,
                target_face_count=target_face_count,
            )
            source_alignment_positions, source_alignment = _align_source_to_target(
                source_loaded.positions,
                repaired_positions,
                sample_size=alignment_sample_size,
                seed=seed,
            )
            prepared_source = _prepare_source_for_variant(source_loaded, source_alignment_positions, variant)
            repaired_mesh = trimesh.Trimesh(vertices=repaired_positions, faces=repaired_faces, process=False)
            repaired_normals = np.asarray(repaired_mesh.vertex_normals, dtype=np.float32)
            predicted_vertex_colors = _predict_variant_colors(
                prepared_source,
                _normalize_points(repaired_positions),
                repaired_normals,
                variant,
            )
            predicted_obj_path = write_obj_with_per_vertex_colors(
                backend_dir / "repaired_baked_vertex_color.obj",
                repaired_positions,
                repaired_faces,
                predicted_vertex_colors.astype(np.float32) / 255.0,
                object_name="RepairedBakedColorMesh",
            )
            predicted_face_colors = _face_colors_from_vertex_colors(repaired_faces, predicted_vertex_colors)
            bambu_report = write_face_color_mesh_to_assets(
                positions=repaired_positions,
                faces=repaired_faces,
                face_colors=predicted_face_colors,
                source_path=source_path,
                out_dir=backend_dir / "bambu_printable",
                max_colors=max_colors,
                object_name="RepairedBakedColorMesh",
                strategy=f"repair_then_bake_{variant.method}_face_palette",
                obj_filename="repaired_baked_region_materials.obj",
                threemf_filename="repaired_baked_region_colorgroup.3mf",
                preview_filename="repaired_baked_region_preview.png",
                swatch_filename="repaired_baked_palette_swatches.png",
                palette_csv_filename="repaired_baked_palette.csv",
                report_filename="repaired_baked_conversion_report.json",
                smooth_iterations=6,
                min_component_size=max(192, min(1024, len(repaired_faces) // 90 or 192)),
                cleanup_passes=8,
                extra_report={
                    "conversion_lane": "repair_then_bake_printable_repaired_obj",
                    "repair_backend": backend,
                    "bake_variant": variant.label,
                    "bake_method": variant.method,
                    "dense_vertex_color_obj_path": str(predicted_obj_path),
                },
            )
            preview_path = backend_dir / "repaired_baked_preview.png"
            _write_subset_face_preview(
                preview_path,
                positions=repaired_positions,
                faces=repaired_faces,
                face_colors=predicted_face_colors,
            )
            source_compare = write_source_export_comparison(
                source_preview_path=source_preview_path,
                export_preview_path=preview_path,
                comparison_path=backend_dir / "source_vs_repaired_comparison.png",
                color_transfer_applied=True,
            )
            repair_stats = _mesh_stats(repaired_positions, repaired_faces)
            bambu_print_ready_assessment = _assess_bambu_print_ready(
                repair_stats=repair_stats,
                source_comparison=source_compare,
                bambu_report=bambu_report,
            )
            row: dict[str, Any] = {
                "backend": backend,
                "repair_metadata": repair_metadata,
                "simplify_metadata": simplify_metadata,
                "repair_stats": repair_stats,
                "source_alignment": source_alignment,
                "predicted_obj_path": str(predicted_obj_path),
                "bambu_printable_report_path": str(bambu_report["report_path"]),
                "bambu_printable_obj_path": str(bambu_report["obj_path"]),
                "bambu_printable_mtl_path": str(bambu_report["mtl_path"]),
                "bambu_printable_3mf_path": str(bambu_report["threemf_path"]),
                "bambu_printable_preview_path": str(bambu_report["preview_path"]),
                "bambu_printable_palette_size": int(bambu_report["palette_size"]),
                "bambu_printable_component_count": int(bambu_report["component_count"]),
                "bambu_printable_tiny_island_count": int(bambu_report["tiny_island_count"]),
                "bambu_print_ready_assessment": bambu_print_ready_assessment,
                "preview_path": str(preview_path),
                "source_comparison": source_compare,
            }

            if provider_positions is not None and provider_faces is not None and provider_vertex_colors is not None and provider_preview_path is not None:
                provider_compare = write_source_export_comparison(
                    source_preview_path=provider_preview_path,
                    export_preview_path=preview_path,
                    comparison_path=backend_dir / "provider_vs_repaired_comparison.png",
                    color_transfer_applied=True,
                )
                repaired_alignment_positions, provider_alignment = _align_source_to_target(
                    repaired_positions,
                    provider_positions,
                    sample_size=alignment_sample_size,
                    seed=seed,
                )
                rng = np.random.default_rng(int(seed))
                sample_indexes = rng.choice(
                    len(provider_positions),
                    size=min(int(sample_size), len(provider_positions)),
                    replace=False,
                )
                provider_points_sample = _normalize_points(provider_positions)[sample_indexes]
                provider_colors_sample = provider_vertex_colors[sample_indexes]
                predicted_provider_sample = _sample_vertex_color_mesh_to_points(
                    repaired_alignment_positions,
                    repaired_faces,
                    predicted_vertex_colors,
                    provider_points_sample,
                    candidate_count=variant.candidate_count,
                )
                provider_metrics = _vertex_color_metrics(predicted_provider_sample, provider_colors_sample)
                board_path = _write_three_panel_board(
                    source_preview_path=source_preview_path,
                    ours_preview_path=preview_path,
                    provider_preview_path=provider_preview_path,
                    output_path=backend_dir / "repair_then_bake_board.png",
                )
                row.update(
                    {
                        "provider_alignment": provider_alignment,
                        "provider_color_metrics": provider_metrics,
                        "provider_comparison": provider_compare,
                        "board_path": str(board_path),
                    }
                )
                np.save(backend_dir / "provider_sample_indexes.npy", sample_indexes)
                np.save(backend_dir / "provider_sample_predicted_colors.npy", predicted_provider_sample)
                np.save(backend_dir / "provider_sample_expected_colors.npy", provider_colors_sample)
            else:
                board_path = _write_three_panel_board(
                    source_preview_path=source_preview_path,
                    ours_preview_path=preview_path,
                    provider_preview_path=None,
                    output_path=backend_dir / "repair_then_bake_board.png",
                )
                row["board_path"] = str(board_path)

            _write_json(backend_dir / "summary.json", row)
            backend_rows.append(row)
        except Exception as exc:
            error_row = {
                "backend": backend,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            _write_json(backend_dir / "summary.json", error_row)
            backend_rows.append(error_row)

    successful_rows = [
        row for row in backend_rows
        if row.get("status") != "error"
    ]
    if provider_target_obj_path:
        successful_rows.sort(
            key=lambda row: (
                float(((row.get("provider_color_metrics") or {}).get("mean_abs_total") or 1e9)),
                float(((row.get("provider_comparison") or {}).get("mean_pixel_drift") or 1e9)),
            )
        )
    else:
        successful_rows.sort(key=_bambu_candidate_sort_key)

    summary = {
        "source_path": str(Path(source_path).expanduser().resolve()),
        "provider_target_obj_path": None if provider_target_obj_path is None else str(Path(provider_target_obj_path).expanduser().resolve()),
        "repair_backends": backends,
        "bake_variant": {
            "label": variant.label,
            "method": variant.method,
            "sampling_mode": variant.sampling_mode,
            "uv_flip_y": bool(variant.uv_flip_y),
            "candidate_count": int(variant.candidate_count),
            "pad_pixels": int(variant.pad_pixels),
        },
        "target_face_count": None if target_face_count is None else int(target_face_count),
        "max_colors": int(max_colors),
        "source_preview_path": str(source_preview_path),
        "provider_preview_path": None if provider_preview_path is None else str(provider_preview_path),
        "results": backend_rows,
        "best_result": successful_rows[0] if successful_rows else None,
        "ready_for_bambu_print": bool(
            successful_rows
            and ((successful_rows[0].get("bambu_print_ready_assessment") or {}).get("ready_for_bambu_print") is True)
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary
