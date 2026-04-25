from __future__ import annotations

from dataclasses import dataclass
import json
from math import cos, radians, sin
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from scipy.spatial import cKDTree
import trimesh

from .bake import bake_texture_to_corner_colors, face_colors_from_corner_colors
from .export_obj import write_bambu_compatible_grouped_obj_with_mtl
from .face_regions import (
    build_face_adjacency,
    compute_face_normals,
    face_centroids,
    merge_small_palette_islands,
    normalize_positions,
    smooth_face_palette_indices,
)
from .model_io import LoadedTexturedMesh
from .pipeline import (
    _build_duck_seeded_parts,
    _legacy_posterize_texture,
    _quantize_face_colors,
    _write_asset_bundle,
)
from .validation import write_bambu_validation_bundle


@dataclass
class AdvancedExperimentResult:
    experiment_name: str
    strategy: str
    report_path: str
    preview_path: str
    validation_report_path: str | None
    comparison_path: str | None
    mean_pixel_drift: float | None
    assessment: str | None


def _derive_source_legacy_corner_labels(
    loaded: LoadedTexturedMesh,
    *,
    max_colors: int,
) -> dict[str, Any]:
    posterized_texture = _legacy_posterize_texture(loaded.texture_rgb, image_palette=max_colors)
    corner_colors, corner_metadata = bake_texture_to_corner_colors(
        posterized_texture,
        loaded.texcoords,
        loaded.faces,
        pad_pixels=max(2, min(8, int(max_colors))),
        sampling_mode="nearest",
    )
    source_face_colors = face_colors_from_corner_colors(corner_colors)
    palette, face_labels = _quantize_face_colors(
        source_face_colors,
        loaded.positions,
        loaded.faces,
        max_colors,
    )
    return {
        "posterized_texture": posterized_texture,
        "source_face_colors": source_face_colors,
        "palette": np.asarray(palette, dtype=np.uint8),
        "face_labels": np.asarray(face_labels, dtype=np.int32),
        "corner_bake_metadata": corner_metadata,
    }


def _rotation_matrix(*, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    yaw = radians(float(yaw_deg))
    pitch = radians(float(pitch_deg))
    rot_y = np.array(
        [
            [cos(yaw), 0.0, sin(yaw)],
            [0.0, 1.0, 0.0],
            [-sin(yaw), 0.0, cos(yaw)],
        ],
        dtype=np.float32,
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos(pitch), -sin(pitch)],
            [0.0, sin(pitch), cos(pitch)],
        ],
        dtype=np.float32,
    )
    return rot_x @ rot_y


def _one_hot_scores(labels: np.ndarray, label_count: int) -> np.ndarray:
    scores = np.zeros((len(labels), max(label_count, 1)), dtype=np.float32)
    if len(labels) == 0 or label_count <= 0:
        return scores
    scores[np.arange(len(labels), dtype=np.int64), np.asarray(labels, dtype=np.int32)] = 1.0
    return scores


def _geodesic_like_refine_labels(
    face_labels: np.ndarray,
    label_scores: np.ndarray,
    faces: np.ndarray,
    positions: np.ndarray,
    *,
    iterations: int = 5,
    smoothness_weight: float = 0.45,
    boundary_power: float = 2.2,
) -> np.ndarray:
    labels = np.asarray(face_labels, dtype=np.int32).copy()
    scores = np.asarray(label_scores, dtype=np.float32)
    if len(labels) == 0 or len(faces) == 0 or scores.size == 0:
        return labels
    adjacency = build_face_adjacency(faces)
    centroids = face_centroids(positions, faces)
    normals = compute_face_normals(positions, faces)
    for _ in range(max(int(iterations), 0)):
        updated = labels.copy()
        for face_index, neighbors in enumerate(adjacency):
            if not neighbors:
                continue
            local_scores = scores[face_index].copy()
            current_centroid = centroids[face_index]
            current_normal = normals[face_index]
            for neighbor in neighbors:
                neighbor_label = int(labels[neighbor])
                edge_length = float(np.linalg.norm(current_centroid - centroids[neighbor]))
                alignment = max(float(np.dot(current_normal, normals[neighbor])), 0.0)
                weight = smoothness_weight * np.exp(-edge_length * 9.0) * np.power(alignment, boundary_power)
                if 0 <= neighbor_label < len(local_scores):
                    local_scores[neighbor_label] += float(weight)
            updated[face_index] = int(np.argmax(local_scores))
        labels = updated
    return labels


def _uv_label_raster(face_labels: np.ndarray, texcoords: np.ndarray, faces: np.ndarray, *, image_size: int = 1024) -> np.ndarray:
    label_img = Image.new("I", (image_size, image_size), 0)
    draw = ImageDraw.Draw(label_img)
    uv = np.asarray(texcoords, dtype=np.float32)
    for face_index, face in enumerate(np.asarray(faces, dtype=np.int64)):
        pts = []
        for vertex_index in face.tolist():
            u, v = uv[int(vertex_index)]
            x = int(np.clip(np.rint(float(u - np.floor(u)) * (image_size - 1)), 0, image_size - 1))
            y = int(np.clip(np.rint((1.0 - float(v - np.floor(v))) * (image_size - 1)), 0, image_size - 1))
            pts.append((x, y))
        draw.polygon(pts, fill=int(face_labels[face_index]) + 1)
    return np.array(label_img, dtype=np.int32)


def _sample_target_face_labels_from_uv_raster(label_raster: np.ndarray, texcoords: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(faces) == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 1), dtype=np.float32)
    height, width = label_raster.shape[:2]
    face_uv = np.asarray(texcoords, dtype=np.float32)[np.asarray(faces, dtype=np.int64)]
    bary = np.array(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.60, 0.20, 0.20],
            [0.20, 0.60, 0.20],
            [0.20, 0.20, 0.60],
        ],
        dtype=np.float32,
    )
    samples = np.tensordot(bary, face_uv, axes=(1, 1))
    samples = np.transpose(samples, (1, 0, 2))
    labels = []
    max_label = 0
    for sample_uvs in samples:
        votes: dict[int, int] = {}
        for u, v in sample_uvs.tolist():
            x = int(np.clip(np.rint(float(u - np.floor(u)) * (width - 1)), 0, width - 1))
            y = int(np.clip(np.rint((1.0 - float(v - np.floor(v))) * (height - 1)), 0, height - 1))
            raw = int(label_raster[y, x]) - 1
            if raw < 0:
                continue
            votes[raw] = votes.get(raw, 0) + 1
            max_label = max(max_label, raw)
        labels.append(max(votes.items(), key=lambda item: item[1])[0] if votes else 0)
    label_array = np.asarray(labels, dtype=np.int32)
    label_scores = _one_hot_scores(label_array, max_label + 1)
    return label_array, label_scores


def _project_for_view(points: np.ndarray, normals: np.ndarray, *, yaw_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = _rotation_matrix(yaw_deg=yaw_deg, pitch_deg=pitch_deg)
    rotated_points = np.asarray(points, dtype=np.float32) @ rotation.T
    rotated_normals = np.asarray(normals, dtype=np.float32) @ rotation.T
    projected = rotated_points[:, :2]
    depth = rotated_points[:, 2]
    return projected, depth, rotated_normals


def _transfer_labels_multiview(
    *,
    source_positions: np.ndarray,
    source_faces: np.ndarray,
    source_face_labels: np.ndarray,
    target_positions: np.ndarray,
    target_faces: np.ndarray,
    label_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_centroids = normalize_positions(face_centroids(source_positions, source_faces))
    target_centroids = normalize_positions(face_centroids(target_positions, target_faces))
    source_normals = compute_face_normals(source_positions, source_faces)
    target_normals = compute_face_normals(target_positions, target_faces)
    views = [
        {"yaw_deg": 0.0, "pitch_deg": 0.0},
        {"yaw_deg": 35.0, "pitch_deg": 0.0},
        {"yaw_deg": -35.0, "pitch_deg": 0.0},
        {"yaw_deg": 0.0, "pitch_deg": 25.0},
        {"yaw_deg": 25.0, "pitch_deg": 18.0},
        {"yaw_deg": -25.0, "pitch_deg": 18.0},
    ]
    scores = np.zeros((len(target_centroids), max(label_count, 1)), dtype=np.float32)
    for view in views:
        src_xy, src_depth, src_normals_view = _project_for_view(source_centroids, source_normals, **view)
        tgt_xy, _, tgt_normals_view = _project_for_view(target_centroids, target_normals, **view)
        visible = src_normals_view[:, 2] < -0.05
        if not np.any(visible):
            continue
        tree = cKDTree(src_xy[visible])
        k = min(8, int(np.sum(visible)))
        distances, indexes = tree.query(tgt_xy, k=k)
        if k == 1:
            distances = distances[:, None]
            indexes = indexes[:, None]
        visible_indices = np.flatnonzero(visible)
        src_idx = visible_indices[np.asarray(indexes, dtype=np.int64)]
        for face_index in range(len(target_centroids)):
            for candidate_index, source_face_index in enumerate(src_idx[face_index].tolist()):
                label = int(source_face_labels[source_face_index])
                dist = float(distances[face_index, candidate_index])
                align = max(float(np.dot(tgt_normals_view[face_index], src_normals_view[source_face_index])), 0.0)
                depth_weight = 1.0 / (1.0 + abs(float(src_depth[source_face_index])))
                scores[face_index, label] += (1.0 / max(dist, 1e-5)) * (0.2 + 0.8 * align) * depth_weight
    if scores.size == 0:
        return np.zeros((len(target_centroids),), dtype=np.int32), scores
    return np.argmax(scores, axis=1).astype(np.int32), scores


def _transfer_labels_closest_face_projection(
    *,
    source_positions: np.ndarray,
    source_faces: np.ndarray,
    source_face_labels: np.ndarray,
    target_positions: np.ndarray,
    target_faces: np.ndarray,
    label_count: int,
    candidates: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    source_triangles = np.asarray(source_positions, dtype=np.float32)[np.asarray(source_faces, dtype=np.int64)]
    source_centroids = face_centroids(source_positions, source_faces)
    source_normals = compute_face_normals(source_positions, source_faces)
    target_centroids = face_centroids(target_positions, target_faces)
    target_normals = compute_face_normals(target_positions, target_faces)
    tree = cKDTree(source_centroids.astype(np.float32))
    k = min(max(int(candidates), 1), len(source_centroids))
    distances, indexes = tree.query(target_centroids.astype(np.float32), k=k)
    if k == 1:
        distances = distances[:, None]
        indexes = indexes[:, None]
    scores = np.zeros((len(target_centroids), max(label_count, 1)), dtype=np.float32)
    for face_index, point in enumerate(np.asarray(target_centroids, dtype=np.float32)):
        candidate_faces = np.asarray(indexes[face_index], dtype=np.int64)
        triangles = source_triangles[candidate_faces]
        points = np.repeat(point[None, :], len(triangles), axis=0)
        closest = trimesh.triangles.closest_point(triangles, points)
        distances_to_surface = np.linalg.norm(closest - points, axis=1)
        for candidate_row, source_face_index in enumerate(candidate_faces.tolist()):
            label = int(source_face_labels[source_face_index])
            alignment = max(float(np.dot(target_normals[face_index], source_normals[source_face_index])), 0.0)
            centroid_weight = 1.0 / max(float(distances[face_index, candidate_row]), 1e-5)
            surface_weight = 1.0 / max(float(distances_to_surface[candidate_row]), 1e-5)
            scores[face_index, label] += centroid_weight * surface_weight * (0.2 + 0.8 * alignment)
    return np.argmax(scores, axis=1).astype(np.int32), scores


def _write_subset_obj(path: Path, positions: np.ndarray, faces: np.ndarray, color: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    face_array = np.asarray(faces, dtype=np.int64)
    used_vertices = np.unique(face_array.reshape(-1))
    remap = {int(old): idx + 1 for idx, old in enumerate(used_vertices.tolist())}
    subset_positions = np.asarray(positions, dtype=np.float32)[used_vertices]
    mtl_path = path.with_suffix(".mtl")
    mtl_name = "part_mat"
    obj_lines = [f"mtllib {mtl_path.name}", f"usemtl {mtl_name}"]
    for vertex in subset_positions.tolist():
        obj_lines.append(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}")
    for face in face_array.tolist():
        a, b, c = (remap[int(face[0])], remap[int(face[1])], remap[int(face[2])])
        obj_lines.append(f"f {a} {b} {c}")
    path.write_text("\n".join(obj_lines) + "\n", encoding="utf-8")
    rgb = np.asarray(color, dtype=np.float32) / 255.0
    mtl_lines = [
        f"newmtl {mtl_name}",
        f"Kd {rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}",
        "Ka 0.000000 0.000000 0.000000",
        "Ks 0.000000 0.000000 0.000000",
        "Ns 32.000000",
        "illum 1",
    ]
    mtl_path.write_text("\n".join(mtl_lines) + "\n", encoding="utf-8")
    return path


def _run_transfer_from_labels(
    *,
    experiment_name: str,
    target_loaded: LoadedTexturedMesh,
    palette: np.ndarray,
    face_labels: np.ndarray,
    output_dir: Path,
    source_preview_path: Path | None,
    probe_exports: list[dict[str, Any]] | None,
    source_mode: str,
    simplify_applied: bool,
    notes: list[str],
    extra_report: dict[str, Any] | None = None,
) -> AdvancedExperimentResult:
    started = perf_counter()
    face_labels = np.asarray(face_labels, dtype=np.int32)
    palette = np.asarray(palette, dtype=np.uint8)
    face_colors = palette[face_labels]
    min_component_size = max(48, min(256, int(len(target_loaded.faces)) // 1800 or 48))
    face_labels = merge_small_palette_islands(
        face_labels,
        face_colors,
        palette,
        target_loaded.faces,
        min_component_size=min_component_size,
    )
    face_labels = smooth_face_palette_indices(
        face_labels,
        palette[face_labels],
        palette,
        target_loaded.faces,
        iterations=3,
    )
    report = _write_asset_bundle(
        loaded=target_loaded,
        face_labels=face_labels,
        palette=palette,
        output_dir=output_dir,
        object_name=experiment_name,
        obj_filename="local_bambu_vertex_colors.obj",
        threemf_filename="local_bambu_palette.3mf",
        preview_filename="local_bambu_export_preview.png",
        swatch_filename="local_bambu_palette_swatches.png",
        palette_csv_filename="local_bambu_palette.csv",
        report_filename="local_bambu_converter_report.json",
        started=started,
        strategy=experiment_name,
        notes=notes,
        extra_report=extra_report,
    )
    validation = write_bambu_validation_bundle(
        output_dir=output_dir,
        source_preview_path=source_preview_path,
        export_preview_path=Path(report["preview_path"]),
        threemf_path=Path(report["threemf_path"]),
        obj_path=Path(report["obj_path"]),
        probe_exports=probe_exports or [],
        source_mode=source_mode,
        simplify_applied=simplify_applied,
        color_transfer_applied=True,
    )
    return AdvancedExperimentResult(
        experiment_name=experiment_name,
        strategy=experiment_name,
        report_path=str(report["report_path"]),
        preview_path=str(report["preview_path"]),
        validation_report_path=(validation or {}).get("validation_report_path"),
        comparison_path=(validation or {}).get("comparison_path"),
        mean_pixel_drift=(validation or {}).get("mean_pixel_drift"),
        assessment=(validation or {}).get("assessment"),
    )


def run_repaired_transfer_experiment_suite(
    *,
    target_loaded: LoadedTexturedMesh,
    color_source_loaded: LoadedTexturedMesh,
    out_dir: str | Path,
    source_preview_path: str | Path | None = None,
    probe_exports: list[dict[str, Any]] | None = None,
    max_colors: int = 12,
    source_mode: str = "repair_simplified_color_transfer",
    simplify_applied: bool = True,
) -> dict[str, Any]:
    output_root = Path(out_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_preview = Path(source_preview_path).expanduser().resolve() if source_preview_path else None

    source_data = _derive_source_legacy_corner_labels(color_source_loaded, max_colors=max_colors)
    palette = np.asarray(source_data["palette"], dtype=np.uint8)
    source_face_labels = np.asarray(source_data["face_labels"], dtype=np.int32)
    label_count = max(int(len(palette)), int(source_face_labels.max()) + 1 if len(source_face_labels) else 0)

    experiments: list[AdvancedExperimentResult] = []

    uv_labels, uv_scores = _sample_target_face_labels_from_uv_raster(
        _uv_label_raster(source_face_labels, color_source_loaded.texcoords, color_source_loaded.faces, image_size=1024),
        target_loaded.texcoords,
        target_loaded.faces,
    )
    experiments.append(
        _run_transfer_from_labels(
            experiment_name="uv_label_transfer",
            target_loaded=target_loaded,
            palette=palette,
            face_labels=uv_labels,
            output_dir=output_root / "uv_label_transfer",
            source_preview_path=source_preview,
            probe_exports=probe_exports,
            source_mode=source_mode,
            simplify_applied=simplify_applied,
            notes=[
                "This experiment rasterizes source face labels into UV space and assigns target face labels by sampling the target UVs directly.",
                "It tests whether 3D correspondence is the main failure point when the repaired mesh still has usable UVs.",
            ],
            extra_report={"region_transfer_mode": "uv_label_transfer"},
        )
    )
    uv_geo = _geodesic_like_refine_labels(uv_labels, uv_scores, target_loaded.faces, target_loaded.positions)
    experiments.append(
        _run_transfer_from_labels(
            experiment_name="uv_label_transfer_geodesic",
            target_loaded=target_loaded,
            palette=palette,
            face_labels=uv_geo,
            output_dir=output_root / "uv_label_transfer_geodesic",
            source_preview_path=source_preview,
            probe_exports=probe_exports,
            source_mode=source_mode,
            simplify_applied=simplify_applied,
            notes=[
                "This experiment starts from UV label transfer and then applies edge-length and normal-aware smoothing on the repaired mesh.",
                "It tests whether region cleanup alone can stabilize UV-space label assignment.",
            ],
            extra_report={"region_transfer_mode": "uv_label_transfer_geodesic"},
        )
    )

    mv_labels, mv_scores = _transfer_labels_multiview(
        source_positions=color_source_loaded.positions,
        source_faces=color_source_loaded.faces,
        source_face_labels=source_face_labels,
        target_positions=target_loaded.positions,
        target_faces=target_loaded.faces,
        label_count=label_count,
    )
    experiments.append(
        _run_transfer_from_labels(
            experiment_name="multiview_projection_transfer",
            target_loaded=target_loaded,
            palette=palette,
            face_labels=mv_labels,
            output_dir=output_root / "multiview_projection_transfer",
            source_preview_path=source_preview,
            probe_exports=probe_exports,
            source_mode=source_mode,
            simplify_applied=simplify_applied,
            notes=[
                "This experiment transfers source labels by voting across multiple orthographic views instead of nearest 3D correspondences.",
                "It tests whether view-space region agreement survives better than mesh-space transfer on stylized ducks.",
            ],
            extra_report={"region_transfer_mode": "multiview_projection_transfer"},
        )
    )
    mv_geo = _geodesic_like_refine_labels(mv_labels, mv_scores, target_loaded.faces, target_loaded.positions)
    experiments.append(
        _run_transfer_from_labels(
            experiment_name="multiview_projection_transfer_geodesic",
            target_loaded=target_loaded,
            palette=palette,
            face_labels=mv_geo,
            output_dir=output_root / "multiview_projection_transfer_geodesic",
            source_preview_path=source_preview,
            probe_exports=probe_exports,
            source_mode=source_mode,
            simplify_applied=simplify_applied,
            notes=[
                "This experiment combines multiview label votes with geodesic-like smoothing and boundary locking.",
                "It tests whether image-space correspondence plus mesh cleanup is stronger than either alone.",
            ],
            extra_report={"region_transfer_mode": "multiview_projection_transfer_geodesic"},
        )
    )

    cf_labels, cf_scores = _transfer_labels_closest_face_projection(
        source_positions=color_source_loaded.positions,
        source_faces=color_source_loaded.faces,
        source_face_labels=source_face_labels,
        target_positions=target_loaded.positions,
        target_faces=target_loaded.faces,
        label_count=label_count,
        candidates=8,
    )
    experiments.append(
        _run_transfer_from_labels(
            experiment_name="closest_face_projection_transfer",
            target_loaded=target_loaded,
            palette=palette,
            face_labels=cf_labels,
            output_dir=output_root / "closest_face_projection_transfer",
            source_preview_path=source_preview,
            probe_exports=probe_exports,
            source_mode=source_mode,
            simplify_applied=simplify_applied,
            notes=[
                "This experiment uses closest source triangles and point-to-triangle projection instead of face-centroid label transfer.",
                "It tests whether a more geometric correspondence improves repaired-mesh region ownership.",
            ],
            extra_report={"region_transfer_mode": "closest_face_projection_transfer"},
        )
    )
    cf_geo = _geodesic_like_refine_labels(cf_labels, cf_scores, target_loaded.faces, target_loaded.positions)
    experiments.append(
        _run_transfer_from_labels(
            experiment_name="closest_face_projection_transfer_geodesic",
            target_loaded=target_loaded,
            palette=palette,
            face_labels=cf_geo,
            output_dir=output_root / "closest_face_projection_transfer_geodesic",
            source_preview_path=source_preview,
            probe_exports=probe_exports,
            source_mode=source_mode,
            simplify_applied=simplify_applied,
            notes=[
                "This experiment combines closest-triangle projection with geodesic-like smoothing and boundary locking.",
                "It tests whether stronger geometric correspondence plus region cleanup closes the repaired-transfer gap.",
            ],
            extra_report={"region_transfer_mode": "closest_face_projection_transfer_geodesic"},
        )
    )

    seeded_labels, seeded_palette, seeded_part_ids = _build_duck_seeded_parts(
        face_colors=np.asarray(source_data["source_face_colors"], dtype=np.uint8),
        positions=color_source_loaded.positions,
        faces=color_source_loaded.faces,
    )
    seeded_same = _write_asset_bundle(
        loaded=color_source_loaded,
        face_labels=seeded_labels,
        palette=seeded_palette,
        output_dir=output_root / "same_mesh_seeded_parts",
        object_name="same_mesh_seeded_parts",
        obj_filename="local_bambu_vertex_colors.obj",
        threemf_filename="local_bambu_palette.3mf",
        preview_filename="local_bambu_export_preview.png",
        swatch_filename="local_bambu_palette_swatches.png",
        palette_csv_filename="local_bambu_palette.csv",
        report_filename="local_bambu_converter_report.json",
        started=perf_counter(),
        strategy="same_mesh_seeded_parts",
        notes=[
            "This same-mesh experiment flattens the source into seeded duck parts before export.",
            "It tests whether explicit print-friendly part zones work well when no repaired transfer is involved.",
        ],
        extra_report={"semantic_part_ids": seeded_part_ids},
    )

    boosted = Image.fromarray(np.asarray(color_source_loaded.texture_rgb, dtype=np.uint8))
    boosted = ImageEnhance.Color(boosted).enhance(1.45)
    boosted = ImageEnhance.Contrast(boosted).enhance(1.35)
    boosted = boosted.filter(ImageFilter.EDGE_ENHANCE_MORE)
    boosted_loaded = LoadedTexturedMesh(
        mesh=None,
        positions=np.asarray(color_source_loaded.positions, dtype=np.float32),
        faces=np.asarray(color_source_loaded.faces, dtype=np.int64),
        texcoords=np.asarray(color_source_loaded.texcoords, dtype=np.float32),
        texture_rgb=np.asarray(boosted, dtype=np.uint8),
        source_path=color_source_loaded.source_path,
        texture_path=color_source_loaded.texture_path,
        source_format=color_source_loaded.source_format,
    )
    boosted_data = _derive_source_legacy_corner_labels(boosted_loaded, max_colors=max_colors)
    boosted_same = _write_asset_bundle(
        loaded=boosted_loaded,
        face_labels=np.asarray(boosted_data["face_labels"], dtype=np.int32),
        palette=np.asarray(boosted_data["palette"], dtype=np.uint8),
        output_dir=output_root / "same_mesh_high_contrast",
        object_name="same_mesh_high_contrast",
        obj_filename="local_bambu_vertex_colors.obj",
        threemf_filename="local_bambu_palette.3mf",
        preview_filename="local_bambu_export_preview.png",
        swatch_filename="local_bambu_palette_swatches.png",
        palette_csv_filename="local_bambu_palette.csv",
        report_filename="local_bambu_converter_report.json",
        started=perf_counter(),
        strategy="same_mesh_high_contrast",
        notes=[
            "This same-mesh experiment boosts saturation, contrast, and edge definition before applying the legacy corner-face pipeline.",
            "It tests whether more print-friendly source art alone improves same-mesh color conversion.",
        ],
        extra_report={"corner_bake_metadata": boosted_data["corner_bake_metadata"]},
    )

    with_drift = [exp for exp in experiments if exp.mean_pixel_drift is not None]
    best_experiment = min(
        with_drift,
        key=lambda item: float(item.mean_pixel_drift),
        default=(experiments[0] if experiments else None),
    )
    multipart_dir = output_root / "multipart_best"
    multipart_manifest: list[dict[str, Any]] = []
    if best_experiment is not None:
        best_report = json.loads(Path(best_experiment.report_path).read_text())
        labels = np.load(Path(best_report["face_palette_indices_path"]))
        palette_npy = np.load(Path(best_report["palette_npy_path"]))
        for part_index in np.unique(labels).tolist():
            face_indexes = np.flatnonzero(labels == int(part_index))
            if len(face_indexes) == 0:
                continue
            subset_faces = np.asarray(target_loaded.faces, dtype=np.int64)[face_indexes]
            part_path = multipart_dir / f"part_{int(part_index):02d}.obj"
            _write_subset_obj(part_path, target_loaded.positions, subset_faces, np.asarray(palette_npy[int(part_index)], dtype=np.uint8))
            multipart_manifest.append(
                {
                    "part_index": int(part_index),
                    "face_count": int(len(face_indexes)),
                    "obj_path": str(part_path),
                    "color": np.asarray(palette_npy[int(part_index)], dtype=np.uint8).tolist(),
                }
            )
        (multipart_dir / "manifest.json").write_text(json.dumps(multipart_manifest, indent=2), encoding="utf-8")

    summary = {
        "status": "ok",
        "experiments": [exp.__dict__ for exp in experiments],
        "best_repaired_experiment": best_experiment.__dict__ if best_experiment else None,
        "same_mesh_seeded_parts": str(Path(seeded_same["report_path"]).resolve()),
        "same_mesh_high_contrast": str(Path(boosted_same["report_path"]).resolve()),
        "multipart_manifest_path": str((multipart_dir / "manifest.json").resolve()) if multipart_manifest else None,
        "blocked_experiments": [
            {
                "name": "manual_part_mask_oracle",
                "reason": "Needs a real human-annotated mask or polygon file for a trustworthy oracle.",
            },
            {
                "name": "template_retopology_experiment",
                "reason": "Needs a canonical duck template mesh or retopo cage, which is not in the workspace.",
            },
            {
                "name": "blender_oracle_comparison",
                "reason": "Requires a local Blender binary; `blender` is not installed in this environment.",
            },
        ],
    }
    summary_path = output_root / "advanced_experiment_suite.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
