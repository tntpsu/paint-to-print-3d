from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from trimesh.registration import icp

from .bake import bake_texture_to_corner_colors, sample_texture_bilinear, seam_pad_texture
from .export_obj_vertex_colors import write_obj_with_per_vertex_colors
from .face_regions import compute_face_normals, face_centroids, sample_texture
from .model_io import LoadedTexturedMesh, load_textured_model
from .surface_transfer import barycentric_weights, interpolate_triangle_colors, _ray_triangle_intersection


@dataclass(frozen=True)
class ProviderOracleVariant:
    label: str
    method: str
    sampling_mode: str = "bilinear"
    uv_flip_y: bool = True
    candidate_count: int = 8
    pad_pixels: int = 4
    distance_power: float = 2.0
    normal_power: float = 0.0
    smooth_neighbors: int = 0
    smooth_blend: float = 0.0
    shading_folds: int = 5


DEFAULT_PROVIDER_ORACLE_VARIANTS: tuple[ProviderOracleVariant, ...] = (
    ProviderOracleVariant("nearest_vertex_bilinear_vflip", method="nearest_vertex", sampling_mode="bilinear", uv_flip_y=True),
    ProviderOracleVariant("nearest_vertex_nearest_vflip", method="nearest_vertex", sampling_mode="nearest", uv_flip_y=True),
    ProviderOracleVariant("nearest_surface_uv_bilinear_vflip_k8_p4", method="nearest_surface_uv", sampling_mode="bilinear", uv_flip_y=True, candidate_count=8, pad_pixels=4),
    ProviderOracleVariant("nearest_surface_uv_bilinear_vnoflip_k8_p4", method="nearest_surface_uv", sampling_mode="bilinear", uv_flip_y=False, candidate_count=8, pad_pixels=4),
    ProviderOracleVariant("weighted_surface_uv_bilinear_vflip_k8_p4", method="weighted_surface_uv", sampling_mode="bilinear", uv_flip_y=True, candidate_count=8, pad_pixels=4, distance_power=2.0),
    ProviderOracleVariant("nearest_surface_corner_bilinear_vflip_k8_p4", method="nearest_surface_corner", sampling_mode="bilinear", uv_flip_y=True, candidate_count=8, pad_pixels=4),
    ProviderOracleVariant("weighted_surface_corner_bilinear_vflip_k8_p4", method="weighted_surface_corner", sampling_mode="bilinear", uv_flip_y=True, candidate_count=8, pad_pixels=4, distance_power=2.0),
    ProviderOracleVariant("raycast_uv_bilinear_vflip_k12_p4", method="raycast_uv", sampling_mode="bilinear", uv_flip_y=True, candidate_count=12, pad_pixels=4),
)


def _normalize_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return pts
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    scale = float(np.max(np.maximum(bbox_max - bbox_min, 1e-6)))
    return (pts - center) / max(scale, 1e-6)


def _transform_texcoords(texcoords: np.ndarray, *, uv_flip_y: bool) -> np.ndarray:
    uv = np.asarray(texcoords, dtype=np.float32).copy()
    if not uv_flip_y and len(uv):
        uv[:, 1] = 1.0 - uv[:, 1]
    return uv


def _sample_texture_mode(texture_rgb: np.ndarray, texcoords: np.ndarray, *, sampling_mode: str) -> np.ndarray:
    if sampling_mode == "bilinear":
        return sample_texture_bilinear(texture_rgb, texcoords)
    if sampling_mode == "nearest":
        return sample_texture(texture_rgb, texcoords)
    raise ValueError(f"Unsupported sampling mode: {sampling_mode}")


def _load_target_vertex_color_obj(target_obj_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh = trimesh.load(str(Path(target_obj_path).expanduser().resolve()), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Target OBJ did not load as a mesh.")
    positions = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not hasattr(mesh.visual, "vertex_colors"):
        raise ValueError("Target OBJ does not contain vertex colors.")
    vertex_colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.uint8)
    return positions, faces, vertex_colors


def _apply_alignment_summary(source_positions: np.ndarray, alignment_summary: dict[str, Any]) -> np.ndarray:
    source_norm = _normalize_points(source_positions).astype(np.float64, copy=False)
    perm = tuple(int(value) for value in alignment_summary.get("best_perm") or [0, 1, 2])
    sign = tuple(float(value) for value in alignment_summary.get("best_sign") or [1.0, 1.0, 1.0])
    rotation = np.asarray(alignment_summary.get("rotation_matrix") or np.eye(3), dtype=np.float64)
    translation = np.asarray(alignment_summary.get("translation") or [0.0, 0.0, 0.0], dtype=np.float64)
    signed_full = source_norm @ (np.diag(np.asarray(sign, dtype=np.float64)) @ np.eye(3, dtype=np.float64)[list(perm)]).T
    aligned = (signed_full @ rotation.T) + translation
    return aligned.astype(np.float32)


def _align_source_to_target(source_positions: np.ndarray, target_positions: np.ndarray, *, sample_size: int = 12000, seed: int = 42) -> tuple[np.ndarray, dict[str, Any]]:
    source_norm = _normalize_points(source_positions).astype(np.float64, copy=False)
    target_norm = _normalize_points(target_positions).astype(np.float64, copy=False)
    if len(source_norm) == 0 or len(target_norm) == 0:
        return source_norm.astype(np.float32), {
            "best_icp_cost": 0.0,
            "best_perm": [0, 1, 2],
            "best_sign": [1.0, 1.0, 1.0],
        }

    rng = np.random.default_rng(int(seed))
    src_idx = rng.choice(len(source_norm), size=min(int(sample_size), len(source_norm)), replace=False)
    tgt_idx = rng.choice(len(target_norm), size=min(int(sample_size), len(target_norm)), replace=False)
    src_sample = source_norm[src_idx]
    tgt_sample = target_norm[tgt_idx]

    best_cost = None
    best_perm = (0, 1, 2)
    best_sign = (1.0, 1.0, 1.0)
    best_matrix = np.eye(4, dtype=np.float64)
    for perm in itertools.permutations(range(3)):
        permute = np.eye(3, dtype=np.float64)[list(perm)]
        for sign in itertools.product([-1.0, 1.0], repeat=3):
            signed = np.diag(np.asarray(sign, dtype=np.float64)) @ permute
            transformed = src_sample @ signed.T
            matrix, _, cost = icp(
                transformed,
                tgt_sample,
                max_iterations=30,
                threshold=1e-6,
                scale=False,
            )
            if best_cost is None or float(cost) < float(best_cost):
                best_cost = float(cost)
                best_perm = perm
                best_sign = sign
                best_matrix = np.asarray(matrix, dtype=np.float64)

    alignment_summary = {
        "best_icp_cost": float(best_cost if best_cost is not None else 0.0),
        "best_perm": [int(value) for value in best_perm],
        "best_sign": [float(value) for value in best_sign],
        "rotation_matrix": best_matrix[:3, :3].round(8).tolist(),
        "translation": best_matrix[:3, 3].round(8).tolist(),
    }
    aligned = _apply_alignment_summary(source_positions, alignment_summary)
    return aligned.astype(np.float32), alignment_summary


def _prepare_source_for_variant(source_loaded: LoadedTexturedMesh, aligned_positions: np.ndarray, variant: ProviderOracleVariant) -> LoadedTexturedMesh:
    return LoadedTexturedMesh(
        mesh=None,
        positions=np.asarray(aligned_positions, dtype=np.float32),
        faces=np.asarray(source_loaded.faces, dtype=np.int64),
        texcoords=_transform_texcoords(source_loaded.texcoords, uv_flip_y=variant.uv_flip_y),
        texture_rgb=np.asarray(source_loaded.texture_rgb, dtype=np.uint8),
        source_path=source_loaded.source_path,
        texture_path=source_loaded.texture_path,
        source_format=source_loaded.source_format,
        normal_texture_rgb=None if source_loaded.normal_texture_rgb is None else np.asarray(source_loaded.normal_texture_rgb, dtype=np.uint8),
        orm_texture_rgb=None if source_loaded.orm_texture_rgb is None else np.asarray(source_loaded.orm_texture_rgb, dtype=np.uint8),
        base_color_factor=None if source_loaded.base_color_factor is None else np.asarray(source_loaded.base_color_factor, dtype=np.float32),
        metallic_factor=float(source_loaded.metallic_factor),
        roughness_factor=float(source_loaded.roughness_factor),
        normal_scale=float(source_loaded.normal_scale),
    )


def _predict_nearest_vertex_colors(
    source_loaded: LoadedTexturedMesh,
    target_points: np.ndarray,
    *,
    sampling_mode: str,
) -> np.ndarray:
    tree = cKDTree(np.asarray(source_loaded.positions, dtype=np.float32))
    _, nearest = tree.query(np.asarray(target_points, dtype=np.float32), k=1)
    sample_uv = np.asarray(source_loaded.texcoords, dtype=np.float32)[np.asarray(nearest, dtype=np.int64)]
    return _sample_texture_mode(source_loaded.texture_rgb, sample_uv, sampling_mode=sampling_mode)


def _predict_nearest_surface_uv_colors(
    source_loaded: LoadedTexturedMesh,
    target_points: np.ndarray,
    *,
    candidate_count: int,
    sampling_mode: str,
    pad_pixels: int,
) -> np.ndarray:
    triangles = np.asarray(source_loaded.positions, dtype=np.float32)[np.asarray(source_loaded.faces, dtype=np.int64)]
    triangle_uvs = np.asarray(source_loaded.texcoords, dtype=np.float32)[np.asarray(source_loaded.faces, dtype=np.int64)]
    centroids = face_centroids(source_loaded.positions, source_loaded.faces).astype(np.float32, copy=False)
    if len(triangles) == 0 or len(target_points) == 0:
        return np.zeros((len(target_points), 3), dtype=np.uint8)
    padded_texture, _, _ = seam_pad_texture(
        source_loaded.texture_rgb,
        source_loaded.texcoords,
        source_loaded.faces,
        pad_pixels=int(pad_pixels),
    )
    tree = cKDTree(centroids)
    k = min(max(int(candidate_count), 1), len(centroids))
    _, indexes = tree.query(np.asarray(target_points, dtype=np.float32), k=k)
    if k == 1:
        indexes = np.asarray(indexes, dtype=np.int64)[:, None]
    predicted = np.zeros((len(target_points), 3), dtype=np.uint8)
    for vertex_index, point in enumerate(np.asarray(target_points, dtype=np.float32)):
        candidate_faces = np.asarray(indexes[vertex_index], dtype=np.int64)
        candidate_triangles = triangles[candidate_faces]
        repeated = np.repeat(point[None, :], len(candidate_triangles), axis=0)
        closest_points = trimesh.triangles.closest_point(candidate_triangles, repeated)
        distances = np.linalg.norm(closest_points - repeated, axis=1)
        best_row = int(np.argmin(distances))
        source_face_index = int(candidate_faces[best_row])
        hit_point = closest_points[best_row]
        weights = barycentric_weights(hit_point, triangles[source_face_index])
        uv = np.sum(triangle_uvs[source_face_index] * weights[:, None], axis=0, dtype=np.float32)[None, :]
        predicted[vertex_index] = _sample_texture_mode(padded_texture, uv, sampling_mode=sampling_mode)[0]
    return predicted


def _compute_candidate_hits(
    source_loaded: LoadedTexturedMesh,
    target_points: np.ndarray,
    *,
    candidate_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    triangles = np.asarray(source_loaded.positions, dtype=np.float32)[np.asarray(source_loaded.faces, dtype=np.int64)]
    centroids = face_centroids(source_loaded.positions, source_loaded.faces).astype(np.float32, copy=False)
    if len(triangles) == 0 or len(target_points) == 0:
        return triangles, centroids, np.zeros((len(target_points), 0), dtype=np.int64), np.zeros((len(target_points), 0, 3), dtype=np.float32), np.zeros((len(target_points), 0), dtype=np.float32)
    tree = cKDTree(centroids)
    k = min(max(int(candidate_count), 1), len(centroids))
    _, indexes = tree.query(np.asarray(target_points, dtype=np.float32), k=k)
    if k == 1:
        indexes = np.asarray(indexes, dtype=np.int64)[:, None]
    target = np.asarray(target_points, dtype=np.float32)
    closest_points = np.zeros((len(target), indexes.shape[1], 3), dtype=np.float32)
    distances = np.zeros((len(target), indexes.shape[1]), dtype=np.float32)
    for row_index, point in enumerate(target):
        candidate_faces = np.asarray(indexes[row_index], dtype=np.int64)
        candidate_triangles = triangles[candidate_faces]
        repeated = np.repeat(point[None, :], len(candidate_triangles), axis=0)
        closest = trimesh.triangles.closest_point(candidate_triangles, repeated).astype(np.float32, copy=False)
        closest_points[row_index] = closest
        distances[row_index] = np.linalg.norm(closest - repeated, axis=1).astype(np.float32, copy=False)
    return triangles, centroids, np.asarray(indexes, dtype=np.int64), closest_points, distances


def _candidate_weights(
    distances: np.ndarray,
    *,
    target_normal: np.ndarray | None = None,
    source_normals: np.ndarray | None = None,
    distance_power: float = 2.0,
    normal_power: float = 0.0,
) -> np.ndarray:
    weights = 1.0 / np.maximum(np.asarray(distances, dtype=np.float32), 1e-6) ** max(float(distance_power), 1e-6)
    if target_normal is not None and source_normals is not None and float(normal_power) > 0.0:
        normal = np.asarray(target_normal, dtype=np.float32)
        norm = float(np.linalg.norm(normal))
        if norm > 1e-8:
            unit = normal / norm
            align = np.abs(np.asarray(source_normals, dtype=np.float32) @ unit)
            weights *= np.maximum(align, 1e-3) ** float(normal_power)
    total = float(weights.sum())
    if total <= 1e-12:
        return np.full_like(weights, 1.0 / max(len(weights), 1), dtype=np.float32)
    return (weights / total).astype(np.float32, copy=False)


def _predict_nearest_surface_corner_colors(
    source_loaded: LoadedTexturedMesh,
    target_points: np.ndarray,
    *,
    candidate_count: int,
    sampling_mode: str,
    pad_pixels: int,
) -> np.ndarray:
    triangles = np.asarray(source_loaded.positions, dtype=np.float32)[np.asarray(source_loaded.faces, dtype=np.int64)]
    centroids = face_centroids(source_loaded.positions, source_loaded.faces).astype(np.float32, copy=False)
    if len(triangles) == 0 or len(target_points) == 0:
        return np.zeros((len(target_points), 3), dtype=np.uint8)
    corner_colors, _ = bake_texture_to_corner_colors(
        source_loaded.texture_rgb,
        source_loaded.texcoords,
        source_loaded.faces,
        pad_pixels=int(pad_pixels),
        sampling_mode=sampling_mode,
    )
    tree = cKDTree(centroids)
    k = min(max(int(candidate_count), 1), len(centroids))
    _, indexes = tree.query(np.asarray(target_points, dtype=np.float32), k=k)
    if k == 1:
        indexes = np.asarray(indexes, dtype=np.int64)[:, None]
    predicted = np.zeros((len(target_points), 3), dtype=np.uint8)
    for vertex_index, point in enumerate(np.asarray(target_points, dtype=np.float32)):
        candidate_faces = np.asarray(indexes[vertex_index], dtype=np.int64)
        candidate_triangles = triangles[candidate_faces]
        repeated = np.repeat(point[None, :], len(candidate_triangles), axis=0)
        closest_points = trimesh.triangles.closest_point(candidate_triangles, repeated)
        distances = np.linalg.norm(closest_points - repeated, axis=1)
        best_row = int(np.argmin(distances))
        source_face_index = int(candidate_faces[best_row])
        hit_point = closest_points[best_row]
        weights = barycentric_weights(hit_point, triangles[source_face_index])
        color = interpolate_triangle_colors(corner_colors[source_face_index], weights)
        predicted[vertex_index] = np.clip(np.rint(color), 0, 255).astype(np.uint8)
    return predicted


def _predict_weighted_surface_uv_colors(
    source_loaded: LoadedTexturedMesh,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    *,
    candidate_count: int,
    sampling_mode: str,
    pad_pixels: int,
    distance_power: float,
    normal_power: float,
) -> np.ndarray:
    triangles, _, indexes, closest_points, distances = _compute_candidate_hits(
        source_loaded,
        target_points,
        candidate_count=candidate_count,
    )
    triangle_uvs = np.asarray(source_loaded.texcoords, dtype=np.float32)[np.asarray(source_loaded.faces, dtype=np.int64)]
    source_normals = compute_face_normals(source_loaded.positions, source_loaded.faces).astype(np.float32, copy=False)
    if len(triangles) == 0 or len(target_points) == 0:
        return np.zeros((len(target_points), 3), dtype=np.uint8)
    padded_texture, _, _ = seam_pad_texture(
        source_loaded.texture_rgb,
        source_loaded.texcoords,
        source_loaded.faces,
        pad_pixels=int(pad_pixels),
    )
    predicted = np.zeros((len(target_points), 3), dtype=np.float32)
    for vertex_index in range(len(target_points)):
        candidate_faces = indexes[vertex_index]
        candidate_uv = []
        for row_index, source_face_index in enumerate(candidate_faces.tolist()):
            hit_point = closest_points[vertex_index, row_index]
            weights = barycentric_weights(hit_point, triangles[int(source_face_index)])
            uv = np.sum(triangle_uvs[int(source_face_index)] * weights[:, None], axis=0, dtype=np.float32)
            candidate_uv.append(uv)
        candidate_uv_array = np.asarray(candidate_uv, dtype=np.float32)
        candidate_colors = _sample_texture_mode(padded_texture, candidate_uv_array, sampling_mode=sampling_mode).astype(np.float32)
        weights = _candidate_weights(
            distances[vertex_index],
            target_normal=np.asarray(target_normals[vertex_index], dtype=np.float32) if len(target_normals) else None,
            source_normals=source_normals[candidate_faces],
            distance_power=distance_power,
            normal_power=normal_power,
        )
        predicted[vertex_index] = np.sum(candidate_colors * weights[:, None], axis=0)
    return np.clip(np.rint(predicted), 0, 255).astype(np.uint8)


def _predict_weighted_surface_corner_colors(
    source_loaded: LoadedTexturedMesh,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    *,
    candidate_count: int,
    sampling_mode: str,
    pad_pixels: int,
    distance_power: float,
    normal_power: float,
) -> np.ndarray:
    triangles, _, indexes, closest_points, distances = _compute_candidate_hits(
        source_loaded,
        target_points,
        candidate_count=candidate_count,
    )
    source_normals = compute_face_normals(source_loaded.positions, source_loaded.faces).astype(np.float32, copy=False)
    if len(triangles) == 0 or len(target_points) == 0:
        return np.zeros((len(target_points), 3), dtype=np.uint8)
    corner_colors, _ = bake_texture_to_corner_colors(
        source_loaded.texture_rgb,
        source_loaded.texcoords,
        source_loaded.faces,
        pad_pixels=int(pad_pixels),
        sampling_mode=sampling_mode,
    )
    predicted = np.zeros((len(target_points), 3), dtype=np.float32)
    for vertex_index in range(len(target_points)):
        candidate_faces = indexes[vertex_index]
        candidate_colors = np.zeros((len(candidate_faces), 3), dtype=np.float32)
        for row_index, source_face_index in enumerate(candidate_faces.tolist()):
            hit_point = closest_points[vertex_index, row_index]
            weights = barycentric_weights(hit_point, triangles[int(source_face_index)])
            candidate_colors[row_index] = interpolate_triangle_colors(corner_colors[int(source_face_index)], weights)
        weights = _candidate_weights(
            distances[vertex_index],
            target_normal=np.asarray(target_normals[vertex_index], dtype=np.float32) if len(target_normals) else None,
            source_normals=source_normals[candidate_faces],
            distance_power=distance_power,
            normal_power=normal_power,
        )
        predicted[vertex_index] = np.sum(candidate_colors * weights[:, None], axis=0)
    return np.clip(np.rint(predicted), 0, 255).astype(np.uint8)


def _smooth_predicted_colors(
    target_points: np.ndarray,
    vertex_colors: np.ndarray,
    *,
    neighbors: int,
    blend: float,
) -> np.ndarray:
    if int(neighbors) <= 1 or float(blend) <= 0.0 or len(target_points) == 0:
        return np.asarray(vertex_colors, dtype=np.uint8)
    points = np.asarray(target_points, dtype=np.float32)
    colors = np.asarray(vertex_colors, dtype=np.float32)
    tree = cKDTree(points)
    k = min(max(int(neighbors), 2), len(points))
    distances, indexes = tree.query(points, k=k)
    if k == 1:
        return np.asarray(vertex_colors, dtype=np.uint8)
    if distances.ndim == 1:
        distances = distances[:, None]
        indexes = indexes[:, None]
    neighbor_distances = np.asarray(distances[:, 1:], dtype=np.float32)
    neighbor_indexes = np.asarray(indexes[:, 1:], dtype=np.int64)
    weights = 1.0 / np.maximum(neighbor_distances, 1e-6)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
    smoothed = np.sum(colors[neighbor_indexes] * weights[:, :, None], axis=1)
    result = colors * (1.0 - float(blend)) + smoothed * float(blend)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def _shade_target_scalars(base_colors: np.ndarray, expected_colors: np.ndarray) -> np.ndarray:
    base = np.asarray(base_colors, dtype=np.float32)
    expected = np.asarray(expected_colors, dtype=np.float32)
    numerators = np.sum(base * expected, axis=1)
    denominators = np.sum(base * base, axis=1)
    return np.clip(np.divide(numerators, np.maximum(denominators, 1e-6)), 0.0, 2.0).astype(np.float32)


def _shade_features(target_points: np.ndarray, target_normals: np.ndarray, base_colors: np.ndarray) -> np.ndarray:
    points = np.asarray(target_points, dtype=np.float32)
    normals = np.asarray(target_normals, dtype=np.float32)
    colors = np.asarray(base_colors, dtype=np.float32) / 255.0
    px = points[:, 0]
    py = points[:, 1]
    pz = points[:, 2]
    nx = normals[:, 0]
    ny = normals[:, 1]
    nz = normals[:, 2]
    luminance = colors @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    color_max = colors.max(axis=1)
    color_min = colors.min(axis=1)
    chroma = color_max - color_min
    radial = np.linalg.norm(points, axis=1)
    xy_radius = np.sqrt(np.maximum(px * px + py * py, 0.0))
    position_basis = np.stack(
        [
            px * px,
            py * py,
            pz * pz,
            px * py,
            py * pz,
            pz * px,
            radial,
            xy_radius,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    # Second-order normal terms approximate simple baked-lighting structure better than
    # raw normals alone and help the repaired model share signal across ducks.
    normal_basis = np.stack(
        [
            nx * ny,
            ny * nz,
            nz * nx,
            nx * nx - ny * ny,
            3.0 * nz * nz - 1.0,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    return np.concatenate(
        [
            points,
            normals,
            colors,
            position_basis,
            normal_basis,
            luminance[:, None],
            chroma[:, None],
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _oracle_shading_scalars(
    target_points: np.ndarray,
    target_normals: np.ndarray,
    base_colors: np.ndarray,
    expected_colors: np.ndarray,
    *,
    model_kind: str,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    features = _shade_features(target_points, target_normals, base_colors)
    targets = _shade_target_scalars(base_colors, expected_colors)
    count = len(targets)
    if count == 0:
        return np.zeros((0,), dtype=np.float32), {"shade_scalar_mae": 0.0}

    fold_count = max(2, min(int(folds), count))
    permutation = np.random.default_rng(int(seed)).permutation(count)
    buckets = np.array_split(permutation, fold_count)
    predicted = np.zeros((count,), dtype=np.float32)

    for bucket in buckets:
        test_idx = np.asarray(bucket, dtype=np.int64)
        train_mask = np.ones(count, dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.nonzero(train_mask)[0]
        if len(train_idx) == 0:
            predicted[test_idx] = targets[test_idx]
            continue
        if model_kind == "ridge":
            model = Ridge(alpha=1.0)
        elif model_kind == "rf":
            model = RandomForestRegressor(
                n_estimators=80,
                max_depth=16,
                random_state=int(seed),
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unsupported oracle shading model: {model_kind}")
        model.fit(features[train_idx], targets[train_idx])
        predicted[test_idx] = np.clip(model.predict(features[test_idx]), 0.0, 2.0)

    mae = float(np.abs(predicted - targets).mean())
    return predicted.astype(np.float32), {
        "shade_scalar_mae": mae,
        "shade_scalar_target_mean": float(targets.mean()),
        "shade_scalar_target_min": float(targets.min()),
        "shade_scalar_target_max": float(targets.max()),
    }


def _predict_raycast_uv_colors(
    source_loaded: LoadedTexturedMesh,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    *,
    candidate_count: int,
    sampling_mode: str,
    pad_pixels: int,
) -> np.ndarray:
    triangles = np.asarray(source_loaded.positions, dtype=np.float32)[np.asarray(source_loaded.faces, dtype=np.int64)]
    triangle_uvs = np.asarray(source_loaded.texcoords, dtype=np.float32)[np.asarray(source_loaded.faces, dtype=np.int64)]
    centroids = face_centroids(source_loaded.positions, source_loaded.faces).astype(np.float32, copy=False)
    source_normals = compute_face_normals(source_loaded.positions, source_loaded.faces).astype(np.float32, copy=False)
    if len(triangles) == 0 or len(target_points) == 0:
        return np.zeros((len(target_points), 3), dtype=np.uint8)
    padded_texture, _, _ = seam_pad_texture(
        source_loaded.texture_rgb,
        source_loaded.texcoords,
        source_loaded.faces,
        pad_pixels=int(pad_pixels),
    )
    tree = cKDTree(centroids)
    k = min(max(int(candidate_count), 1), len(centroids))
    _, indexes = tree.query(np.asarray(target_points, dtype=np.float32), k=k)
    if k == 1:
        indexes = np.asarray(indexes, dtype=np.int64)[:, None]
    predicted = np.zeros((len(target_points), 3), dtype=np.uint8)
    fallback = _predict_nearest_surface_uv_colors(
        source_loaded,
        target_points,
        candidate_count=candidate_count,
        sampling_mode=sampling_mode,
        pad_pixels=pad_pixels,
    )
    for vertex_index, point in enumerate(np.asarray(target_points, dtype=np.float32)):
        normal = np.asarray(target_normals[vertex_index], dtype=np.float32)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-8:
            predicted[vertex_index] = fallback[vertex_index]
            continue
        direction_base = normal / norm
        best_t = None
        best_face_index = None
        best_point = None
        candidate_faces = np.asarray(indexes[vertex_index], dtype=np.int64)
        for direction in (direction_base, -direction_base):
            for source_face_index in candidate_faces.tolist():
                align = float(np.dot(direction, source_normals[int(source_face_index)]))
                if align > 0.98:
                    continue
                distance, hit_point = _ray_triangle_intersection(point, direction, triangles[int(source_face_index)])
                if distance is None or hit_point is None:
                    continue
                if best_t is None or float(distance) < float(best_t):
                    best_t = float(distance)
                    best_face_index = int(source_face_index)
                    best_point = hit_point
        if best_face_index is None or best_point is None:
            predicted[vertex_index] = fallback[vertex_index]
            continue
        weights = barycentric_weights(best_point, triangles[best_face_index])
        uv = np.sum(triangle_uvs[best_face_index] * weights[:, None], axis=0, dtype=np.float32)[None, :]
        predicted[vertex_index] = _sample_texture_mode(padded_texture, uv, sampling_mode=sampling_mode)[0]
    return predicted


def _predict_variant_colors(
    source_loaded: LoadedTexturedMesh,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    variant: ProviderOracleVariant,
) -> np.ndarray:
    if variant.method == "nearest_vertex":
        return _predict_nearest_vertex_colors(source_loaded, target_points, sampling_mode=variant.sampling_mode)
    if variant.method == "nearest_surface_uv":
        return _predict_nearest_surface_uv_colors(
            source_loaded,
            target_points,
            candidate_count=variant.candidate_count,
            sampling_mode=variant.sampling_mode,
            pad_pixels=variant.pad_pixels,
        )
    if variant.method == "nearest_surface_corner":
        return _predict_nearest_surface_corner_colors(
            source_loaded,
            target_points,
            candidate_count=variant.candidate_count,
            sampling_mode=variant.sampling_mode,
            pad_pixels=variant.pad_pixels,
        )
    if variant.method == "weighted_surface_uv":
        predicted = _predict_weighted_surface_uv_colors(
            source_loaded,
            target_points,
            target_normals,
            candidate_count=variant.candidate_count,
            sampling_mode=variant.sampling_mode,
            pad_pixels=variant.pad_pixels,
            distance_power=variant.distance_power,
            normal_power=variant.normal_power,
        )
        return _smooth_predicted_colors(
            target_points,
            predicted,
            neighbors=variant.smooth_neighbors,
            blend=variant.smooth_blend,
        )
    if variant.method == "weighted_surface_corner":
        predicted = _predict_weighted_surface_corner_colors(
            source_loaded,
            target_points,
            target_normals,
            candidate_count=variant.candidate_count,
            sampling_mode=variant.sampling_mode,
            pad_pixels=variant.pad_pixels,
            distance_power=variant.distance_power,
            normal_power=variant.normal_power,
        )
        return _smooth_predicted_colors(
            target_points,
            predicted,
            neighbors=variant.smooth_neighbors,
            blend=variant.smooth_blend,
        )
    if variant.method == "raycast_uv":
        return _predict_raycast_uv_colors(
            source_loaded,
            target_points,
            target_normals,
            candidate_count=variant.candidate_count,
            sampling_mode=variant.sampling_mode,
            pad_pixels=variant.pad_pixels,
        )
    raise ValueError(f"Unsupported provider-oracle method: {variant.method}")


def _predict_oracle_shaded_variant_colors(
    source_loaded: LoadedTexturedMesh,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    expected_colors: np.ndarray,
    variant: ProviderOracleVariant,
    *,
    seed: int,
    base_colors: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if base_colors is None:
        base_colors = _predict_nearest_surface_uv_colors(
            source_loaded,
            target_points,
            candidate_count=variant.candidate_count,
            sampling_mode=variant.sampling_mode,
            pad_pixels=variant.pad_pixels,
        ).astype(np.float32)
    else:
        base_colors = np.asarray(base_colors, dtype=np.float32)
    if variant.method == "oracle_shaded_surface_uv_ridge":
        model_kind = "ridge"
    elif variant.method == "oracle_shaded_surface_uv_rf":
        model_kind = "rf"
    else:
        raise ValueError(f"Unsupported oracle shaded method: {variant.method}")
    shade_scalars, metadata = _oracle_shading_scalars(
        target_points,
        target_normals,
        base_colors,
        np.asarray(expected_colors, dtype=np.float32),
        model_kind=model_kind,
        folds=variant.shading_folds,
        seed=seed,
    )
    shaded = np.clip(base_colors * shade_scalars[:, None], 0.0, 255.0)
    return np.clip(np.rint(shaded), 0, 255).astype(np.uint8), {
        **metadata,
        "base_mean_abs_total": float(np.abs(base_colors - np.asarray(expected_colors, dtype=np.float32)).mean()),
    }


def _vertex_color_metrics(predicted: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    pred = np.asarray(predicted, dtype=np.uint8)
    exp = np.asarray(expected, dtype=np.uint8)
    if len(pred) != len(exp):
        raise ValueError("predicted and expected colors must align")
    diff = np.abs(pred.astype(np.int16) - exp.astype(np.int16))
    return {
        "sampled_count": int(len(pred)),
        "mean_abs_rgb": [float(value) for value in diff.mean(axis=0).tolist()] if len(diff) else [0.0, 0.0, 0.0],
        "mean_abs_total": float(diff.mean()) if len(diff) else 0.0,
        "fraction_within_8": float(np.mean(np.all(diff <= 8, axis=1))) if len(diff) else 1.0,
        "fraction_within_16": float(np.mean(np.all(diff <= 16, axis=1))) if len(diff) else 1.0,
        "fraction_within_32": float(np.mean(np.all(diff <= 32, axis=1))) if len(diff) else 1.0,
        "expected_unique_colors": int(len(np.unique(exp, axis=0))) if len(exp) else 0,
        "predicted_unique_colors": int(len(np.unique(pred, axis=0))) if len(pred) else 0,
    }


def _face_colors_from_vertex_colors(faces: np.ndarray, vertex_colors: np.ndarray) -> np.ndarray:
    face_array = np.asarray(faces, dtype=np.int64)
    colors = np.asarray(vertex_colors, dtype=np.uint8)
    if len(face_array) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    averaged = np.clip(np.rint(colors[face_array].mean(axis=1)), 0, 255).astype(np.uint8)
    return averaged


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_provider_oracle_experiments(
    *,
    source_path: str | Path,
    target_obj_path: str | Path,
    out_dir: str | Path,
    sample_size: int = 5000,
    seed: int = 42,
    variants: list[dict[str, Any]] | None = None,
    export_best_full: bool = False,
    alignment_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_loaded = load_textured_model(source_path)
    target_positions, target_faces, target_vertex_colors = _load_target_vertex_color_obj(target_obj_path)
    if alignment_summary is None:
        aligned_source_positions, alignment_summary = _align_source_to_target(source_loaded.positions, target_positions, seed=seed)
    else:
        aligned_source_positions = _apply_alignment_summary(source_loaded.positions, alignment_summary)

    _write_json(output_dir / "alignment_summary.json", alignment_summary)

    variant_rows: list[dict[str, Any]] = []
    variant_specs = [
        ProviderOracleVariant(**spec) for spec in variants
    ] if variants else list(DEFAULT_PROVIDER_ORACLE_VARIANTS)

    rng = np.random.default_rng(int(seed))
    sample_indexes = rng.choice(len(target_positions), size=min(int(sample_size), len(target_positions)), replace=False)
    sampled_points = np.asarray(_normalize_points(target_positions)[sample_indexes], dtype=np.float32)
    sampled_expected = np.asarray(target_vertex_colors[sample_indexes], dtype=np.uint8)
    needs_normals = (
        any(spec.method in {"raycast_uv"} for spec in variant_specs)
        or any(spec.method in {"weighted_surface_uv", "weighted_surface_corner"} and float(spec.normal_power) > 0.0 for spec in variant_specs)
        or any(spec.method in {"oracle_shaded_surface_uv_ridge", "oracle_shaded_surface_uv_rf"} for spec in variant_specs)
    )
    target_normals = np.zeros_like(target_positions, dtype=np.float32)
    sampled_normals = np.zeros((len(sample_indexes), 3), dtype=np.float32)
    if needs_normals:
        target_mesh = trimesh.Trimesh(vertices=target_positions, faces=target_faces, process=False)
        target_normals = np.asarray(target_mesh.vertex_normals, dtype=np.float32)
        if len(target_normals) != len(target_positions):
            target_normals = np.zeros_like(target_positions, dtype=np.float32)
        sampled_normals = np.asarray(target_normals[sample_indexes], dtype=np.float32)

    sampled_prediction_cache: dict[tuple[Any, ...], np.ndarray] = {}

    for variant in variant_specs:
        variant_dir = output_dir / variant.label
        variant_dir.mkdir(parents=True, exist_ok=True)
        prepared_source = _prepare_source_for_variant(source_loaded, aligned_source_positions, variant)
        extra_metadata: dict[str, Any] = {}
        base_cache_key = (
            "nearest_surface_uv",
            variant.sampling_mode,
            bool(variant.uv_flip_y),
            int(variant.candidate_count),
            int(variant.pad_pixels),
        )
        if variant.method in {"oracle_shaded_surface_uv_ridge", "oracle_shaded_surface_uv_rf"}:
            base_colors = sampled_prediction_cache.get(base_cache_key)
            if base_colors is None:
                base_colors = _predict_nearest_surface_uv_colors(
                    prepared_source,
                    sampled_points,
                    candidate_count=variant.candidate_count,
                    sampling_mode=variant.sampling_mode,
                    pad_pixels=variant.pad_pixels,
                )
                sampled_prediction_cache[base_cache_key] = np.asarray(base_colors, dtype=np.uint8)
            predicted, extra_metadata = _predict_oracle_shaded_variant_colors(
                prepared_source,
                sampled_points,
                sampled_normals,
                sampled_expected,
                variant,
                seed=seed,
                base_colors=base_colors,
            )
        else:
            variant_cache_key = (
                variant.method,
                variant.sampling_mode,
                bool(variant.uv_flip_y),
                int(variant.candidate_count),
                int(variant.pad_pixels),
                float(variant.distance_power),
                float(variant.normal_power),
                int(variant.smooth_neighbors),
                float(variant.smooth_blend),
            )
            predicted = sampled_prediction_cache.get(variant_cache_key)
            if predicted is None:
                predicted = _predict_variant_colors(
                    prepared_source,
                    sampled_points,
                    sampled_normals,
                    variant,
                )
                sampled_prediction_cache[variant_cache_key] = np.asarray(predicted, dtype=np.uint8)
            if variant.method == "nearest_surface_uv":
                sampled_prediction_cache[base_cache_key] = np.asarray(predicted, dtype=np.uint8)
        metrics = _vertex_color_metrics(predicted, sampled_expected)
        row = {
            "label": variant.label,
            "method": variant.method,
            "sampling_mode": variant.sampling_mode,
            "uv_flip_y": bool(variant.uv_flip_y),
            "candidate_count": int(variant.candidate_count),
            "pad_pixels": int(variant.pad_pixels),
            "distance_power": float(variant.distance_power),
            "normal_power": float(variant.normal_power),
            "smooth_neighbors": int(variant.smooth_neighbors),
            "smooth_blend": float(variant.smooth_blend),
            "shading_folds": int(variant.shading_folds),
            **metrics,
            **extra_metadata,
            "variant_dir": str(variant_dir),
        }
        _write_json(variant_dir / "summary.json", row)
        np.save(variant_dir / "sample_indexes.npy", sample_indexes)
        np.save(variant_dir / "predicted_vertex_colors.npy", predicted)
        np.save(variant_dir / "expected_vertex_colors.npy", sampled_expected)
        variant_rows.append(row)

    variant_rows.sort(key=lambda item: (float(item["mean_abs_total"]), -float(item["fraction_within_16"])))
    best_result = variant_rows[0] if variant_rows else None

    summary = {
        "source_path": str(Path(source_path).expanduser().resolve()),
        "target_obj_path": str(Path(target_obj_path).expanduser().resolve()),
        "sample_size": int(len(sample_indexes)),
        "alignment": alignment_summary,
        "variant_count": int(len(variant_rows)),
        "best_result": best_result,
        "variants": variant_rows,
    }

    if export_best_full and best_result is not None:
        best_variant = next(spec for spec in variant_specs if spec.label == best_result["label"])
        prepared_source = _prepare_source_for_variant(source_loaded, aligned_source_positions, best_variant)
        full_predicted = _predict_variant_colors(
            prepared_source,
            _normalize_points(target_positions),
            target_normals,
            best_variant,
        )
        full_dir = output_dir / best_variant.label
        full_metrics = _vertex_color_metrics(full_predicted, target_vertex_colors)
        predicted_obj_path = write_obj_with_per_vertex_colors(
            full_dir / "predicted_vertex_color.obj",
            target_positions,
            target_faces,
            full_predicted.astype(np.float32) / 255.0,
            object_name="PredictedVertexColorMesh",
        )
        full_face_colors = _face_colors_from_vertex_colors(target_faces, full_predicted)
        expected_face_colors = _face_colors_from_vertex_colors(target_faces, target_vertex_colors)
        np.save(full_dir / "predicted_vertex_colors_full.npy", full_predicted)
        np.save(full_dir / "expected_vertex_colors_full.npy", target_vertex_colors)
        np.save(full_dir / "predicted_face_colors_full.npy", full_face_colors)
        np.save(full_dir / "expected_face_colors_full.npy", expected_face_colors)
        full_summary = {
            "predicted_obj_path": str(predicted_obj_path),
            **full_metrics,
        }
        _write_json(full_dir / "full_summary.json", full_summary)
        summary["best_full_result"] = full_summary

    _write_json(output_dir / "summary.json", summary)
    return summary
