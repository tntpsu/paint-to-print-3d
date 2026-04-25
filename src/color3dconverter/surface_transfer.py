from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from .bake import bake_texture_to_corner_colors
from .face_regions import compute_face_normals, face_centroids
from .model_io import LoadedTexturedMesh


def barycentric_weights(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    tri = np.asarray(triangle, dtype=np.float32)
    p = np.asarray(point, dtype=np.float32)
    a = tri[0]
    b = tri[1]
    c = tri[2]
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=np.float32)


def interpolate_triangle_colors(colors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    rgb = np.asarray(colors, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32).reshape((3, 1))
    return np.sum(rgb * w, axis=0).astype(np.float32)


def _source_triangle_corner_colors(
    loaded: LoadedTexturedMesh,
    *,
    sampling_mode: str = "bilinear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    corner_colors, _ = bake_texture_to_corner_colors(
        loaded.texture_rgb,
        loaded.texcoords,
        loaded.faces,
        pad_pixels=4,
        sampling_mode=sampling_mode,
    )
    triangles = np.asarray(loaded.positions, dtype=np.float32)[np.asarray(loaded.faces, dtype=np.int64)]
    triangle_colors = np.asarray(corner_colors, dtype=np.float32)
    centroids = face_centroids(loaded.positions, loaded.faces).astype(np.float32, copy=False)
    normals = compute_face_normals(loaded.positions, loaded.faces).astype(np.float32, copy=False)
    return triangles, triangle_colors, centroids, normals


def transfer_face_colors_nearest_surface(
    *,
    source_loaded: LoadedTexturedMesh,
    target_face_points: np.ndarray,
    candidate_count: int = 8,
    sampling_mode: str = "bilinear",
) -> np.ndarray:
    triangles, triangle_colors, centroids, _ = _source_triangle_corner_colors(source_loaded, sampling_mode=sampling_mode)
    if len(triangles) == 0 or len(target_face_points) == 0:
        return np.zeros((len(target_face_points), 3), dtype=np.uint8)
    tree = cKDTree(centroids.astype(np.float32))
    k = min(max(int(candidate_count), 1), len(centroids))
    distances, indexes = tree.query(np.asarray(target_face_points, dtype=np.float32), k=k)
    if k == 1:
        distances = distances[:, None]
        indexes = indexes[:, None]
    result = np.zeros((len(target_face_points), 3), dtype=np.float32)
    for face_index, point in enumerate(np.asarray(target_face_points, dtype=np.float32)):
        candidate_faces = np.asarray(indexes[face_index], dtype=np.int64)
        candidate_triangles = triangles[candidate_faces]
        repeated_points = np.repeat(point[None, :], len(candidate_triangles), axis=0)
        closest_points = trimesh.triangles.closest_point(candidate_triangles, repeated_points)
        surface_distances = np.linalg.norm(closest_points - repeated_points, axis=1)
        best_row = int(np.argmin(surface_distances))
        source_face_index = int(candidate_faces[best_row])
        hit_point = closest_points[best_row]
        weights = barycentric_weights(hit_point, triangles[source_face_index])
        result[face_index] = interpolate_triangle_colors(triangle_colors[source_face_index], weights)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def _ray_triangle_intersection(origin: np.ndarray, direction: np.ndarray, triangle: np.ndarray) -> tuple[float | None, np.ndarray | None]:
    eps = 1e-8
    tri = np.asarray(triangle, dtype=np.float32)
    o = np.asarray(origin, dtype=np.float32)
    d = np.asarray(direction, dtype=np.float32)
    v0, v1, v2 = tri
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = np.cross(d, e2)
    det = float(np.dot(e1, pvec))
    if abs(det) < eps:
        return None, None
    inv_det = 1.0 / det
    tvec = o - v0
    u = float(np.dot(tvec, pvec)) * inv_det
    if u < 0.0 or u > 1.0:
        return None, None
    qvec = np.cross(tvec, e1)
    v = float(np.dot(d, qvec)) * inv_det
    if v < 0.0 or u + v > 1.0:
        return None, None
    t = float(np.dot(e2, qvec)) * inv_det
    if t <= eps:
        return None, None
    point = o + d * t
    return t, point.astype(np.float32)


def transfer_face_colors_raycast(
    *,
    source_loaded: LoadedTexturedMesh,
    target_face_points: np.ndarray,
    target_face_normals: np.ndarray,
    candidate_count: int = 12,
    sampling_mode: str = "bilinear",
) -> np.ndarray:
    triangles, triangle_colors, centroids, source_normals = _source_triangle_corner_colors(source_loaded, sampling_mode=sampling_mode)
    if len(triangles) == 0 or len(target_face_points) == 0:
        return np.zeros((len(target_face_points), 3), dtype=np.uint8)
    tree = cKDTree(centroids.astype(np.float32))
    k = min(max(int(candidate_count), 1), len(centroids))
    _, indexes = tree.query(np.asarray(target_face_points, dtype=np.float32), k=k)
    if k == 1:
        indexes = indexes[:, None]
    fallback = transfer_face_colors_nearest_surface(
        source_loaded=source_loaded,
        target_face_points=target_face_points,
        candidate_count=candidate_count,
        sampling_mode=sampling_mode,
    )
    result = fallback.astype(np.float32)
    points = np.asarray(target_face_points, dtype=np.float32)
    normals = np.asarray(target_face_normals, dtype=np.float32)
    for face_index, point in enumerate(points):
        normal = normals[face_index]
        normal_len = float(np.linalg.norm(normal))
        if normal_len <= 1e-8:
            continue
        direction_base = normal / normal_len
        best_t = None
        best_face_index = None
        best_point = None
        candidate_faces = np.asarray(indexes[face_index], dtype=np.int64)
        for direction in (direction_base, -direction_base):
            for source_face_index in candidate_faces.tolist():
                align = float(np.dot(direction, source_normals[int(source_face_index)]))
                if align > 0.98:
                    continue
                t, hit = _ray_triangle_intersection(point, direction, triangles[int(source_face_index)])
                if t is None:
                    continue
                if best_t is None or t < best_t:
                    best_t = t
                    best_face_index = int(source_face_index)
                    best_point = hit
        if best_face_index is None or best_point is None:
            continue
        weights = barycentric_weights(best_point, triangles[best_face_index])
        result[face_index] = interpolate_triangle_colors(triangle_colors[best_face_index], weights)
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)
