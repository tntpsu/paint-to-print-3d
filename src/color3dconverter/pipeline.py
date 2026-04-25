from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
import trimesh

from .bake import (
    bake_texture_to_corner_colors,
    bake_texture_to_vertex_colors,
    face_colors_from_corner_colors,
)
from .color_adjustments import (
    apply_brightness_contrast,
    apply_hue_saturation,
    apply_layer_blend,
    apply_levels,
    posterize,
    remap,
)
from .export_3mf import write_colorgroup_3mf
from .export_obj import write_bambu_compatible_grouped_obj_with_mtl
from .export_obj_vertex_colors import write_obj_with_vertex_colors
from .face_regions import (
    average_by_cluster,
    build_connected_face_components,
    build_region_first_face_palette,
    compact_palette,
    compute_face_normals,
    face_areas,
    face_centroids,
    merge_small_palette_islands,
    nearest_palette_indices,
    normalize_positions,
    refine_face_labels_with_graph_smoothing,
    sample_texture,
    smooth_face_palette_indices,
    transfer_face_region_ownership,
    transfer_vertex_colors_from_source,
    weighted_kmeans_palette,
)
from .model_io import LoadedTexturedMesh, load_geometry_model, load_textured_model
from .regions import assign_faces_to_texture_regions, build_texture_regions, clean_texture_regions


@dataclass(frozen=True)
class SourceFaceRegionModel:
    palette: np.ndarray
    face_labels: np.ndarray
    anchor_labels: dict[str, int | None]
    metadata: dict[str, Any]


def _legacy_find_distinct_colors(colors: np.ndarray, threshold: float = 30.0) -> list[np.ndarray]:
    if len(colors) <= 1:
        return [np.asarray(color, dtype=np.float32) for color in colors]
    distinct = [np.asarray(colors[0], dtype=np.float32)]
    for color in np.asarray(colors[1:], dtype=np.float32):
        is_distinct = True
        for existing in distinct:
            if float(np.linalg.norm(color - existing)) < threshold:
                is_distinct = False
                break
        if is_distinct:
            distinct.append(np.asarray(color, dtype=np.float32))
    return distinct


def _legacy_posterize_texture(texture_rgb: np.ndarray, image_palette: int) -> np.ndarray:
    pixels = np.asarray(texture_rgb, dtype=np.uint8).reshape((-1, 3))
    if len(pixels) == 0:
        return np.asarray(texture_rgb, dtype=np.uint8).copy()
    cluster_count = max(1, min(int(image_palette), len(pixels)))
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    posterized = np.asarray(texture_rgb, dtype=np.uint8).copy().reshape((-1, 3))
    posterized[:] = np.clip(np.rint(kmeans.cluster_centers_[labels]), 0, 255).astype(np.uint8)
    return posterized.reshape(np.asarray(texture_rgb).shape)


def _legacy_sample_vertex_colors(texture_rgb: np.ndarray, texcoords: np.ndarray) -> np.ndarray:
    texture = np.asarray(texture_rgb, dtype=np.uint8)
    height, width = texture.shape[:2]
    sampled = np.zeros((len(texcoords), 3), dtype=np.uint8)
    for index, uv in enumerate(np.asarray(texcoords, dtype=np.float32)):
        u = float(uv[0])
        v = float(uv[1])
        x = int(u * (width - 1))
        y = int((1.0 - v) * (height - 1))
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        sampled[index] = texture[y, x]
    return sampled


def _legacy_quantize_vertex_colors(vertex_colors: np.ndarray, num_colors: int) -> tuple[np.ndarray, np.ndarray]:
    if len(vertex_colors) == 0:
        return np.zeros((0, 3), dtype=np.uint8), np.zeros((0,), dtype=np.int32)
    colors_uint8 = np.asarray(vertex_colors, dtype=np.uint8)
    unique_colors = np.unique(colors_uint8, axis=0)
    if len(unique_colors) <= max(1, int(num_colors)):
        labels = nearest_palette_indices(colors_uint8.astype(np.float32), unique_colors.astype(np.float32))
        return unique_colors.astype(np.uint8), labels.astype(np.int32)
    analysis_clusters = max(1, min(16, max(int(num_colors), len(colors_uint8) // 100)))
    analysis_kmeans = KMeans(n_clusters=analysis_clusters, random_state=42, n_init=10)
    analysis_kmeans.fit(colors_uint8)
    distinct_colors = _legacy_find_distinct_colors(np.asarray(analysis_kmeans.cluster_centers_, dtype=np.float32))
    final_num_colors = max(1, min(len(distinct_colors), int(num_colors)))
    kmeans = KMeans(n_clusters=final_num_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(np.asarray(vertex_colors, dtype=np.float32))
    palette = np.clip(np.rint(kmeans.cluster_centers_), 0, 255).astype(np.uint8)
    return palette, labels.astype(np.int32)


def _legacy_assign_face_labels(faces: np.ndarray, vertex_labels: np.ndarray) -> np.ndarray:
    face_labels = np.zeros((len(faces),), dtype=np.int32)
    for face_index, face in enumerate(np.asarray(faces, dtype=np.int64)):
        label_counts = Counter(
            [
                int(vertex_labels[int(face[0])]),
                int(vertex_labels[int(face[1])]),
                int(vertex_labels[int(face[2])]),
            ]
        )
        face_labels[face_index] = int(label_counts.most_common(1)[0][0])
    return face_labels


def _build_legacy_source_face_region_model(color_source_loaded: LoadedTexturedMesh, *, max_colors: int) -> SourceFaceRegionModel:
    posterized_texture = _legacy_posterize_texture(color_source_loaded.texture_rgb, image_palette=max_colors)
    source_vertex_colors = _legacy_sample_vertex_colors(posterized_texture, color_source_loaded.texcoords)
    palette, source_vertex_labels = _legacy_quantize_vertex_colors(source_vertex_colors, num_colors=max_colors)
    source_face_labels = _legacy_assign_face_labels(color_source_loaded.faces, source_vertex_labels)
    anchor_labels = _infer_duck_part_anchor_labels(
        face_labels=source_face_labels,
        palette=palette,
        positions=color_source_loaded.positions,
        faces=color_source_loaded.faces,
    )
    return SourceFaceRegionModel(
        palette=np.asarray(palette, dtype=np.uint8),
        face_labels=np.asarray(source_face_labels, dtype=np.int32),
        anchor_labels=anchor_labels,
        metadata={
            "source_region_model": "legacy_fast_face_regions",
            "source_palette_size": int(len(palette)),
            "source_face_label_count": int(len(np.unique(source_face_labels))) if len(source_face_labels) else 0,
        },
    )


def _quantize_face_colors(face_colors: np.ndarray, positions: np.ndarray, faces: np.ndarray, num_colors: int) -> tuple[np.ndarray, np.ndarray]:
    colors = np.asarray(face_colors, dtype=np.uint8)
    if len(colors) == 0:
        return np.zeros((0, 3), dtype=np.uint8), np.zeros((0,), dtype=np.int32)
    unique_colors = np.unique(colors, axis=0)
    if len(unique_colors) <= max(1, int(num_colors)):
        labels = nearest_palette_indices(colors.astype(np.float32), unique_colors.astype(np.float32))
        return unique_colors.astype(np.uint8), labels.astype(np.int32)
    weights = np.sqrt(np.maximum(face_areas(positions, faces), 1e-8))
    palette, labels = weighted_kmeans_palette(colors.astype(np.float32), weights, max_colors=max(1, int(num_colors)))
    return np.asarray(palette, dtype=np.uint8), np.asarray(labels, dtype=np.int32)


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    if values.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    r, g, b = values[:, 0], values[:, 1], values[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc
    safe_delta = np.where(delta > 0.0, delta, 1.0)
    s = np.divide(delta, maxc, out=np.zeros_like(delta), where=maxc > 0.0)
    rc = (maxc - r) / safe_delta
    gc = (maxc - g) / safe_delta
    bc = (maxc - b) / safe_delta
    h = np.where(
        r == maxc,
        bc - gc,
        np.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc),
    )
    h = (h / 6.0) % 1.0
    h = np.where(delta > 0.0, h, 0.0)
    return np.column_stack([h, s, v]).astype(np.float32)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    values = np.asarray(hsv, dtype=np.float32)
    if values.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    h, s, v = values[:, 0], values[:, 1], values[:, 2]
    i = (h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.column_stack([r, g, b]).astype(np.float32)


def _apply_same_mesh_blender_cleanup(face_colors: np.ndarray, *, n_regions: int) -> np.ndarray:
    colors = np.asarray(face_colors, dtype=np.float32) / 255.0
    if colors.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    adjusted = colors.copy()
    adjusted = np.clip((adjusted - 0.5) * 1.18 + 0.5 + 0.015, 0.0, 1.0)
    hsv = _rgb_to_hsv(adjusted)
    hsv[:, 1] = np.clip(hsv[:, 1] * 1.18, 0.0, 1.0)
    hsv[:, 2] = np.clip(hsv[:, 2] * 1.05, 0.0, 1.0)
    adjusted = _hsv_to_rgb(hsv)
    levels = max(3, min(8, int(n_regions)))
    adjusted = np.round(adjusted * (levels - 1)) / float(levels - 1)
    return np.clip(np.rint(adjusted * 255.0), 0, 255).astype(np.uint8)


def _apply_same_mesh_hue_vcm_cleanup(face_colors: np.ndarray, *, n_regions: int) -> np.ndarray:
    colors = np.asarray(face_colors, dtype=np.float32) / 255.0
    if colors.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    luminance = colors @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    in_black = float(np.percentile(colors, 6.0))
    in_white = float(np.percentile(colors, 94.0))
    if in_white - in_black < 0.08:
        in_black, in_white = 0.04, 0.96

    adjusted = apply_levels(colors, in_black, in_white, gamma=0.92, out_black=0.0, out_white=1.0)
    adjusted = np.clip(apply_brightness_contrast(adjusted, brightness=0.018, contrast=0.22), 0.0, 1.0)
    adjusted = np.clip(apply_hue_saturation(adjusted, hue_shift=0.5, saturation=1.16, value=1.04), 0.0, 1.0)

    poster_levels = max(3, min(8, int(n_regions) + 1))
    poster_layer = posterize(adjusted, poster_levels)
    adjusted = np.clip(apply_layer_blend(adjusted, poster_layer, "OVERLAY", 0.68), 0.0, 1.0)

    remapped_luma = remap(
        luminance,
        float(np.percentile(luminance, 8.0)),
        float(np.percentile(luminance, 92.0)),
        0.0,
        1.0,
    )
    luma_layer = np.repeat(remapped_luma[:, None], 3, axis=1)
    adjusted = np.clip(apply_layer_blend(adjusted, luma_layer, "SCREEN", 0.15), 0.0, 1.0)
    return np.clip(np.rint(adjusted * 255.0), 0, 255).astype(np.uint8)


def _legacy_transfer_vertex_labels_from_source(
    *,
    source_positions: np.ndarray,
    source_vertex_labels: np.ndarray,
    target_positions: np.ndarray,
    neighbors: int = 4,
    chunk_size: int = 2048,
) -> np.ndarray:
    source_points = normalize_positions(source_positions)
    target_points = normalize_positions(target_positions)
    if len(source_points) == 0 or len(target_points) == 0:
        return np.zeros((len(target_points),), dtype=np.int32)
    neighbor_count = max(1, min(int(neighbors), len(source_points)))
    source_labels = np.asarray(source_vertex_labels, dtype=np.int32)
    transferred = np.zeros((len(target_points),), dtype=np.int32)
    for start in range(0, len(target_points), chunk_size):
        stop = min(start + chunk_size, len(target_points))
        chunk = target_points[start:stop]
        distances = ((chunk[:, None, :] - source_points[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argpartition(distances, kth=neighbor_count - 1, axis=1)[:, :neighbor_count]
        nearest_distances = np.take_along_axis(distances, nearest, axis=1)
        nearest_labels = source_labels[nearest]
        weights = 1.0 / np.maximum(nearest_distances, 1e-8)
        for row_index in range(stop - start):
            votes: dict[int, float] = {}
            for label, weight in zip(nearest_labels[row_index].tolist(), weights[row_index].tolist(), strict=False):
                label_int = int(label)
                votes[label_int] = votes.get(label_int, 0.0) + float(weight)
            transferred[start + row_index] = int(max(votes.items(), key=lambda item: item[1])[0])
    return transferred


def _duck_color_signals(color: np.ndarray) -> dict[str, float]:
    rgb = np.asarray(color, dtype=np.float32) / 255.0
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    blue = max(0.0, b - 0.55 * r - 0.35 * g)
    warm = max(0.0, 0.55 * r + 0.35 * g - 0.45 * b)
    yellow = max(0.0, min(r, g) - 0.75 * b)
    red = max(0.0, r - 0.65 * g - 0.45 * b)
    orange = max(0.0, 0.75 * r + 0.35 * g - 1.1 * b - 0.2 * abs(r - g))
    brown = max(0.0, 0.45 * r + 0.25 * g - 0.25 * b - 0.35 * max(g - r, 0.0))
    tan = max(0.0, warm + 0.25 * min(r, g) - 0.4 * b)
    neutral = max(0.0, 1.0 - (max(r, g, b) - min(r, g, b)) * 4.0)
    return {
        "blue": blue,
        "warm": warm,
        "yellow": yellow,
        "red": red,
        "orange": orange,
        "brown": brown,
        "tan": tan,
        "neutral": neutral,
    }


def _relative_luminance(color: np.ndarray) -> float:
    rgb = np.asarray(color, dtype=np.float32) / 255.0
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _closeness(value: float, center: float, radius: float) -> float:
    if radius <= 1e-6:
        return 0.0
    return max(0.0, 1.0 - abs(value - center) / radius)


def _infer_duck_part_anchor_labels(
    *,
    face_labels: np.ndarray,
    palette: np.ndarray,
    positions: np.ndarray,
    faces: np.ndarray,
) -> dict[str, int | None]:
    labels = np.asarray(face_labels, dtype=np.int32)
    if len(labels) == 0 or len(faces) == 0 or len(palette) == 0:
        return {"body": None, "bandana": None, "hat": None, "beak": None}

    component_ids = build_connected_face_components(labels, faces)
    centroids = normalize_positions(face_centroids(positions, faces))
    total_faces = max(len(labels), 1)
    component_rows: list[dict[str, Any]] = []
    for component_id in range(int(component_ids.max()) + 1 if len(component_ids) else 0):
        member_indexes = np.flatnonzero(component_ids == component_id)
        if len(member_indexes) < 12:
            continue
        component_labels = labels[member_indexes]
        unique_labels, counts = np.unique(component_labels, return_counts=True)
        dominant_label = int(unique_labels[int(np.argmax(counts))])
        centroid = centroids[member_indexes].mean(axis=0)
        signals = _duck_color_signals(palette[dominant_label])
        component_rows.append(
            {
                "label": dominant_label,
                "size": float(len(member_indexes)) / float(total_faces),
                "centroid": centroid,
                "signals": signals,
            }
        )

    if not component_rows:
        return {"body": None, "bandana": None, "hat": None, "beak": None}

    def scored_labels_for_role(role: str) -> list[tuple[float, int]]:
        scored: list[tuple[float, int]] = []
        for row in component_rows:
            x, y, z = (float(row["centroid"][0]), float(row["centroid"][1]), float(row["centroid"][2]))
            size = float(row["size"])
            signals = row["signals"]
            if role == "body":
                score = (
                    2.35 * signals["yellow"]
                    + 0.55 * signals["warm"]
                    + 0.7 * size
                    + 0.35 * _closeness(y, -0.05, 0.42)
                    + 0.2 * _closeness(x, 0.0, 0.45)
                    - 0.45 * signals["brown"]
                    - 0.9 * max(y - 0.18, 0.0)
                )
            elif role == "bandana":
                score = (
                    2.2 * signals["red"]
                    + 0.55 * max(x, 0.0)
                    + 0.45 * _closeness(y, -0.04, 0.22)
                    + 0.2 * size
                )
            elif role == "hat":
                score = (
                    1.6 * max(signals["brown"], signals["tan"])
                    + 1.65 * max(y, 0.0)
                    + 0.25 * _closeness(x, 0.0, 0.55)
                    + 0.15 * size
                    - 1.95 * signals["yellow"]
                    - 0.45 * signals["red"]
                )
            elif role == "beak":
                score = (
                    2.2 * signals["orange"]
                    + 0.45 * signals["red"]
                    + 0.35 * signals["warm"]
                    + 1.15 * max(x, 0.0)
                    + 0.55 * _closeness(y, 0.0, 0.2)
                    + 0.2 * _closeness(z, 0.0, 0.2)
                    - 0.8 * signals["brown"]
                    - 0.35 * signals["yellow"]
                )
            else:
                continue
            scored.append((float(score), int(row["label"])))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    used_labels: set[int] = set()
    chosen: dict[str, int | None] = {}
    for role in ("bandana", "beak", "body", "hat"):
        selected: int | None = None
        for _, label in scored_labels_for_role(role):
            if label in used_labels:
                continue
            selected = label
            used_labels.add(label)
            break
        chosen[role] = selected
    return {
        "body": chosen.get("body"),
        "bandana": chosen.get("bandana"),
        "hat": chosen.get("hat"),
        "beak": chosen.get("beak"),
    }


def _apply_duck_part_anchor_bias(
    *,
    face_labels: np.ndarray,
    palette: np.ndarray,
    face_colors: np.ndarray,
    positions: np.ndarray,
    faces: np.ndarray,
    anchor_labels: dict[str, int | None],
) -> np.ndarray:
    labels = np.asarray(face_labels, dtype=np.int32).copy()
    if len(labels) == 0 or len(faces) == 0 or len(palette) == 0:
        return labels
    centroids = normalize_positions(face_centroids(positions, faces))
    palette_f = np.asarray(palette, dtype=np.float32)
    face_colors_f = np.asarray(face_colors, dtype=np.float32)
    x = centroids[:, 0]
    y = centroids[:, 1]
    z = centroids[:, 2]

    def vec_closeness(values: np.ndarray, center: float, radius: float) -> np.ndarray:
        if radius <= 1e-6:
            return np.zeros_like(values, dtype=np.float32)
        return np.maximum(0.0, 1.0 - np.abs(values - center) / radius).astype(np.float32, copy=False)

    def role_zone_score(role: str) -> np.ndarray:
        if role == "body":
            return vec_closeness(x, 0.03, 0.38) * vec_closeness(y, -0.08, 0.32) * vec_closeness(z, 0.0, 0.28)
        if role == "bandana":
            return vec_closeness(x, 0.14, 0.22) * vec_closeness(y, -0.04, 0.18) * vec_closeness(z, 0.0, 0.22)
        if role == "hat":
            return np.maximum(0.0, (y - 0.08) / 0.32).astype(np.float32, copy=False) * vec_closeness(x, 0.02, 0.45) * vec_closeness(z, 0.0, 0.34)
        if role == "beak":
            return np.maximum(0.0, (x - 0.16) / 0.22).astype(np.float32, copy=False) * vec_closeness(y, 0.0, 0.16) * vec_closeness(z, 0.0, 0.16)
        return np.zeros((len(labels),), dtype=np.float32)

    role_bias = {
        "body": 16000.0,
        "bandana": 15000.0,
        "hat": 13000.0,
        "beak": 17000.0,
    }
    best_scores = np.sum((face_colors_f - palette_f[labels]) ** 2, axis=1).astype(np.float32, copy=False)
    for role, anchor_label in anchor_labels.items():
        if anchor_label is None:
            continue
        candidate_label = int(anchor_label)
        zone_score = role_zone_score(role)
        if not np.any(zone_score >= 0.45):
            continue
        candidate_scores = np.sum((face_colors_f - palette_f[candidate_label]) ** 2, axis=1).astype(np.float32, copy=False)
        candidate_scores = candidate_scores - role_bias[role] * zone_score
        better = (zone_score >= 0.45) & (candidate_scores < best_scores)
        if np.any(better):
            labels[better] = candidate_label
            best_scores[better] = candidate_scores[better]
    return labels


def _select_palette_label(
    palette: np.ndarray,
    face_labels: np.ndarray,
    *,
    signal_name: str,
    min_signal: float = 0.0,
) -> int | None:
    if len(palette) == 0:
        return None
    counts = np.bincount(np.asarray(face_labels, dtype=np.int32), minlength=len(palette)).astype(np.float32)
    best_label: int | None = None
    best_score = float("-inf")
    for label, color in enumerate(np.asarray(palette, dtype=np.uint8)):
        signals = _duck_color_signals(color)
        signal = float(signals.get(signal_name, 0.0))
        if signal < float(min_signal):
            continue
        score = signal + 0.12 * float(counts[label]) / float(max(counts.max(), 1.0))
        if score > best_score:
            best_score = score
            best_label = int(label)
    return best_label


def _select_zone_palette_label(
    palette: np.ndarray,
    face_labels: np.ndarray,
    zone_mask: np.ndarray,
    *,
    signal_name: str | None = None,
    min_signal: float = 0.0,
    exclude_labels: set[int] | None = None,
    prefer_dominance: float = 1.0,
) -> int | None:
    labels = np.asarray(face_labels, dtype=np.int32)
    zone = np.asarray(zone_mask, dtype=bool)
    if len(labels) == 0 or not np.any(zone):
        return None
    counts = np.bincount(labels[zone], minlength=len(palette)).astype(np.float32)
    if not np.any(counts):
        return None
    excluded = exclude_labels or set()
    best_label: int | None = None
    best_score = float("-inf")
    total = float(max(counts.sum(), 1.0))
    for label, color in enumerate(np.asarray(palette, dtype=np.uint8)):
        if label in excluded or counts[label] <= 0:
            continue
        signals = _duck_color_signals(color)
        signal = float(signals.get(signal_name, 0.0)) if signal_name else 0.0
        if signal_name and signal < float(min_signal):
            continue
        luminance = _relative_luminance(color)
        neutral = float(signals["neutral"])
        if neutral >= 0.30 and luminance >= 0.62:
            continue
        score = prefer_dominance * float(counts[label]) / total + signal
        if best_score < score:
            best_score = score
            best_label = int(label)
    return best_label


def _apply_duck_color_intent_rules(
    *,
    face_labels: np.ndarray,
    palette: np.ndarray,
    positions: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clean up common duck-print intent mistakes after region transfer.

    The rules are conservative: large misplaced neutral or warm patches are
    rewritten, while small emblems, stars, eyes, and shield details survive.
    """
    labels = np.asarray(face_labels, dtype=np.int32).copy()
    palette_array = np.asarray(palette, dtype=np.uint8)
    face_array = np.asarray(faces, dtype=np.int64)
    if len(labels) == 0 or len(face_array) == 0 or len(palette_array) == 0:
        return labels, {"policy": "duck_color_intent_v1", "applied": False, "reassigned_faces": 0}

    centroids = normalize_positions(face_centroids(positions, face_array))
    x = centroids[:, 0]
    y = centroids[:, 1]
    z = centroids[:, 2]
    beak_zone = (x > 0.29) & (y > 0.04) & (y < 0.30) & (np.abs(z) < 0.22)
    blue_body_zone = (
        ((y > -0.32) & (np.abs(x) < 0.34) & (np.abs(z) < 0.38))
        | ((y > -0.38) & (x < 0.12) & (np.abs(z) < 0.42))
    )
    head_cap_zone = (y > 0.08) & (x < 0.22) & (np.abs(z) < 0.36)
    face_zone = (x > 0.06) & (y > -0.06) & (np.abs(z) < 0.28)
    fallback_blue_label = _select_palette_label(palette_array, labels, signal_name="blue", min_signal=0.08)
    beak_label = _select_zone_palette_label(
        palette_array,
        labels,
        beak_zone,
        signal_name="orange",
        min_signal=0.08,
        prefer_dominance=1.8,
    )
    if beak_label is None:
        beak_label = _select_palette_label(palette_array, labels, signal_name="orange", min_signal=0.08)
    body_label = _select_zone_palette_label(
        palette_array,
        labels,
        blue_body_zone,
        exclude_labels=set() if beak_label is None else {int(beak_label)},
        prefer_dominance=1.0,
    )
    if body_label is None:
        body_label = fallback_blue_label
    if body_label is None:
        return labels, {"policy": "duck_color_intent_v1", "applied": False, "reason": "no body label", "reassigned_faces": 0}

    component_ids = build_connected_face_components(labels, face_array)
    reassigned_total = 0
    component_rewrites: list[dict[str, Any]] = []
    component_count = int(component_ids.max()) + 1 if len(component_ids) else 0
    for component_id in range(component_count):
        member_indexes = np.flatnonzero(component_ids == component_id)
        if len(member_indexes) == 0:
            continue
        label = int(labels[int(member_indexes[0])])
        if label == body_label:
            continue
        signals = _duck_color_signals(palette_array[label])
        member_count = int(len(member_indexes))
        share = float(member_count) / float(max(len(labels), 1))
        zone_blue_share = float(np.mean(blue_body_zone[member_indexes]))
        zone_head_share = float(np.mean(head_cap_zone[member_indexes]))
        zone_face_share = float(np.mean(face_zone[member_indexes]))
        zone_beak_share = float(np.mean(beak_zone[member_indexes]))
        luminance = _relative_luminance(palette_array[label])
        is_misplaced_beak_color = (
            float(signals["orange"]) >= 0.08
            and float(signals["warm"]) >= 0.20
            and float(signals["orange"]) >= float(signals["red"])
            and luminance < 0.68
        )
        is_neutral = float(signals["neutral"]) >= 0.30
        is_light_neutral_detail = is_neutral and luminance >= 0.62

        should_reassign = False
        reason = ""
        if is_misplaced_beak_color and zone_beak_share < 0.35 and zone_blue_share >= 0.30:
            should_reassign = True
            reason = "yellow_orange_color_outside_beak"
        elif is_neutral and not is_light_neutral_detail and zone_head_share >= 0.48 and (member_count >= 500 or share >= 0.006):
            should_reassign = True
            reason = "large_neutral_head_cap"
        elif is_neutral and not is_light_neutral_detail and zone_blue_share >= 0.70 and (member_count >= 900 or share >= 0.010):
            should_reassign = True
            reason = "large_neutral_body_patch"
        elif float(signals["red"]) >= 0.16 and zone_head_share >= 0.28 and member_count <= 800:
            should_reassign = True
            reason = "small_red_head_mark"
        elif (
            int(label) != int(beak_label)
            and float(signals["red"]) >= 0.16
            and zone_face_share >= 0.80
            and zone_beak_share < 0.35
            and member_count <= 240
        ):
            should_reassign = True
            reason = "small_red_face_mark"

        if should_reassign:
            labels[member_indexes] = int(body_label)
            reassigned_total += member_count
            component_rewrites.append(
                {
                    "component_id": int(component_id),
                    "from_label": label,
                    "to_label": int(body_label),
                    "face_count": member_count,
                    "reason": reason,
                    "zone_blue_share": round(zone_blue_share, 4),
                    "zone_head_share": round(zone_head_share, 4),
                    "zone_face_share": round(zone_face_share, 4),
                    "zone_beak_share": round(zone_beak_share, 4),
                    "luminance": round(float(luminance), 4),
                }
            )

    return labels, {
        "policy": "duck_color_intent_v1",
        "applied": bool(reassigned_total),
        "body_label": int(body_label),
        "fallback_blue_label": None if fallback_blue_label is None else int(fallback_blue_label),
        "beak_label": None if beak_label is None else int(beak_label),
        "reassigned_faces": int(reassigned_total),
        "component_rewrites": component_rewrites,
    }


def _duck_role_zone_scores(centroids: np.ndarray) -> dict[str, np.ndarray]:
    points = np.asarray(centroids, dtype=np.float32)
    if len(points) == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return {"body": empty, "bandana": empty, "hat": empty, "beak": empty}
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    def vec_closeness(values: np.ndarray, center: float, radius: float) -> np.ndarray:
        if radius <= 1e-6:
            return np.zeros_like(values, dtype=np.float32)
        return np.maximum(0.0, 1.0 - np.abs(values - center) / radius).astype(np.float32, copy=False)

    return {
        "body": vec_closeness(x, 0.03, 0.40) * vec_closeness(y, -0.08, 0.34) * vec_closeness(z, 0.0, 0.30),
        "bandana": vec_closeness(x, 0.14, 0.22) * vec_closeness(y, -0.04, 0.20) * vec_closeness(z, 0.0, 0.24),
        "hat": np.maximum(0.0, (y - 0.08) / 0.32).astype(np.float32, copy=False) * vec_closeness(x, 0.02, 0.46) * vec_closeness(z, 0.0, 0.36),
        "beak": np.maximum(0.0, (x - 0.16) / 0.22).astype(np.float32, copy=False) * vec_closeness(y, 0.0, 0.18) * vec_closeness(z, 0.0, 0.18),
    }


def _build_duck_semantic_parts(
    *,
    face_labels: np.ndarray,
    palette: np.ndarray,
    face_colors: np.ndarray,
    positions: np.ndarray,
    faces: np.ndarray,
    anchor_labels: dict[str, int | None],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    labels = np.asarray(face_labels, dtype=np.int32)
    colors = np.asarray(face_colors, dtype=np.uint8)
    centroids = normalize_positions(face_centroids(positions, faces))
    part_ids = {"body": 0, "bandana": 1, "hat": 2, "beak": 3, "accent": 4}
    if len(labels) == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 3), dtype=np.uint8), part_ids

    zone_scores = _duck_role_zone_scores(centroids)
    semantic_labels = np.full((len(labels),), part_ids["accent"], dtype=np.int32)
    best_scores = np.full((len(labels),), 0.18, dtype=np.float32)

    for role in ("body", "bandana", "hat", "beak"):
        score = zone_scores[role].copy()
        anchor_label = anchor_labels.get(role)
        if anchor_label is not None:
            score += 0.55 * (labels == int(anchor_label)).astype(np.float32)
        better = score > best_scores
        semantic_labels[better] = int(part_ids[role])
        best_scores[better] = score[better]

    part_palette = np.clip(
        np.rint(
            average_by_cluster(
                colors.astype(np.float32),
                semantic_labels,
                len(part_ids),
            )
        ),
        0,
        255,
    ).astype(np.uint8)
    return semantic_labels, part_palette, part_ids


def _build_duck_seeded_parts(
    *,
    face_colors: np.ndarray,
    positions: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    colors = np.asarray(face_colors, dtype=np.uint8)
    centroids = normalize_positions(face_centroids(positions, faces))
    part_ids = {
        "body": 0,
        "bandana": 1,
        "hat_brim": 2,
        "hat_crown": 3,
        "beak": 4,
        "boots": 5,
        "accent": 6,
    }
    if len(colors) == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 3), dtype=np.uint8), part_ids

    x = centroids[:, 0]
    y = centroids[:, 1]
    z = centroids[:, 2]

    def vec_closeness(values: np.ndarray, center: float, radius: float) -> np.ndarray:
        if radius <= 1e-6:
            return np.zeros_like(values, dtype=np.float32)
        return np.maximum(0.0, 1.0 - np.abs(values - center) / radius).astype(np.float32, copy=False)

    zone_scores = {
        "body": vec_closeness(x, 0.03, 0.42) * vec_closeness(y, -0.06, 0.36) * vec_closeness(z, 0.0, 0.32),
        "bandana": vec_closeness(x, 0.15, 0.24) * vec_closeness(y, -0.02, 0.18) * vec_closeness(z, 0.0, 0.24),
        "hat_brim": vec_closeness(y, 0.14, 0.12) * vec_closeness(x, 0.02, 0.46) * vec_closeness(z, 0.0, 0.38),
        "hat_crown": np.maximum(0.0, (y - 0.16) / 0.22).astype(np.float32, copy=False) * vec_closeness(x, 0.02, 0.34) * vec_closeness(z, 0.0, 0.28),
        "beak": np.maximum(0.0, (x - 0.18) / 0.20).astype(np.float32, copy=False) * vec_closeness(y, 0.0, 0.18) * vec_closeness(z, 0.0, 0.18),
        "boots": np.maximum(0.0, (-y - 0.12) / 0.18).astype(np.float32, copy=False) * vec_closeness(x, 0.0, 0.34) * vec_closeness(z, 0.0, 0.28),
    }

    semantic_labels = np.full((len(colors),), part_ids["accent"], dtype=np.int32)
    best_scores = np.full((len(colors),), 0.18, dtype=np.float32)
    for role in ("hat_crown", "hat_brim", "beak", "bandana", "boots", "body"):
        score = zone_scores[role]
        better = score > best_scores
        semantic_labels[better] = int(part_ids[role])
        best_scores[better] = score[better]

    part_palette = np.clip(
        np.rint(
            average_by_cluster(
                colors.astype(np.float32),
                semantic_labels,
                len(part_ids),
            )
        ),
        0,
        255,
    ).astype(np.uint8)
    return semantic_labels, part_palette, part_ids


def _convert_loaded_mesh_legacy_fast_face_labels(
    loaded: LoadedTexturedMesh,
    *,
    out_dir: str | Path | None = None,
    n_regions: int = 5,
    object_name: str | None = None,
    obj_filename: str = "region_materials.obj",
    threemf_filename: str = "region_colorgroup.3mf",
    preview_filename: str = "region_preview.png",
    swatch_filename: str = "palette_swatches.png",
    palette_csv_filename: str = "palette.csv",
    report_filename: str = "conversion_report.json",
) -> dict[str, Any]:
    started = perf_counter()
    output_dir = Path(out_dir).expanduser().resolve() if out_dir else loaded.source_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    posterized_texture = _legacy_posterize_texture(loaded.texture_rgb, image_palette=n_regions)
    vertex_colors = _legacy_sample_vertex_colors(posterized_texture, loaded.texcoords)
    palette, vertex_labels = _legacy_quantize_vertex_colors(vertex_colors, num_colors=n_regions)
    face_labels = _legacy_assign_face_labels(loaded.faces, vertex_labels)
    return _write_asset_bundle(
        loaded=LoadedTexturedMesh(
            mesh=loaded.mesh,
            positions=loaded.positions,
            faces=loaded.faces,
            texcoords=loaded.texcoords,
            texture_rgb=posterized_texture,
            source_path=loaded.source_path,
            texture_path=loaded.texture_path,
            source_format=loaded.source_format,
            normal_texture_rgb=loaded.normal_texture_rgb,
            orm_texture_rgb=loaded.orm_texture_rgb,
            base_color_factor=loaded.base_color_factor,
            metallic_factor=loaded.metallic_factor,
            roughness_factor=loaded.roughness_factor,
            normal_scale=loaded.normal_scale,
        ),
        face_labels=face_labels,
        palette=palette,
        output_dir=output_dir,
        object_name=object_name,
        obj_filename=obj_filename,
        threemf_filename=threemf_filename,
        preview_filename=preview_filename,
        swatch_filename=swatch_filename,
        palette_csv_filename=palette_csv_filename,
        report_filename=report_filename,
        started=started,
        strategy="legacy_fast_face_labels",
        notes=[
            "This conversion ports the old 3dcolor fast path: posterize texture, sample UV vertex colors, quantize, then assign face labels by vertex-majority.",
            "It is intended as a regression-compatible Bambu OBJ path based on the historically working Grinch outputs.",
            "3MF export still uses standards-based colorgroup face colors.",
        ],
        extra_report={
            "image_palette": int(n_regions),
            "quantized_color_count": int(len(palette)),
            "legacy_fast_path": True,
        },
    )


def _convert_loaded_mesh_blender_like_bake_face_labels(
    loaded: LoadedTexturedMesh,
    *,
    out_dir: str | Path | None = None,
    n_regions: int = 5,
    object_name: str | None = None,
    obj_filename: str = "region_materials.obj",
    threemf_filename: str = "region_colorgroup.3mf",
    preview_filename: str = "region_preview.png",
    swatch_filename: str = "palette_swatches.png",
    palette_csv_filename: str = "palette.csv",
    report_filename: str = "conversion_report.json",
) -> dict[str, Any]:
    started = perf_counter()
    output_dir = Path(out_dir).expanduser().resolve() if out_dir else loaded.source_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    baked_vertex_colors, bake_metadata = bake_texture_to_vertex_colors(
        loaded.texture_rgb,
        loaded.texcoords,
        loaded.faces,
        pad_pixels=max(2, min(8, int(n_regions))),
        sampling_mode="bilinear",
    )
    palette, vertex_labels = _legacy_quantize_vertex_colors(baked_vertex_colors, num_colors=n_regions)
    face_labels = _legacy_assign_face_labels(loaded.faces, vertex_labels)
    return _write_asset_bundle(
        loaded=loaded,
        face_labels=face_labels,
        palette=palette,
        output_dir=output_dir,
        object_name=object_name,
        obj_filename=obj_filename,
        threemf_filename=threemf_filename,
        preview_filename=preview_filename,
        swatch_filename=swatch_filename,
        palette_csv_filename=palette_csv_filename,
        report_filename=report_filename,
        started=started,
        strategy="blender_like_bake_face_labels",
        notes=[
            "This conversion follows a Blender-like bake flow: build a UV island mask, expand seams, bilinearly sample texture into baked vertex colors, then quantize and assign face labels by vertex-majority.",
            "It is meant to preserve cleaner baked color attributes before later printable-region cleanup.",
            "3MF export still uses standards-based colorgroup face colors.",
        ],
        extra_report={
            "bake_metadata": bake_metadata,
            "blender_like_bake": True,
            "quantized_color_count": int(len(palette)),
        },
    )


def _convert_loaded_mesh_legacy_corner_face_labels(
    loaded: LoadedTexturedMesh,
    *,
    out_dir: str | Path | None = None,
    n_regions: int = 5,
    object_name: str | None = None,
    obj_filename: str = "region_materials.obj",
    threemf_filename: str = "region_colorgroup.3mf",
    preview_filename: str = "region_preview.png",
    swatch_filename: str = "palette_swatches.png",
    palette_csv_filename: str = "palette.csv",
    report_filename: str = "conversion_report.json",
) -> dict[str, Any]:
    started = perf_counter()
    output_dir = Path(out_dir).expanduser().resolve() if out_dir else loaded.source_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    posterized_texture = _legacy_posterize_texture(loaded.texture_rgb, image_palette=n_regions)
    corner_colors, corner_metadata = bake_texture_to_corner_colors(
        posterized_texture,
        loaded.texcoords,
        loaded.faces,
        pad_pixels=max(2, min(8, int(n_regions))),
        sampling_mode="nearest",
    )
    face_colors = face_colors_from_corner_colors(corner_colors)
    palette, face_labels = _quantize_face_colors(face_colors, loaded.positions, loaded.faces, n_regions)
    return _write_asset_bundle(
        loaded=LoadedTexturedMesh(
            mesh=loaded.mesh,
            positions=loaded.positions,
            faces=loaded.faces,
            texcoords=loaded.texcoords,
            texture_rgb=posterized_texture,
            source_path=loaded.source_path,
            texture_path=loaded.texture_path,
            source_format=loaded.source_format,
            normal_texture_rgb=loaded.normal_texture_rgb,
            orm_texture_rgb=loaded.orm_texture_rgb,
            base_color_factor=loaded.base_color_factor,
            metallic_factor=loaded.metallic_factor,
            roughness_factor=loaded.roughness_factor,
            normal_scale=loaded.normal_scale,
        ),
        face_labels=face_labels,
        palette=palette,
        output_dir=output_dir,
        object_name=object_name,
        obj_filename=obj_filename,
        threemf_filename=threemf_filename,
        preview_filename=preview_filename,
        swatch_filename=swatch_filename,
        palette_csv_filename=palette_csv_filename,
        report_filename=report_filename,
        started=started,
        strategy="legacy_corner_face_labels",
        notes=[
            "This conversion posterizes the source texture, samples seam-preserving corner colors, then quantizes face colors directly instead of voting through shared vertices.",
            "It is intended to preserve UV seam boundaries that get blurred by shared-vertex label collapse.",
            "3MF export still uses standards-based colorgroup face colors.",
        ],
        extra_report={
            "legacy_corner_path": True,
            "corner_bake_metadata": corner_metadata,
            "quantized_color_count": int(len(palette)),
        },
    )


def _convert_loaded_mesh_blender_cleanup_face_labels(
    loaded: LoadedTexturedMesh,
    *,
    out_dir: str | Path | None = None,
    n_regions: int = 5,
    object_name: str | None = None,
    obj_filename: str = "region_materials.obj",
    threemf_filename: str = "region_colorgroup.3mf",
    preview_filename: str = "region_preview.png",
    swatch_filename: str = "palette_swatches.png",
    palette_csv_filename: str = "palette.csv",
    report_filename: str = "conversion_report.json",
) -> dict[str, Any]:
    started = perf_counter()
    output_dir = Path(out_dir).expanduser().resolve() if out_dir else loaded.source_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    corner_colors, corner_metadata = bake_texture_to_corner_colors(
        loaded.texture_rgb,
        loaded.texcoords,
        loaded.faces,
        pad_pixels=max(2, min(8, int(n_regions))),
        sampling_mode="bilinear",
    )
    face_colors = face_colors_from_corner_colors(corner_colors)
    cleaned_face_colors = _apply_same_mesh_blender_cleanup(face_colors, n_regions=n_regions)
    palette, face_labels = _quantize_face_colors(cleaned_face_colors, loaded.positions, loaded.faces, n_regions)
    face_labels = merge_small_palette_islands(
        face_labels,
        cleaned_face_colors,
        palette,
        loaded.faces,
        min_component_size=max(6, min(48, len(loaded.faces) // 120 or 6)),
    )
    face_labels = smooth_face_palette_indices(
        face_labels,
        cleaned_face_colors,
        palette,
        loaded.faces,
        iterations=2,
    )
    palette, face_labels = compact_palette(palette, face_labels)
    return _write_asset_bundle(
        loaded=loaded,
        face_labels=face_labels,
        palette=palette,
        output_dir=output_dir,
        object_name=object_name,
        obj_filename=obj_filename,
        threemf_filename=threemf_filename,
        preview_filename=preview_filename,
        swatch_filename=swatch_filename,
        palette_csv_filename=palette_csv_filename,
        report_filename=report_filename,
        started=started,
        strategy="blender_cleanup_face_labels",
        notes=[
            "This no-repair conversion mirrors the useful Blender addon behaviors we inspected: seam-aware corner-color bake, HSV/contrast cleanup, posterize-style compression, and island cleanup.",
            "It is meant to emulate a manual Blender vertex-color cleanup flow in pure Python while staying on the original mesh topology.",
            "3MF export still uses standards-based colorgroup face colors.",
        ],
        extra_report={
            "blender_cleanup_path": True,
            "corner_bake_metadata": corner_metadata,
            "cleanup_profile": {
                "contrast_gain": 1.18,
                "brightness_offset": 0.015,
                "saturation_gain": 1.18,
                "value_gain": 1.05,
                "posterize_levels": max(3, min(8, int(n_regions))),
            },
            "quantized_color_count": int(len(palette)),
        },
    )


def _convert_loaded_mesh_hue_vcm_cleanup_face_labels(
    loaded: LoadedTexturedMesh,
    *,
    out_dir: str | Path | None = None,
    n_regions: int = 5,
    object_name: str | None = None,
    obj_filename: str = "region_materials.obj",
    threemf_filename: str = "region_colorgroup.3mf",
    preview_filename: str = "region_preview.png",
    swatch_filename: str = "palette_swatches.png",
    palette_csv_filename: str = "palette.csv",
    report_filename: str = "conversion_report.json",
) -> dict[str, Any]:
    started = perf_counter()
    output_dir = Path(out_dir).expanduser().resolve() if out_dir else loaded.source_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    corner_colors, corner_metadata = bake_texture_to_corner_colors(
        loaded.texture_rgb,
        loaded.texcoords,
        loaded.faces,
        pad_pixels=max(2, min(8, int(n_regions))),
        sampling_mode="bilinear",
    )
    face_colors = face_colors_from_corner_colors(corner_colors)
    cleaned_face_colors = _apply_same_mesh_hue_vcm_cleanup(face_colors, n_regions=n_regions)
    palette, face_labels = _quantize_face_colors(cleaned_face_colors, loaded.positions, loaded.faces, n_regions)
    face_labels = merge_small_palette_islands(
        face_labels,
        cleaned_face_colors,
        palette,
        loaded.faces,
        min_component_size=max(8, min(56, len(loaded.faces) // 96 or 8)),
    )
    face_labels = smooth_face_palette_indices(
        face_labels,
        cleaned_face_colors,
        palette,
        loaded.faces,
        iterations=3,
    )
    palette, face_labels = compact_palette(palette, face_labels)
    return _write_asset_bundle(
        loaded=loaded,
        face_labels=face_labels,
        palette=palette,
        output_dir=output_dir,
        object_name=object_name,
        obj_filename=obj_filename,
        threemf_filename=threemf_filename,
        preview_filename=preview_filename,
        swatch_filename=swatch_filename,
        palette_csv_filename=palette_csv_filename,
        report_filename=report_filename,
        started=started,
        strategy="hue_vcm_cleanup_face_labels",
        notes=[
            "This no-repair conversion ports addon-style cleanup into pure Python: HUE levels, brightness/contrast, hue-saturation, overlay/screen blending, and Vertex Color Master posterize/remap cleanup.",
            "It keeps seam-aware corner-color baking, then consolidates palette islands on the original mesh instead of transferring across repaired topology.",
            "3MF export still uses standards-based colorgroup face colors.",
        ],
        extra_report={
            "hue_vcm_cleanup_path": True,
            "corner_bake_metadata": corner_metadata,
            "cleanup_profile": {
                "levels_black": 0.06,
                "levels_white": 0.94,
                "gamma": 0.92,
                "brightness": 0.018,
                "contrast": 0.22,
                "saturation": 1.16,
                "value": 1.04,
                "posterize_levels": max(3, min(8, int(n_regions) + 1)),
            },
            "quantized_color_count": int(len(palette)),
        },
    )


def _write_export_preview(
    path: Path,
    positions: np.ndarray,
    faces: np.ndarray,
    palette: np.ndarray,
    face_palette_indices: np.ndarray,
) -> None:
    from PIL import Image, ImageDraw
    import math

    if len(positions) == 0 or len(faces) == 0:
        Image.new("RGB", (960, 720), (245, 241, 234)).save(path)
        return
    centered = np.asarray(positions, dtype=np.float32)
    bbox_min = centered.min(axis=0)
    bbox_max = centered.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extents = np.maximum(bbox_max - bbox_min, 1e-6)
    scale = float(np.max(extents))
    normalized = (centered - center) / scale
    angle_y = math.radians(-32.0)
    angle_x = math.radians(20.0)
    rot_y = np.array([[math.cos(angle_y), 0.0, math.sin(angle_y)], [0.0, 1.0, 0.0], [-math.sin(angle_y), 0.0, math.cos(angle_y)]], dtype=np.float32)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(angle_x), -math.sin(angle_x)], [0.0, math.sin(angle_x), math.cos(angle_x)]], dtype=np.float32)
    transformed = normalized @ rot_y.T
    transformed = transformed @ rot_x.T
    projected = transformed[:, :2]
    projected[:, 1] *= -1.0
    min_xy = projected.min(axis=0)
    max_xy = projected.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    width = 960
    height = 720
    margin = 72.0
    scale_2d = min((width - margin * 2.0) / float(span[0]), (height - margin * 2.0) / float(span[1]))
    projected = (projected - (min_xy + max_xy) / 2.0) * scale_2d
    projected[:, 0] += width / 2.0
    projected[:, 1] += height / 2.0
    face_points = projected[faces]
    face_depth = transformed[faces][:, :, 2].mean(axis=1)
    face_vertices = transformed[faces]
    face_normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
    normal_lengths = np.linalg.norm(face_normals, axis=1)
    valid_normals = normal_lengths > 1e-8
    if np.any(valid_normals):
        face_normals[valid_normals] /= normal_lengths[valid_normals][:, None]
    view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    light_dir = np.array([0.45, 0.55, 1.0], dtype=np.float32)
    light_dir /= np.linalg.norm(light_dir)
    facing = (face_normals @ view_dir) < 0.0
    adjusted_normals = face_normals.copy()
    adjusted_normals[facing] *= -1.0
    lighting = np.clip(adjusted_normals @ light_dir, 0.0, 1.0)
    lighting = 0.42 + 0.58 * lighting
    draw_order = np.argsort(face_depth)
    image = Image.new("RGB", (width, height), (244, 240, 233))
    draw = ImageDraw.Draw(image, "RGBA")
    for face_index in draw_order.tolist():
        polygon = face_points[face_index]
        color = np.asarray(palette[int(face_palette_indices[face_index])], dtype=np.float32)
        lit = np.clip(np.rint(color * float(lighting[face_index])), 0, 255).astype(np.uint8)
        outline = tuple(int(channel) for channel in np.clip(lit * 0.72, 0, 255).tolist()) + (220,)
        fill = tuple(int(channel) for channel in lit.tolist()) + (255,)
        draw.polygon([tuple(point.tolist()) for point in polygon], fill=fill, outline=outline)
    image.save(path)


def _write_palette_swatches(path: Path, palette: np.ndarray, face_labels: np.ndarray) -> None:
    from PIL import Image, ImageDraw
    import math

    columns = 4
    swatch_w = 160
    swatch_h = 92
    rows = max(1, math.ceil(len(palette) / columns))
    image = Image.new("RGB", (columns * swatch_w, rows * swatch_h), (248, 244, 236))
    draw = ImageDraw.Draw(image)
    counts = np.bincount(face_labels, minlength=len(palette))
    for index, color in enumerate(np.asarray(palette, dtype=np.uint8)):
        col = index % columns
        row = index // columns
        x0 = col * swatch_w
        y0 = row * swatch_h
        x1 = x0 + swatch_w - 8
        y1 = y0 + swatch_h - 8
        fill = tuple(int(channel) for channel in color.tolist())
        draw.rounded_rectangle((x0 + 8, y0 + 8, x1, y1), radius=16, fill=fill, outline=(24, 24, 24), width=2)
        draw.text((x0 + 14, y0 + 16), f"{index + 1}: #{color[0]:02X}{color[1]:02X}{color[2]:02X}", fill=(16, 16, 16))
        draw.text((x0 + 14, y0 + 44), f"Faces: {int(counts[index])}", fill=(16, 16, 16))
    image.save(path)


def _palette_rows(palette: np.ndarray, face_labels: np.ndarray) -> list[dict[str, Any]]:
    counts = np.bincount(face_labels, minlength=len(palette))
    rows = []
    for index, color in enumerate(np.asarray(palette, dtype=np.uint8)):
        rows.append(
            {
                "palette_index": int(index),
                "hex": f"#{int(color[0]):02X}{int(color[1]):02X}{int(color[2]):02X}",
                "rgb": [int(color[0]), int(color[1]), int(color[2])],
                "face_count": int(counts[index]),
            }
        )
    rows.sort(key=lambda item: (-item["face_count"], item["palette_index"]))
    return rows


def _component_metrics(face_labels: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(face_labels, dtype=np.int32)
    face_array = np.asarray(faces, dtype=np.int64)
    face_count = int(len(labels))
    if face_count == 0 or len(face_array) == 0:
        return {
            "component_count": 0,
            "tiny_island_count": 0,
            "largest_component_share": 0.0,
        }
    component_ids = build_connected_face_components(labels, face_array)
    if len(component_ids) == 0:
        return {
            "component_count": 0,
            "tiny_island_count": 0,
            "largest_component_share": 0.0,
        }
    component_sizes = np.bincount(component_ids)
    tiny_threshold = max(4, min(64, max(1, face_count // 500)))
    tiny_island_count = int(np.sum(component_sizes < tiny_threshold))
    largest_component_share = round(float(component_sizes.max()) / float(face_count), 4) if len(component_sizes) else 0.0
    return {
        "component_count": int(len(component_sizes)),
        "tiny_island_count": tiny_island_count,
        "largest_component_share": largest_component_share,
    }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mesh_geometry_stats(positions: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(positions, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    face_count = int(len(np.asarray(faces)))
    return {
        "vertex_count": int(len(np.asarray(positions))),
        "face_count": face_count,
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "body_count": None if face_count > 200_000 else int(len(mesh.split(only_watertight=False))),
    }


def _simplify_loaded_geometry(
    loaded: LoadedTexturedMesh,
    *,
    target_face_count: int | None,
) -> tuple[LoadedTexturedMesh, dict[str, Any]]:
    face_array = np.asarray(loaded.faces, dtype=np.int64)
    if target_face_count is None or len(face_array) <= int(target_face_count):
        return loaded, {
            "simplification_applied": False,
            "target_face_count": None if target_face_count is None else int(target_face_count),
            "source_face_count": int(len(face_array)),
            "result_face_count": int(len(face_array)),
        }

    mesh = trimesh.Trimesh(
        vertices=np.asarray(loaded.positions, dtype=np.float32),
        faces=face_array,
        process=False,
    )
    simplified = mesh.simplify_quadric_decimation(face_count=int(target_face_count))
    simplified_positions = np.asarray(simplified.vertices, dtype=np.float32)
    simplified_faces = np.asarray(simplified.faces, dtype=np.int64)
    simplified_loaded = LoadedTexturedMesh(
        mesh=simplified,
        positions=simplified_positions,
        faces=simplified_faces,
        texcoords=np.zeros((len(simplified_positions), 2), dtype=np.float32),
        texture_rgb=np.asarray(loaded.texture_rgb, dtype=np.uint8),
        source_path=loaded.source_path,
        texture_path=loaded.texture_path,
        source_format=loaded.source_format,
        normal_texture_rgb=loaded.normal_texture_rgb,
        orm_texture_rgb=loaded.orm_texture_rgb,
        base_color_factor=loaded.base_color_factor,
        metallic_factor=loaded.metallic_factor,
        roughness_factor=loaded.roughness_factor,
        normal_scale=loaded.normal_scale,
    )
    return simplified_loaded, {
        "simplification_applied": True,
        "target_face_count": int(target_face_count),
        "source_face_count": int(len(face_array)),
        "result_face_count": int(len(simplified_faces)),
    }


def _target_anchor_face_colors(target_loaded: LoadedTexturedMesh, palette: np.ndarray, face_labels: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    diagnostics = _texture_diagnostics(target_loaded.texture_rgb)
    if diagnostics.get("texture_role") == "suspected_normal_map":
        return np.asarray(palette, dtype=np.uint8)[np.asarray(face_labels, dtype=np.int32)], diagnostics
    return _sample_face_texture_colors(target_loaded), diagnostics


def assess_repaired_transfer_candidate(
    report: dict[str, Any],
    *,
    max_auto_faces: int = 250_000,
    max_components_per_palette_color: int = 48,
    max_tiny_islands: int = 96,
    min_largest_component_share: float = 0.08,
) -> dict[str, Any]:
    """Decide whether a repaired-geometry transfer is safe to auto-prefer."""
    face_count = _safe_int(report.get("face_count"))
    palette_size = max(_safe_int(report.get("palette_size") or report.get("region_count"), 1), 1)
    component_count = _safe_int(report.get("component_count"))
    tiny_island_count = _safe_int(report.get("tiny_island_count"))
    largest_component_share = _safe_float(report.get("largest_component_share"))

    max_component_count = max(palette_size * int(max_components_per_palette_color), palette_size + 8)
    reasons: list[str] = []
    if face_count > int(max_auto_faces):
        reasons.append(
            f"target face count {face_count:,} is above the auto-use threshold {int(max_auto_faces):,}"
        )
    if component_count > max_component_count:
        reasons.append(
            f"connected region count {component_count:,} is high for a {palette_size}-color palette"
        )
    if tiny_island_count > int(max_tiny_islands):
        reasons.append(
            f"tiny island count {tiny_island_count:,} is above the auto-use threshold {int(max_tiny_islands):,}"
        )
    if face_count > 0 and largest_component_share < float(min_largest_component_share):
        reasons.append(
            f"largest connected region share {largest_component_share:.3f} is below {float(min_largest_component_share):.3f}"
        )
    geometry_stats = report.get("target_geometry_stats") or {}
    if geometry_stats:
        if geometry_stats.get("is_watertight") is not True:
            reasons.append("target geometry is not watertight")
        body_count = geometry_stats.get("body_count")
        if body_count is None:
            reasons.append("target geometry body count was not measured")
        elif int(body_count) != 1:
            reasons.append(f"target geometry has {int(body_count)} bodies instead of 1")

    ready_for_auto = not reasons
    return {
        "status": "ready_for_auto" if ready_for_auto else "needs_review",
        "ready_for_auto": ready_for_auto,
        "reasons": reasons,
        "recommendation": (
            "Eligible to compare against same-mesh export in the lane chooser."
            if ready_for_auto
            else "Do not auto-prefer this repaired transfer yet; compare same-mesh/provider-baked output or regenerate cleaner source art."
        ),
        "thresholds": {
            "max_auto_faces": int(max_auto_faces),
            "max_components_per_palette_color": int(max_components_per_palette_color),
            "max_tiny_islands": int(max_tiny_islands),
            "min_largest_component_share": float(min_largest_component_share),
        },
    }


def _texture_diagnostics(texture_rgb: np.ndarray) -> dict[str, Any]:
    texture = np.asarray(texture_rgb, dtype=np.uint8)
    if texture.size == 0:
        return {
            "texture_role": "empty",
            "mean_rgb": [0.0, 0.0, 0.0],
            "std_rgb": [0.0, 0.0, 0.0],
            "min_rgb": [0, 0, 0],
            "max_rgb": [0, 0, 0],
        }
    flat = texture.reshape((-1, 3)).astype(np.float32)
    mean_rgb = flat.mean(axis=0)
    std_rgb = flat.std(axis=0)
    min_rgb = flat.min(axis=0).astype(np.uint8)
    max_rgb = flat.max(axis=0).astype(np.uint8)
    blue_bias = float(mean_rgb[2] - max(mean_rgb[0], mean_rgb[1]))
    red_green_gap = float(abs(mean_rgb[0] - mean_rgb[1]))
    smooth_normal_signature = (
        float(mean_rgb[2]) >= 170.0
        and blue_bias >= 35.0
        and red_green_gap <= 35.0
        and float(std_rgb[2]) <= 45.0
    )
    high_blue_channel_signature = (
        float(mean_rgb[2]) >= 150.0
        and blue_bias >= 55.0
        and red_green_gap <= 20.0
    )
    likely_normal_map = smooth_normal_signature or high_blue_channel_signature
    return {
        "texture_role": "suspected_normal_map" if likely_normal_map else "base_color_candidate",
        "mean_rgb": [round(float(value), 3) for value in mean_rgb.tolist()],
        "std_rgb": [round(float(value), 3) for value in std_rgb.tolist()],
        "min_rgb": [int(value) for value in min_rgb.tolist()],
        "max_rgb": [int(value) for value in max_rgb.tolist()],
        "blue_bias": round(blue_bias, 3),
    }


def assess_provider_bake_candidate(report: dict[str, Any], texture_diagnostics: dict[str, Any]) -> dict[str, Any]:
    assessment = assess_repaired_transfer_candidate(
        report,
        max_auto_faces=400_000,
        max_tiny_islands=128,
    )
    reasons = list(assessment.get("reasons") or [])
    if texture_diagnostics.get("texture_role") == "suspected_normal_map":
        reasons.append("provider baked base-color texture looks like a normal map, not printable color art")
    if reasons:
        assessment = {
            **assessment,
            "status": "needs_review",
            "ready_for_auto": False,
            "reasons": reasons,
            "recommendation": "Do not auto-prefer this provider-bake lane; use same-mesh/local transfer or request a fresh provider texture bake.",
        }
    return assessment


def _provider_repair_metadata(repair_result_path: str | Path | None) -> dict[str, Any]:
    if not repair_result_path:
        return {}
    path = Path(repair_result_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results") or []
    first_result = results[0] if results else {}
    metadata = first_result.get("metadata") or {}
    return {
        "provider_repair_result_path": str(path),
        "provider_repair_mode": payload.get("mode"),
        "provider_repair_status": payload.get("status"),
        "provider_repair_task_id": payload.get("task_id"),
        "provider_repair_metadata": metadata,
        "provider_repair_texture": metadata.get("texture") or {},
        "provider_repair_output": metadata.get("output") or {},
    }


def _write_palette_csv(path: Path, palette_rows: list[dict[str, Any]], total_faces: int) -> None:
    lines = ["palette_index,hex,r,g,b,face_count,face_share_percent"]
    denominator = max(int(total_faces), 1)
    for row in palette_rows:
        rgb = row.get("rgb") or [0, 0, 0]
        face_count = int(row.get("face_count") or 0)
        share = round((face_count / denominator) * 100.0, 2)
        lines.append(f"{int(row.get('palette_index') or 0)},{row.get('hex') or '#000000'},{int(rgb[0])},{int(rgb[1])},{int(rgb[2])},{face_count},{share}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_asset_bundle(
    *,
    loaded: LoadedTexturedMesh,
    face_labels: np.ndarray,
    palette: np.ndarray,
    output_dir: Path,
    object_name: str | None,
    obj_filename: str,
    threemf_filename: str,
    preview_filename: str,
    swatch_filename: str,
    palette_csv_filename: str,
    report_filename: str,
    started: float,
    strategy: str,
    notes: list[str],
    extra_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    obj_output_path, mtl_output_path = write_bambu_compatible_grouped_obj_with_mtl(
        output_dir / obj_filename,
        loaded.positions,
        loaded.faces,
        face_labels,
        palette,
    )
    vertex_color_obj_path = write_obj_with_vertex_colors(
        output_dir / f"{Path(obj_filename).stem}_vertex_color.obj",
        loaded.positions,
        loaded.faces,
        np.asarray(palette, dtype=np.float32)[np.asarray(face_labels, dtype=np.int32)] / 255.0,
        texcoords=loaded.texcoords,
        object_name=object_name or loaded.source_path.stem,
    )
    threemf_path = write_colorgroup_3mf(
        output_dir / threemf_filename,
        loaded.positions,
        loaded.faces,
        palette,
        face_labels,
        object_name=object_name or loaded.source_path.stem,
    )
    preview_path = output_dir / preview_filename
    swatch_path = output_dir / swatch_filename
    palette_csv_path = output_dir / palette_csv_filename
    face_label_path = output_dir / f"{Path(obj_filename).stem}_face_palette_indices.npy"
    palette_npy_path = output_dir / f"{Path(obj_filename).stem}_palette.npy"
    _write_export_preview(preview_path, loaded.positions, loaded.faces, palette, face_labels)
    _write_palette_swatches(swatch_path, palette, face_labels)
    palette_rows = _palette_rows(palette, face_labels)
    _write_palette_csv(palette_csv_path, palette_rows, len(loaded.faces))
    np.save(face_label_path, np.asarray(face_labels, dtype=np.int32))
    np.save(palette_npy_path, np.asarray(palette, dtype=np.uint8))
    component_metrics = _component_metrics(face_labels, loaded.faces)
    largest_palette_face_share = round(
        float(max((row["face_count"] for row in palette_rows), default=0)) / float(max(len(loaded.faces), 1)),
        4,
    )

    report = {
        "status": "ok",
        "source_path": str(loaded.source_path),
        "source_format": loaded.source_format,
        "source_texture_path": str(loaded.texture_path) if loaded.texture_path else None,
        "strategy": strategy,
        "region_count": int(len(palette)),
        "palette_size": int(len(palette)),
        "vertex_count": int(len(loaded.positions)),
        "face_count": int(len(loaded.faces)),
        "obj_path": str(obj_output_path),
        "vertex_color_obj_path": str(vertex_color_obj_path),
        "mtl_path": str(mtl_output_path),
        "obj_export_style": "bambu_simple",
        "threemf_path": str(threemf_path),
        "preview_path": str(preview_path),
        "palette_swatch_path": str(swatch_path),
        "palette_csv_path": str(palette_csv_path),
        "face_palette_indices_path": str(face_label_path),
        "palette_npy_path": str(palette_npy_path),
        "palette": palette_rows,
        "largest_palette_face_share": largest_palette_face_share,
        "duration_seconds": round(perf_counter() - started, 3),
        "notes": notes,
        **component_metrics,
    }
    if extra_report:
        report.update(extra_report)
    report_path = output_dir / report_filename
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def write_face_color_mesh_to_assets(
    *,
    positions: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
    source_path: str | Path,
    out_dir: str | Path,
    max_colors: int = 8,
    object_name: str | None = None,
    strategy: str = "face_color_palette",
    obj_filename: str = "region_materials.obj",
    threemf_filename: str = "region_colorgroup.3mf",
    preview_filename: str = "region_preview.png",
    swatch_filename: str = "palette_swatches.png",
    palette_csv_filename: str = "palette.csv",
    report_filename: str = "conversion_report.json",
    smooth_iterations: int = 2,
    min_component_size: int | None = None,
    cleanup_passes: int = 2,
    extra_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write Bambu-friendly grouped assets from per-face RGB colors."""
    started = perf_counter()
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pos = np.asarray(positions, dtype=np.float32)
    face_array = np.asarray(faces, dtype=np.int64)
    colors = np.asarray(face_colors, dtype=np.uint8)
    palette, face_labels = _quantize_face_colors(colors, pos, face_array, max_colors)
    minimum_component_size = (
        max(6, min(64, len(face_array) // 160 or 6))
        if min_component_size is None
        else int(min_component_size)
    )
    cleanup_pass_count = max(1, int(cleanup_passes))
    smooth_iteration_count = max(0, int(smooth_iterations))
    for pass_index in range(cleanup_pass_count):
        face_labels = merge_small_palette_islands(
            face_labels,
            colors,
            palette,
            face_array,
            min_component_size=minimum_component_size,
        )
        if smooth_iteration_count > 0:
            face_labels = smooth_face_palette_indices(
                face_labels,
                colors,
                palette,
                face_array,
                iterations=smooth_iteration_count if pass_index == 0 else 1,
            )
    face_labels = merge_small_palette_islands(
        face_labels,
        colors,
        palette,
        face_array,
        min_component_size=minimum_component_size,
    )
    palette, face_labels = compact_palette(palette, face_labels)
    if len(palette):
        palette = np.clip(
            np.rint(average_by_cluster(colors.astype(np.float32), face_labels, len(palette))),
            0,
            255,
        ).astype(np.uint8)
    loaded = LoadedTexturedMesh(
        mesh=None,
        positions=pos,
        faces=face_array,
        texcoords=np.zeros((len(pos), 2), dtype=np.float32),
        texture_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
        source_path=Path(source_path).expanduser().resolve(),
        texture_path=None,
        source_format="face_color_mesh",
    )
    report = _write_asset_bundle(
        loaded=loaded,
        face_labels=face_labels,
        palette=palette,
        output_dir=output_dir,
        object_name=object_name,
        obj_filename=obj_filename,
        threemf_filename=threemf_filename,
        preview_filename=preview_filename,
        swatch_filename=swatch_filename,
        palette_csv_filename=palette_csv_filename,
        report_filename=report_filename,
        started=started,
        strategy=strategy,
        notes=[
            "This conversion starts from repaired-geometry face colors, reduces them to a practical filament palette, and writes Bambu-friendly grouped OBJ/MTL assets.",
            "The vertex-color debug OBJ may preserve denser visual detail, but this grouped OBJ is the printable acceptance artifact.",
        ],
        extra_report={
            "max_colors": int(max_colors),
            "min_component_size": int(minimum_component_size),
            "smooth_iterations": int(smooth_iteration_count),
            "cleanup_passes": int(cleanup_pass_count),
            **(extra_report or {}),
        },
    )
    return report


def _sample_face_texture_colors(loaded: LoadedTexturedMesh) -> np.ndarray:
    if len(loaded.faces) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    vertex_colors = sample_texture(loaded.texture_rgb, loaded.texcoords)
    return np.clip(
        np.rint(vertex_colors[np.asarray(loaded.faces, dtype=np.int64)].mean(axis=1)),
        0,
        255,
    ).astype(np.uint8)


def _transfer_source_texture_regions_to_target(
    *,
    target_loaded: LoadedTexturedMesh,
    color_source_loaded: LoadedTexturedMesh,
    max_colors: int,
) -> dict[str, Any]:
    source_face_colors = _sample_face_texture_colors(color_source_loaded)
    target_face_count = int(len(target_loaded.faces))
    source_face_count = int(len(color_source_loaded.faces))
    if source_face_count == 0 or target_face_count == 0:
        raise ValueError("Both target and source meshes must contain faces for texture-region transfer.")

    region_budget = max(4, min(int(max_colors), 12))
    texture_regions, texture_palette = build_texture_regions(color_source_loaded.texture_rgb, n_regions=region_budget)
    cleaned_regions = clean_texture_regions(texture_regions, n_regions=len(texture_palette))
    source_face_region_labels = assign_faces_to_texture_regions(
        color_source_loaded.faces,
        color_source_loaded.texcoords,
        cleaned_regions,
    )
    source_region_count = int(source_face_region_labels.max()) + 1 if len(source_face_region_labels) else 0
    if source_region_count <= 0:
        raise ValueError("No source texture regions were assigned to source faces.")

    source_region_palette = np.clip(
        np.rint(
            average_by_cluster(
                source_face_colors.astype(np.float32),
                source_face_region_labels,
                source_region_count,
            )
        ),
        0,
        255,
    ).astype(np.uint8)
    transferred = transfer_face_region_ownership(
        source_positions=color_source_loaded.positions,
        source_faces=color_source_loaded.faces,
        source_face_labels=source_face_region_labels,
        target_positions=target_loaded.positions,
        target_faces=target_loaded.faces,
        neighbors=48,
        chunk_size=1024,
    )
    target_face_region_labels = np.asarray(transferred["target_face_labels"], dtype=np.int32)
    target_face_colors = source_region_palette[target_face_region_labels]
    min_component_size = max(48, min(256, target_face_count // 1800 or 48))
    target_face_region_labels = merge_small_palette_islands(
        target_face_region_labels,
        target_face_colors,
        source_region_palette,
        target_loaded.faces,
        min_component_size=min_component_size,
    )
    target_face_region_labels = smooth_face_palette_indices(
        target_face_region_labels,
        target_face_colors,
        source_region_palette,
        target_loaded.faces,
        iterations=3,
    )
    palette, face_palette_indices = compact_palette(source_region_palette, target_face_region_labels)
    return {
        "palette": np.asarray(palette, dtype=np.uint8),
        "face_palette_indices": np.asarray(face_palette_indices, dtype=np.int32),
        "raw_region_count": int(source_region_count),
        "effective_palette_size": int(len(palette)),
        "region_budget": int(region_budget),
        "source_face_region_labels": source_face_region_labels,
        "source_component_count": int(transferred["source_component_count"]),
    }


def convert_loaded_mesh_to_color_assets(
    loaded: LoadedTexturedMesh,
    *,
    out_dir: str | Path | None = None,
    n_regions: int = 5,
    strategy: str = "region_first",
    object_name: str | None = None,
    obj_filename: str = "region_materials.obj",
    threemf_filename: str = "region_colorgroup.3mf",
    preview_filename: str = "region_preview.png",
    swatch_filename: str = "palette_swatches.png",
    palette_csv_filename: str = "palette.csv",
    report_filename: str = "conversion_report.json",
) -> dict[str, Any]:
    if strategy in {"legacy_fast_face_labels", "legacy_face_regions"}:
        return _convert_loaded_mesh_legacy_fast_face_labels(
            loaded,
            out_dir=out_dir,
            n_regions=n_regions,
            object_name=object_name,
            obj_filename=obj_filename,
            threemf_filename=threemf_filename,
            preview_filename=preview_filename,
            swatch_filename=swatch_filename,
            palette_csv_filename=palette_csv_filename,
            report_filename=report_filename,
        )
    if strategy in {"legacy_corner_face_labels", "legacy_corner_face_regions"}:
        return _convert_loaded_mesh_legacy_corner_face_labels(
            loaded,
            out_dir=out_dir,
            n_regions=n_regions,
            object_name=object_name,
            obj_filename=obj_filename,
            threemf_filename=threemf_filename,
            preview_filename=preview_filename,
            swatch_filename=swatch_filename,
            palette_csv_filename=palette_csv_filename,
            report_filename=report_filename,
        )
    if strategy in {"blender_cleanup_face_labels", "blender_cleanup_face_regions"}:
        return _convert_loaded_mesh_blender_cleanup_face_labels(
            loaded,
            out_dir=out_dir,
            n_regions=n_regions,
            object_name=object_name,
            obj_filename=obj_filename,
            threemf_filename=threemf_filename,
            preview_filename=preview_filename,
            swatch_filename=swatch_filename,
            palette_csv_filename=palette_csv_filename,
            report_filename=report_filename,
        )
    if strategy in {"hue_vcm_cleanup_face_labels", "hue_vcm_cleanup_face_regions"}:
        return _convert_loaded_mesh_hue_vcm_cleanup_face_labels(
            loaded,
            out_dir=out_dir,
            n_regions=n_regions,
            object_name=object_name,
            obj_filename=obj_filename,
            threemf_filename=threemf_filename,
            preview_filename=preview_filename,
            swatch_filename=swatch_filename,
            palette_csv_filename=palette_csv_filename,
            report_filename=report_filename,
        )
    if strategy in {
        "blender_like_bake_face_labels",
        "blender_like_bake_face_regions",
        "geometry_transfer_blender_like_bake_face_regions",
    }:
        return _convert_loaded_mesh_blender_like_bake_face_labels(
            loaded,
            out_dir=out_dir,
            n_regions=n_regions,
            object_name=object_name,
            obj_filename=obj_filename,
            threemf_filename=threemf_filename,
            preview_filename=preview_filename,
            swatch_filename=swatch_filename,
            palette_csv_filename=palette_csv_filename,
            report_filename=report_filename,
        )
    started = perf_counter()
    region_labels, palette = build_texture_regions(loaded.texture_rgb, n_regions=n_regions)
    cleaned_labels = clean_texture_regions(region_labels, n_regions=len(palette))
    face_labels = assign_faces_to_texture_regions(loaded.faces, loaded.texcoords, cleaned_labels)

    output_dir = Path(out_dir).expanduser().resolve() if out_dir else loaded.source_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return _write_asset_bundle(
        loaded=loaded,
        face_labels=face_labels,
        palette=palette,
        output_dir=output_dir,
        object_name=object_name,
        obj_filename=obj_filename,
        threemf_filename=threemf_filename,
        preview_filename=preview_filename,
        swatch_filename=swatch_filename,
        palette_csv_filename=palette_csv_filename,
        report_filename=report_filename,
        started=started,
        strategy="region_first",
        notes=[
            "This conversion builds printable color regions from texture space before assigning colors to faces.",
            "3MF export uses standards-based colorgroup face colors.",
        ],
    )


def convert_face_colored_mesh_to_assets(
    loaded: LoadedTexturedMesh,
    *,
    face_colors: np.ndarray,
    out_dir: str | Path | None = None,
    max_colors: int = 12,
    object_name: str | None = None,
    obj_filename: str = "region_materials.obj",
    threemf_filename: str = "region_colorgroup.3mf",
    preview_filename: str = "region_preview.png",
    swatch_filename: str = "palette_swatches.png",
    palette_csv_filename: str = "palette.csv",
    report_filename: str = "conversion_report.json",
    strategy: str = "region_first_face_color",
    notes: list[str] | None = None,
    extra_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = perf_counter()
    output_dir = Path(out_dir).expanduser().resolve() if out_dir else loaded.source_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    palette_result = build_region_first_face_palette(
        positions=loaded.positions,
        faces=loaded.faces,
        face_colors=face_colors,
        max_colors=max_colors,
    )
    extra = dict(extra_report or {})
    extra.update(
        {
            "region_count": int(palette_result["region_count"]),
            "semantic_group_count": int(palette_result["semantic_group_count"]),
            "effective_max_colors": int(palette_result["effective_max_colors"]),
        }
    )
    return _write_asset_bundle(
        loaded=loaded,
        face_labels=np.asarray(palette_result["face_palette_indices"], dtype=np.int32),
        palette=np.asarray(palette_result["palette"], dtype=np.uint8),
        output_dir=output_dir,
        object_name=object_name,
        obj_filename=obj_filename,
        threemf_filename=threemf_filename,
        preview_filename=preview_filename,
        swatch_filename=swatch_filename,
        palette_csv_filename=palette_csv_filename,
        report_filename=report_filename,
        started=started,
        strategy=strategy,
        notes=notes
        or [
            "This conversion starts from face colors, groups contiguous regions, and then derives a printable palette.",
            "3MF export uses standards-based colorgroup face colors.",
        ],
        extra_report=extra,
    )


def convert_color_transferred_mesh_to_assets(
    *,
    target_loaded: LoadedTexturedMesh,
    color_source_loaded: LoadedTexturedMesh,
    out_dir: str | Path | None = None,
    max_colors: int = 12,
    strategy: str = "region_first",
    object_name: str | None = None,
    obj_filename: str = "region_materials.obj",
    threemf_filename: str = "region_colorgroup.3mf",
    preview_filename: str = "region_preview.png",
    swatch_filename: str = "palette_swatches.png",
    palette_csv_filename: str = "palette.csv",
    report_filename: str = "conversion_report.json",
) -> dict[str, Any]:
    started = perf_counter()
    output_dir = Path(out_dir).expanduser().resolve() if out_dir else target_loaded.source_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if strategy in {"legacy_corner_face_regions", "geometry_transfer_legacy_corner_face_regions"}:
        posterized_texture = _legacy_posterize_texture(color_source_loaded.texture_rgb, image_palette=max_colors)
        corner_colors, corner_metadata = bake_texture_to_corner_colors(
            posterized_texture,
            color_source_loaded.texcoords,
            color_source_loaded.faces,
            pad_pixels=max(2, min(8, int(max_colors))),
            sampling_mode="nearest",
        )
        source_face_colors = face_colors_from_corner_colors(corner_colors)
        palette, source_face_labels = _quantize_face_colors(
            source_face_colors,
            color_source_loaded.positions,
            color_source_loaded.faces,
            max_colors,
        )
        anchor_labels = _infer_duck_part_anchor_labels(
            face_labels=source_face_labels,
            palette=palette,
            positions=color_source_loaded.positions,
            faces=color_source_loaded.faces,
        )
        transferred = transfer_face_region_ownership(
            source_positions=color_source_loaded.positions,
            source_faces=color_source_loaded.faces,
            source_face_labels=source_face_labels,
            target_positions=target_loaded.positions,
            target_faces=target_loaded.faces,
            neighbors=56,
            chunk_size=1024,
            distance_power=1.15,
            normal_power=1.35,
        )
        face_labels = np.asarray(transferred["target_face_labels"], dtype=np.int32)
        target_face_texture_colors, target_texture_diagnostics = _target_anchor_face_colors(target_loaded, palette, face_labels)
        face_labels = _apply_duck_part_anchor_bias(
            face_labels=face_labels,
            palette=palette,
            face_colors=target_face_texture_colors,
            positions=target_loaded.positions,
            faces=target_loaded.faces,
            anchor_labels=anchor_labels,
        )
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
            face_colors,
            palette,
            target_loaded.faces,
            iterations=3,
        )
        palette, face_labels = compact_palette(palette, face_labels)
        return _write_asset_bundle(
            loaded=target_loaded,
            face_labels=face_labels,
            palette=palette,
            output_dir=output_dir,
            object_name=object_name,
            obj_filename=obj_filename,
            threemf_filename=threemf_filename,
            preview_filename=preview_filename,
            swatch_filename=swatch_filename,
            palette_csv_filename=palette_csv_filename,
            report_filename=report_filename,
            started=started,
            strategy="geometry_transfer_legacy_corner_face_regions",
            notes=[
                "This conversion derives source face regions from seam-preserving corner colors before transferring them to the repaired mesh.",
                "It is intended to keep UV seam detail intact instead of collapsing colors through shared vertex-majority labels first.",
                "3MF export still uses standards-based colorgroup face colors.",
            ],
            extra_report={
                "color_source_path": str(color_source_loaded.source_path),
                "color_source_format": color_source_loaded.source_format,
                "color_transfer_applied": True,
                "color_source_vertex_count": int(len(color_source_loaded.positions)),
                "target_vertex_count": int(len(target_loaded.positions)),
                "region_count": int(len(palette)),
                "semantic_group_count": int(len(palette)),
                "effective_max_colors": int(len(palette)),
                "legacy_corner_path": True,
                "region_transfer_mode": "corner_face_regions",
                "source_component_count": int(transferred["source_component_count"]),
                "duck_part_anchor_labels": anchor_labels,
                "target_texture_diagnostics": target_texture_diagnostics,
                "corner_bake_metadata": corner_metadata,
            },
        )
    if strategy in {"duck_semantic_parts", "geometry_transfer_duck_semantic_parts"}:
        posterized_texture = _legacy_posterize_texture(color_source_loaded.texture_rgb, image_palette=max_colors)
        corner_colors, corner_metadata = bake_texture_to_corner_colors(
            posterized_texture,
            color_source_loaded.texcoords,
            color_source_loaded.faces,
            pad_pixels=max(2, min(8, int(max_colors))),
            sampling_mode="nearest",
        )
        source_face_colors = face_colors_from_corner_colors(corner_colors)
        source_palette, source_face_labels = _quantize_face_colors(
            source_face_colors,
            color_source_loaded.positions,
            color_source_loaded.faces,
            max_colors,
        )
        anchor_labels = _infer_duck_part_anchor_labels(
            face_labels=source_face_labels,
            palette=source_palette,
            positions=color_source_loaded.positions,
            faces=color_source_loaded.faces,
        )
        source_part_labels, part_palette, part_ids = _build_duck_semantic_parts(
            face_labels=source_face_labels,
            palette=source_palette,
            face_colors=source_face_colors,
            positions=color_source_loaded.positions,
            faces=color_source_loaded.faces,
            anchor_labels=anchor_labels,
        )
        transferred = transfer_face_region_ownership(
            source_positions=color_source_loaded.positions,
            source_faces=color_source_loaded.faces,
            source_face_labels=source_part_labels,
            target_positions=target_loaded.positions,
            target_faces=target_loaded.faces,
            neighbors=56,
            chunk_size=1024,
            distance_power=1.15,
            normal_power=1.45,
        )
        face_labels = np.asarray(transferred["target_face_labels"], dtype=np.int32)
        target_zone_scores = _duck_role_zone_scores(normalize_positions(face_centroids(target_loaded.positions, target_loaded.faces)))
        for role in ("bandana", "beak", "hat", "body"):
            part_id = int(part_ids[role])
            better = target_zone_scores[role] >= 0.62
            face_labels[better] = part_id
        face_colors = part_palette[face_labels]
        min_component_size = max(48, min(256, int(len(target_loaded.faces)) // 1800 or 48))
        face_labels = merge_small_palette_islands(
            face_labels,
            face_colors,
            part_palette,
            target_loaded.faces,
            min_component_size=min_component_size,
        )
        face_labels = smooth_face_palette_indices(
            face_labels,
            face_colors,
            part_palette,
            target_loaded.faces,
            iterations=3,
        )
        palette, face_labels = compact_palette(part_palette, face_labels)
        return _write_asset_bundle(
            loaded=target_loaded,
            face_labels=face_labels,
            palette=palette,
            output_dir=output_dir,
            object_name=object_name,
            obj_filename=obj_filename,
            threemf_filename=threemf_filename,
            preview_filename=preview_filename,
            swatch_filename=swatch_filename,
            palette_csv_filename=palette_csv_filename,
            report_filename=report_filename,
            started=started,
            strategy="geometry_transfer_duck_semantic_parts",
            notes=[
                "This experiment converts the source duck into semantic part ownership: body, bandana, hat, beak, and accent.",
                "Those part IDs are transferred to repaired geometry before palette colors are reassigned, so the export favors printable paint zones over local texture noise.",
                "3MF export still uses standards-based colorgroup face colors.",
            ],
            extra_report={
                "color_source_path": str(color_source_loaded.source_path),
                "color_source_format": color_source_loaded.source_format,
                "color_transfer_applied": True,
                "region_count": int(len(palette)),
                "semantic_group_count": int(len(palette)),
                "effective_max_colors": int(len(palette)),
                "semantic_part_ids": part_ids,
                "duck_part_anchor_labels": anchor_labels,
                "corner_bake_metadata": corner_metadata,
                "source_component_count": int(transferred["source_component_count"]),
                "region_transfer_mode": "duck_semantic_parts",
            },
        )
    if strategy in {"duck_seeded_parts", "geometry_transfer_duck_seeded_parts"}:
        posterized_texture = _legacy_posterize_texture(color_source_loaded.texture_rgb, image_palette=max_colors)
        corner_colors, corner_metadata = bake_texture_to_corner_colors(
            posterized_texture,
            color_source_loaded.texcoords,
            color_source_loaded.faces,
            pad_pixels=max(2, min(8, int(max_colors))),
            sampling_mode="nearest",
        )
        source_face_colors = face_colors_from_corner_colors(corner_colors)
        source_part_labels, part_palette, part_ids = _build_duck_seeded_parts(
            face_colors=source_face_colors,
            positions=color_source_loaded.positions,
            faces=color_source_loaded.faces,
        )
        transferred = transfer_face_region_ownership(
            source_positions=color_source_loaded.positions,
            source_faces=color_source_loaded.faces,
            source_face_labels=source_part_labels,
            target_positions=target_loaded.positions,
            target_faces=target_loaded.faces,
            neighbors=56,
            chunk_size=1024,
            distance_power=1.15,
            normal_power=1.45,
        )
        face_labels = np.asarray(transferred["target_face_labels"], dtype=np.int32)
        min_component_size = max(48, min(256, int(len(target_loaded.faces)) // 1800 or 48))
        face_colors = part_palette[face_labels]
        face_labels = merge_small_palette_islands(
            face_labels,
            face_colors,
            part_palette,
            target_loaded.faces,
            min_component_size=min_component_size,
        )
        face_labels = smooth_face_palette_indices(
            face_labels,
            face_colors,
            part_palette,
            target_loaded.faces,
            iterations=3,
        )
        palette, face_labels = compact_palette(part_palette, face_labels)
        return _write_asset_bundle(
            loaded=target_loaded,
            face_labels=face_labels,
            palette=palette,
            output_dir=output_dir,
            object_name=object_name,
            obj_filename=obj_filename,
            threemf_filename=threemf_filename,
            preview_filename=preview_filename,
            swatch_filename=swatch_filename,
            palette_csv_filename=palette_csv_filename,
            report_filename=report_filename,
            started=started,
            strategy="geometry_transfer_duck_seeded_parts",
            notes=[
                "This experiment uses explicit duck-part seed zones on the source mesh: body, bandana, hat brim, hat crown, beak, boots, and accent.",
                "Those seeded part IDs are transferred to repaired geometry directly, avoiding any dependence on inferred semantic groups from muddy source colors.",
                "3MF export still uses standards-based colorgroup face colors.",
            ],
            extra_report={
                "color_source_path": str(color_source_loaded.source_path),
                "color_source_format": color_source_loaded.source_format,
                "color_transfer_applied": True,
                "region_count": int(len(palette)),
                "semantic_group_count": int(len(palette)),
                "effective_max_colors": int(len(palette)),
                "semantic_part_ids": part_ids,
                "corner_bake_metadata": corner_metadata,
                "source_component_count": int(transferred["source_component_count"]),
                "region_transfer_mode": "duck_seeded_parts",
            },
        )
    if strategy in {"legacy_fast_face_labels", "legacy_face_regions"}:
        source_regions = _build_legacy_source_face_region_model(color_source_loaded, max_colors=max_colors)
        palette = source_regions.palette
        source_face_labels = source_regions.face_labels
        anchor_labels = source_regions.anchor_labels
        transferred = transfer_face_region_ownership(
            source_positions=color_source_loaded.positions,
            source_faces=color_source_loaded.faces,
            source_face_labels=source_face_labels,
            target_positions=target_loaded.positions,
            target_faces=target_loaded.faces,
            neighbors=48,
            chunk_size=1024,
        )
        face_labels = np.asarray(transferred["target_face_labels"], dtype=np.int32)
        target_face_texture_colors, target_texture_diagnostics = _target_anchor_face_colors(target_loaded, palette, face_labels)
        face_labels = _apply_duck_part_anchor_bias(
            face_labels=face_labels,
            palette=palette,
            face_colors=target_face_texture_colors,
            positions=target_loaded.positions,
            faces=target_loaded.faces,
            anchor_labels=anchor_labels,
        )
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
            face_colors,
            palette,
            target_loaded.faces,
            iterations=3,
        )
        palette, face_labels = compact_palette(palette, face_labels)
        return _write_asset_bundle(
            loaded=target_loaded,
            face_labels=face_labels,
            palette=palette,
            output_dir=output_dir,
            object_name=object_name,
            obj_filename=obj_filename,
            threemf_filename=threemf_filename,
            preview_filename=preview_filename,
            swatch_filename=swatch_filename,
            palette_csv_filename=palette_csv_filename,
            report_filename=report_filename,
            started=started,
            strategy="geometry_transfer_legacy_face_regions",
            notes=[
                "This conversion uses the old 3dcolor fast path on the source mesh, extracts connected source face regions, and transfers those region labels onto the target geometry.",
                "It is designed to preserve sharper printable paint zones during repaired-geometry transfer than per-vertex label voting.",
                "3MF export still uses standards-based colorgroup face colors.",
            ],
            extra_report={
                "color_source_path": str(color_source_loaded.source_path),
                "color_source_format": color_source_loaded.source_format,
                "color_transfer_applied": True,
                "color_source_vertex_count": int(len(color_source_loaded.positions)),
                "target_vertex_count": int(len(target_loaded.positions)),
                "region_count": int(len(palette)),
                "semantic_group_count": int(len(palette)),
                "effective_max_colors": int(len(palette)),
                "legacy_fast_path": True,
                "region_transfer_mode": "connected_face_regions",
                "source_component_count": int(transferred["source_component_count"]),
                "duck_part_anchor_labels": anchor_labels,
                "target_texture_diagnostics": target_texture_diagnostics,
                **source_regions.metadata,
            },
        )
    if strategy in {"legacy_face_regions_graph", "geometry_transfer_legacy_face_regions_graph"}:
        source_regions = _build_legacy_source_face_region_model(color_source_loaded, max_colors=max_colors)
        palette = source_regions.palette
        source_face_labels = source_regions.face_labels
        anchor_labels = source_regions.anchor_labels
        transferred = transfer_face_region_ownership(
            source_positions=color_source_loaded.positions,
            source_faces=color_source_loaded.faces,
            source_face_labels=source_face_labels,
            target_positions=target_loaded.positions,
            target_faces=target_loaded.faces,
            neighbors=64,
            chunk_size=1024,
            distance_power=1.35,
            normal_power=2.0,
            return_label_scores=True,
        )
        face_labels = np.asarray(transferred["target_face_labels"], dtype=np.int32)
        raw_label_scores = transferred.get("target_label_scores")
        label_scores = np.asarray(raw_label_scores, dtype=np.float32) if raw_label_scores is not None else np.zeros((0, 0), dtype=np.float32)
        if label_scores.size:
            face_labels = refine_face_labels_with_graph_smoothing(
                face_labels,
                label_scores,
                target_loaded.faces,
                target_loaded.positions,
                iterations=5,
                smoothness_weight=0.42,
                boundary_power=1.8,
            )
        target_face_texture_colors, target_texture_diagnostics = _target_anchor_face_colors(target_loaded, palette, face_labels)
        face_labels = _apply_duck_part_anchor_bias(
            face_labels=face_labels,
            palette=palette,
            face_colors=target_face_texture_colors,
            positions=target_loaded.positions,
            faces=target_loaded.faces,
            anchor_labels=anchor_labels,
        )
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
            face_colors,
            palette,
            target_loaded.faces,
            iterations=3,
        )
        palette, face_labels = compact_palette(palette, face_labels)
        return _write_asset_bundle(
            loaded=target_loaded,
            face_labels=face_labels,
            palette=palette,
            output_dir=output_dir,
            object_name=object_name,
            obj_filename=obj_filename,
            threemf_filename=threemf_filename,
            preview_filename=preview_filename,
            swatch_filename=swatch_filename,
            palette_csv_filename=palette_csv_filename,
            report_filename=report_filename,
            started=started,
            strategy="geometry_transfer_legacy_face_regions_graph",
            notes=[
                "This experiment uses legacy source face regions, stronger normal-aware ownership transfer, and a graph-smoothed label refinement pass on the target mesh.",
                "It is intended to improve curved-surface region boundaries before the usual island cleanup and smoothing.",
                "3MF export still uses standards-based colorgroup face colors.",
            ],
            extra_report={
                "color_source_path": str(color_source_loaded.source_path),
                "color_source_format": color_source_loaded.source_format,
                "color_transfer_applied": True,
                "color_source_vertex_count": int(len(color_source_loaded.positions)),
                "target_vertex_count": int(len(target_loaded.positions)),
                "region_count": int(len(palette)),
                "semantic_group_count": int(len(palette)),
                "effective_max_colors": int(len(palette)),
                "legacy_fast_path": True,
                "region_transfer_mode": "connected_face_regions_graph",
                "source_component_count": int(transferred["source_component_count"]),
                "duck_part_anchor_labels": anchor_labels,
                "target_texture_diagnostics": target_texture_diagnostics,
                **source_regions.metadata,
            },
        )
    if strategy in {
        "blender_like_bake_face_labels",
        "blender_like_bake_face_regions",
        "geometry_transfer_blender_like_bake_face_regions",
        "geometry_transfer_blender_like_bake_duck_intent",
    }:
        duck_intent_enabled = strategy == "geometry_transfer_blender_like_bake_duck_intent"
        baked_vertex_colors, bake_metadata = bake_texture_to_vertex_colors(
            color_source_loaded.texture_rgb,
            color_source_loaded.texcoords,
            color_source_loaded.faces,
            pad_pixels=max(2, min(8, int(max_colors))),
            sampling_mode="bilinear",
        )
        palette, source_vertex_labels = _legacy_quantize_vertex_colors(baked_vertex_colors, num_colors=max_colors)
        source_face_labels = _legacy_assign_face_labels(color_source_loaded.faces, source_vertex_labels)
        anchor_labels = _infer_duck_part_anchor_labels(
            face_labels=source_face_labels,
            palette=palette,
            positions=color_source_loaded.positions,
            faces=color_source_loaded.faces,
        )
        transferred = transfer_face_region_ownership(
            source_positions=color_source_loaded.positions,
            source_faces=color_source_loaded.faces,
            source_face_labels=source_face_labels,
            target_positions=target_loaded.positions,
            target_faces=target_loaded.faces,
            neighbors=48,
            chunk_size=1024,
        )
        face_labels = np.asarray(transferred["target_face_labels"], dtype=np.int32)
        target_face_texture_colors, target_texture_diagnostics = _target_anchor_face_colors(target_loaded, palette, face_labels)
        face_labels = _apply_duck_part_anchor_bias(
            face_labels=face_labels,
            palette=palette,
            face_colors=target_face_texture_colors,
            positions=target_loaded.positions,
            faces=target_loaded.faces,
            anchor_labels=anchor_labels,
        )
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
            face_colors,
            palette,
            target_loaded.faces,
            iterations=3,
        )
        palette, face_labels = compact_palette(palette, face_labels)
        duck_color_intent: dict[str, Any] | None = None
        if duck_intent_enabled:
            face_labels, duck_color_intent = _apply_duck_color_intent_rules(
                face_labels=face_labels,
                palette=palette,
                positions=target_loaded.positions,
                faces=target_loaded.faces,
            )
            palette, face_labels = compact_palette(palette, face_labels)
        return _write_asset_bundle(
            loaded=target_loaded,
            face_labels=face_labels,
            palette=palette,
            output_dir=output_dir,
            object_name=object_name,
            obj_filename=obj_filename,
            threemf_filename=threemf_filename,
            preview_filename=preview_filename,
            swatch_filename=swatch_filename,
            palette_csv_filename=palette_csv_filename,
            report_filename=report_filename,
            started=started,
            strategy="geometry_transfer_blender_like_bake_duck_intent" if duck_intent_enabled else "geometry_transfer_blender_like_bake_face_regions",
            notes=[
                "This conversion uses a Blender-like source bake first, then transfers connected baked face regions onto the target geometry.",
                "It is intended to preserve cleaner source paint regions than raw nearest texture sampling or per-vertex label voting during repaired-geometry transfer.",
                *(
                    ["Duck color-intent cleanup keeps large body/head regions blue, keeps warm colors mostly on the beak, and preserves small emblems."]
                    if duck_intent_enabled
                    else []
                ),
                "3MF export still uses standards-based colorgroup face colors.",
            ],
            extra_report={
                "color_source_path": str(color_source_loaded.source_path),
                "color_source_format": color_source_loaded.source_format,
                "color_transfer_applied": True,
                "color_source_vertex_count": int(len(color_source_loaded.positions)),
                "target_vertex_count": int(len(target_loaded.positions)),
                "region_count": int(len(palette)),
                "semantic_group_count": int(len(palette)),
                "effective_max_colors": int(len(palette)),
                "blender_like_bake": True,
                "bake_metadata": bake_metadata,
                "region_transfer_mode": "connected_face_regions",
                "source_component_count": int(transferred["source_component_count"]),
                "duck_part_anchor_labels": anchor_labels,
                "duck_color_intent": duck_color_intent,
                "target_texture_diagnostics": target_texture_diagnostics,
            },
        )

    try:
        transferred = _transfer_source_texture_regions_to_target(
            target_loaded=target_loaded,
            color_source_loaded=color_source_loaded,
            max_colors=max_colors,
        )
        return _write_asset_bundle(
            loaded=target_loaded,
            face_labels=np.asarray(transferred["face_palette_indices"], dtype=np.int32),
            palette=np.asarray(transferred["palette"], dtype=np.uint8),
            output_dir=output_dir,
            object_name=object_name,
            obj_filename=obj_filename,
            threemf_filename=threemf_filename,
            preview_filename=preview_filename,
            swatch_filename=swatch_filename,
            palette_csv_filename=palette_csv_filename,
            report_filename=report_filename,
            started=started,
            strategy="geometry_transfer_texture_regions",
            notes=[
                "This conversion segments the source texture into cleaner regions first, then transfers those regions onto the target geometry.",
                "It is designed to preserve larger printable paint zones instead of letting transferred shading dominate the palette.",
                "3MF export uses standards-based colorgroup face colors.",
            ],
            extra_report={
                "color_source_path": str(color_source_loaded.source_path),
                "color_source_format": color_source_loaded.source_format,
                "color_transfer_applied": True,
                "color_source_vertex_count": int(len(color_source_loaded.positions)),
                "target_vertex_count": int(len(target_loaded.positions)),
                "region_count": int(transferred["raw_region_count"]),
                "semantic_group_count": int(transferred["effective_palette_size"]),
                "effective_max_colors": int(transferred["effective_palette_size"]),
                "texture_region_budget": int(transferred["region_budget"]),
            },
        )
    except Exception:
        source_vertex_colors = sample_texture(color_source_loaded.texture_rgb, color_source_loaded.texcoords)
        transferred_vertex_colors = transfer_vertex_colors_from_source(
            source_positions=color_source_loaded.positions,
            source_vertex_colors=source_vertex_colors,
            target_positions=target_loaded.positions,
            neighbors=1,
        )
        face_colors = np.clip(np.rint(transferred_vertex_colors[target_loaded.faces].mean(axis=1)), 0, 255).astype(np.uint8)
        return convert_face_colored_mesh_to_assets(
            target_loaded,
            face_colors=face_colors,
            out_dir=out_dir,
            max_colors=max_colors,
            object_name=object_name,
            obj_filename=obj_filename,
            threemf_filename=threemf_filename,
            preview_filename=preview_filename,
            swatch_filename=swatch_filename,
            palette_csv_filename=palette_csv_filename,
            report_filename=report_filename,
            strategy="geometry_transfer_region_first_fallback",
            notes=[
                "This conversion fell back to transferred vertex colors after the texture-region transfer path could not finish cleanly.",
                "3MF export uses standards-based colorgroup face colors.",
            ],
            extra_report={
                "color_source_path": str(color_source_loaded.source_path),
                "color_source_format": color_source_loaded.source_format,
                "color_transfer_applied": True,
                "color_source_vertex_count": int(len(color_source_loaded.positions)),
                "target_vertex_count": int(len(target_loaded.positions)),
            },
        )


def convert_repaired_color_transfer_to_assets(
    color_source_path: str | Path,
    target_path: str | Path,
    *,
    color_source_texture_path: str | Path | None = None,
    target_texture_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    max_colors: int = 12,
    target_face_count: int | None = None,
    strategy: str = "legacy_fast_face_labels",
    object_name: str | None = None,
    obj_filename: str = "region_materials.obj",
    threemf_filename: str = "region_colorgroup.3mf",
    preview_filename: str = "region_preview.png",
    swatch_filename: str = "palette_swatches.png",
    palette_csv_filename: str = "palette.csv",
    report_filename: str = "conversion_report.json",
) -> dict[str, Any]:
    color_source_loaded = load_textured_model(color_source_path, texture_path=color_source_texture_path)
    target_loaded = load_geometry_model(target_path, texture_path=target_texture_path)
    target_original_geometry_stats = _mesh_geometry_stats(target_loaded.positions, target_loaded.faces)
    target_loaded, target_simplification = _simplify_loaded_geometry(
        target_loaded,
        target_face_count=target_face_count,
    )
    target_geometry_stats = _mesh_geometry_stats(target_loaded.positions, target_loaded.faces)
    report = convert_color_transferred_mesh_to_assets(
        target_loaded=target_loaded,
        color_source_loaded=color_source_loaded,
        out_dir=out_dir,
        max_colors=max_colors,
        strategy=strategy,
        object_name=object_name,
        obj_filename=obj_filename,
        threemf_filename=threemf_filename,
        preview_filename=preview_filename,
        swatch_filename=swatch_filename,
        palette_csv_filename=palette_csv_filename,
        report_filename=report_filename,
    )
    report.update(
        {
            "conversion_lane": "repaired_geometry_region_transfer",
            "target_path": str(target_loaded.source_path),
            "target_source_format": target_loaded.source_format,
            "target_texture_path": str(target_loaded.texture_path) if target_loaded.texture_path else None,
            "target_original_geometry_stats": target_original_geometry_stats,
            "target_simplification": target_simplification,
            "target_geometry_stats": target_geometry_stats,
        }
    )
    report["repaired_transfer_assessment"] = assess_repaired_transfer_candidate(report)
    report_path = Path(str(report["report_path"]))
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def convert_provider_baked_model_to_assets(
    provider_baked_path: str | Path,
    *,
    texture_path: str | Path | None = None,
    repair_result_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    n_regions: int = 8,
    strategy: str = "blender_cleanup_face_labels",
    object_name: str | None = None,
) -> dict[str, Any]:
    loaded = load_textured_model(provider_baked_path, texture_path=texture_path)
    texture_diagnostics = _texture_diagnostics(loaded.texture_rgb)
    report = convert_loaded_mesh_to_color_assets(
        loaded,
        out_dir=out_dir,
        n_regions=n_regions,
        strategy=strategy,
        object_name=object_name,
    )
    report.update(
        {
            "conversion_lane": "provider_baked_repaired_same_mesh",
            "provider_baked_path": str(Path(provider_baked_path).expanduser().resolve()),
            "provider_bake_texture_diagnostics": texture_diagnostics,
            "provider_bake_assessment": assess_provider_bake_candidate(report, texture_diagnostics),
            **_provider_repair_metadata(repair_result_path),
        }
    )
    report_path = Path(str(report["report_path"]))
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def convert_model_to_color_assets(
    source_path: str | Path,
    *,
    texture_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    n_regions: int = 5,
    strategy: str = "region_first",
    object_name: str | None = None,
    obj_filename: str = "region_materials.obj",
    threemf_filename: str = "region_colorgroup.3mf",
    preview_filename: str = "region_preview.png",
    swatch_filename: str = "palette_swatches.png",
    palette_csv_filename: str = "palette.csv",
    report_filename: str = "conversion_report.json",
) -> dict[str, Any]:
    loaded = load_textured_model(source_path, texture_path=texture_path)
    return convert_loaded_mesh_to_color_assets(
        loaded,
        out_dir=out_dir,
        n_regions=n_regions,
        strategy=strategy,
        object_name=object_name,
        obj_filename=obj_filename,
        threemf_filename=threemf_filename,
        preview_filename=preview_filename,
        swatch_filename=swatch_filename,
        palette_csv_filename=palette_csv_filename,
        report_filename=report_filename,
    )


def convert_textured_obj_to_region_assets(
    obj_path: str | Path,
    *,
    texture_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    n_regions: int = 5,
    strategy: str = "region_first",
    object_name: str | None = None,
) -> dict[str, Any]:
    return convert_model_to_color_assets(
        obj_path,
        texture_path=texture_path,
        out_dir=out_dir,
        n_regions=n_regions,
        strategy=strategy,
        object_name=object_name,
    )
