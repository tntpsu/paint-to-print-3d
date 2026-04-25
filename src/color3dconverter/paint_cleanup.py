from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .face_regions import build_face_adjacency, face_centroids, normalize_positions


def _safe_palette_lookup(palette: np.ndarray, label: int) -> np.ndarray:
    palette_array = np.asarray(palette, dtype=np.uint8)
    if len(palette_array) == 0:
        return np.zeros((3,), dtype=np.uint8)
    index = max(0, min(int(label), len(palette_array) - 1))
    return palette_array[index]


def _relative_luminance(color: np.ndarray) -> float:
    rgb = np.asarray(color, dtype=np.float32) / 255.0
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _color_signals(color: np.ndarray) -> dict[str, float]:
    rgb = np.asarray(color, dtype=np.float32) / 255.0
    r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    return {
        "luminance": _relative_luminance(color),
        "neutral": max(0.0, 1.0 - (max(r, g, b) - min(r, g, b)) * 4.0),
        "warm": max(0.0, 0.55 * r + 0.35 * g - 0.45 * b),
        "orange": max(0.0, 0.75 * r + 0.35 * g - 1.1 * b - 0.2 * abs(r - g)),
        "red": max(0.0, r - 0.65 * g - 0.45 * b),
        "blue": max(0.0, b - 0.55 * r - 0.35 * g),
        "dark": max(0.0, 0.42 - _relative_luminance(color)),
    }


def _connected_components(labels: np.ndarray, adjacency: list[list[int]]) -> tuple[np.ndarray, int]:
    component_ids = np.full((len(labels),), -1, dtype=np.int32)
    next_component = 0
    for start in range(len(labels)):
        if component_ids[start] >= 0:
            continue
        label = int(labels[start])
        stack = [start]
        component_ids[start] = next_component
        while stack:
            face_index = stack.pop()
            for neighbor in adjacency[face_index]:
                if component_ids[neighbor] >= 0 or int(labels[neighbor]) != label:
                    continue
                component_ids[neighbor] = next_component
                stack.append(neighbor)
        next_component += 1
    return component_ids, next_component


def paint_component_metrics(face_labels: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(face_labels, dtype=np.int32)
    face_array = np.asarray(faces, dtype=np.int64)
    face_count = int(len(labels))
    if face_count == 0 or len(face_array) == 0:
        return {
            "component_count": 0,
            "tiny_island_count": 0,
            "largest_component_share": 0.0,
            "tiny_threshold": 0,
        }
    adjacency = build_face_adjacency(face_array)
    component_ids, component_count = _connected_components(labels, adjacency)
    component_sizes = np.bincount(component_ids, minlength=component_count)
    tiny_threshold = max(4, min(64, max(1, face_count // 500)))
    return {
        "component_count": int(component_count),
        "tiny_island_count": int(np.sum(component_sizes < tiny_threshold)),
        "largest_component_share": round(float(component_sizes.max()) / float(face_count), 4) if len(component_sizes) else 0.0,
        "tiny_threshold": int(tiny_threshold),
    }


def _protected_detail_mask(
    *,
    labels: np.ndarray,
    palette: np.ndarray,
    positions: np.ndarray,
    faces: np.ndarray,
    protected_labels: set[int],
    enable_semantic_protection: bool,
) -> np.ndarray:
    if len(labels) == 0:
        return np.zeros((0,), dtype=bool)
    mask = np.isin(labels, np.array(sorted(protected_labels), dtype=np.int32)) if protected_labels else np.zeros((len(labels),), dtype=bool)
    if not enable_semantic_protection or len(faces) == 0:
        return mask

    centroids = normalize_positions(face_centroids(positions, faces))
    x = centroids[:, 0]
    y = centroids[:, 1]
    z = centroids[:, 2]
    head_or_face_zone = (x > 0.02) & (y > -0.18) & (y < 0.38) & (np.abs(z) < 0.38)
    beak_zone = (x > 0.22) & (y > -0.06) & (y < 0.30) & (np.abs(z) < 0.28)
    front_emblem_zone = (x > -0.10) & (x < 0.28) & (y > -0.30) & (y < 0.18) & (np.abs(z) < 0.42)

    for label in np.unique(labels).tolist():
        label_int = int(label)
        color = _safe_palette_lookup(palette, label_int)
        signals = _color_signals(color)
        label_faces = labels == label_int
        is_light_neutral = signals["neutral"] >= 0.30 and signals["luminance"] >= 0.62
        is_dark_eye_like = signals["dark"] >= 0.08 and signals["luminance"] <= 0.36
        is_beak_like = signals["orange"] >= 0.08 and signals["warm"] >= 0.18
        is_saturated_emblem = max(signals["red"], signals["blue"]) >= 0.16 and signals["luminance"] >= 0.18
        if is_light_neutral:
            mask |= label_faces & (head_or_face_zone | front_emblem_zone)
        if is_dark_eye_like:
            mask |= label_faces & head_or_face_zone
        if is_beak_like:
            mask |= label_faces & beak_zone
        if is_saturated_emblem:
            mask |= label_faces & front_emblem_zone
    return mask


def cleanup_paint_region_labels(
    *,
    face_labels: np.ndarray,
    palette: np.ndarray,
    positions: np.ndarray,
    faces: np.ndarray,
    min_component_size: int | None = None,
    max_passes: int = 4,
    protected_labels: set[int] | None = None,
    enable_semantic_protection: bool = True,
    protected_component_share: float = 0.65,
    max_absorb_face_share: float = 0.025,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Absorb tiny paint islands into neighboring regions without using AI.

    This is intentionally label-space cleanup. It does not invent new colors or
    change geometry; it only rewrites small connected components to a nearby
    existing palette label when that component is not protected detail.
    """
    labels = np.asarray(face_labels, dtype=np.int32).copy()
    face_array = np.asarray(faces, dtype=np.int64)
    pos = np.asarray(positions, dtype=np.float32)
    palette_array = np.asarray(palette, dtype=np.uint8)
    face_count = int(len(labels))
    if face_count == 0 or len(face_array) == 0 or len(palette_array) == 0:
        return labels, {
            "status": "skipped",
            "reason": "empty labels, faces, or palette",
            "before": paint_component_metrics(labels, face_array),
            "after": paint_component_metrics(labels, face_array),
        }

    minimum_size = (
        max(24, min(512, face_count // 350 or 24))
        if min_component_size is None
        else max(1, int(min_component_size))
    )
    pass_limit = max(1, int(max_passes))
    adjacency = build_face_adjacency(face_array)
    protected = _protected_detail_mask(
        labels=labels,
        palette=palette_array,
        positions=pos,
        faces=face_array,
        protected_labels=set(protected_labels or set()),
        enable_semantic_protection=enable_semantic_protection,
    )
    before = paint_component_metrics(labels, face_array)
    absorbed_components = 0
    absorbed_faces = 0
    protected_components = 0
    pass_summaries: list[dict[str, Any]] = []

    for pass_index in range(pass_limit):
        component_ids, component_count = _connected_components(labels, adjacency)
        component_sizes = np.bincount(component_ids, minlength=component_count)
        global_label_counts = np.bincount(labels, minlength=max(len(palette_array), int(labels.max()) + 1 if len(labels) else 0)).astype(np.float32)
        updates: list[tuple[np.ndarray, int, int, int]] = []
        protected_this_pass = 0

        for component_id, component_size in enumerate(component_sizes.tolist()):
            size = int(component_size)
            if size >= minimum_size or (float(size) / float(max(face_count, 1))) > float(max_absorb_face_share):
                continue
            member_indexes = np.flatnonzero(component_ids == int(component_id))
            if len(member_indexes) == 0:
                continue
            label = int(labels[int(member_indexes[0])])
            protected_share = float(np.mean(protected[member_indexes])) if len(member_indexes) else 0.0
            if protected_share >= float(protected_component_share) and size >= 2:
                protected_this_pass += 1
                continue

            neighbor_counts: Counter[int] = Counter()
            for face_index in member_indexes.tolist():
                for neighbor in adjacency[int(face_index)]:
                    neighbor_label = int(labels[int(neighbor)])
                    if neighbor_label != label:
                        neighbor_counts[neighbor_label] += 1
            if not neighbor_counts:
                continue

            source_color = _safe_palette_lookup(palette_array, label).astype(np.float32)
            best_label = label
            best_score: float | None = None
            for candidate_label, border_count in neighbor_counts.items():
                candidate_color = _safe_palette_lookup(palette_array, int(candidate_label)).astype(np.float32)
                color_distance = float(np.sum((source_color - candidate_color) ** 2))
                label_share = float(global_label_counts[int(candidate_label)]) / float(max(face_count, 1))
                score = color_distance / float(max(border_count, 1)) - (label_share * 160.0)
                if best_score is None or score < best_score:
                    best_score = score
                    best_label = int(candidate_label)
            if best_label == label:
                continue
            updates.append((member_indexes, label, best_label, size))

        for member_indexes, _old_label, new_label, size in updates:
            labels[member_indexes] = int(new_label)
            absorbed_components += 1
            absorbed_faces += int(size)

        protected_components += protected_this_pass
        pass_summaries.append(
            {
                "pass_index": int(pass_index),
                "absorbed_components": int(len(updates)),
                "absorbed_faces": int(sum(item[3] for item in updates)),
                "protected_components": int(protected_this_pass),
            }
        )
        if not updates:
            break

    after = paint_component_metrics(labels, face_array)
    improved = (
        int(after["component_count"]) < int(before["component_count"])
        or int(after["tiny_island_count"]) < int(before["tiny_island_count"])
    )
    return labels, {
        "status": "improved" if improved else "unchanged",
        "policy": "deterministic_connected_component_absorption_v1",
        "config": {
            "min_component_size": int(minimum_size),
            "max_passes": int(pass_limit),
            "protected_labels": sorted(int(label) for label in (protected_labels or set())),
            "enable_semantic_protection": bool(enable_semantic_protection),
            "protected_component_share": float(protected_component_share),
            "max_absorb_face_share": float(max_absorb_face_share),
        },
        "before": before,
        "after": after,
        "absorbed_components": int(absorbed_components),
        "absorbed_faces": int(absorbed_faces),
        "protected_components": int(protected_components),
        "pass_summaries": pass_summaries,
    }
