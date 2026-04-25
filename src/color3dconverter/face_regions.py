from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


def sample_texture(texture_rgb: np.ndarray, texcoords: np.ndarray) -> np.ndarray:
    if texcoords.size == 0:
        return np.full((0, 3), 255, dtype=np.uint8)
    height, width = texture_rgb.shape[:2]
    uv = np.asarray(texcoords, dtype=np.float32)
    uv = uv - np.floor(uv)
    x = np.clip(np.rint(uv[:, 0] * (width - 1)).astype(np.int64), 0, width - 1)
    y = np.clip(np.rint((1.0 - uv[:, 1]) * (height - 1)).astype(np.int64), 0, height - 1)
    return texture_rgb[y, x]


def normalize_positions(positions: np.ndarray) -> np.ndarray:
    points = np.asarray(positions, dtype=np.float32)
    if len(points) == 0:
        return points
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    scale = float(np.max(np.maximum(bbox_max - bbox_min, 1e-6)))
    return (points - center) / max(scale, 1e-6)


def transfer_vertex_colors_from_source(
    *,
    source_positions: np.ndarray,
    source_vertex_colors: np.ndarray,
    target_positions: np.ndarray,
    neighbors: int = 1,
    chunk_size: int = 2048,
) -> np.ndarray:
    source_points = normalize_positions(source_positions)
    target_points = normalize_positions(target_positions)
    if len(source_points) == 0 or len(target_points) == 0:
        return np.full((len(target_points), 3), 255, dtype=np.uint8)

    neighbor_count = max(1, min(int(neighbors), len(source_points)))
    source_colors_f = np.asarray(source_vertex_colors, dtype=np.float32)
    transferred = np.zeros((len(target_points), 3), dtype=np.float32)
    for start in range(0, len(target_points), chunk_size):
        stop = min(start + chunk_size, len(target_points))
        chunk = target_points[start:stop]
        distances = ((chunk[:, None, :] - source_points[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argpartition(distances, kth=neighbor_count - 1, axis=1)[:, :neighbor_count]
        nearest_distances = np.take_along_axis(distances, nearest, axis=1)
        weights = 1.0 / np.maximum(nearest_distances, 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)
        nearest_colors = source_colors_f[nearest]
        transferred[start:stop] = (nearest_colors * weights[:, :, None]).sum(axis=1)
    return np.clip(np.rint(transferred), 0, 255).astype(np.uint8)


def average_by_cluster(values: np.ndarray, inverse: np.ndarray, cluster_count: int) -> np.ndarray:
    if len(values) == 0 or cluster_count <= 0:
        return np.zeros((0, values.shape[1] if values.ndim > 1 else 1), dtype=np.float32)
    value_f = values.astype(np.float64, copy=False)
    totals = np.zeros((cluster_count, value_f.shape[1]), dtype=np.float64)
    np.add.at(totals, inverse, value_f)
    counts = np.bincount(inverse, minlength=cluster_count).astype(np.float64)
    counts[counts == 0] = 1.0
    return (totals / counts[:, None]).astype(np.float32)


def weighted_average_by_cluster(values: np.ndarray, inverse: np.ndarray, weights: np.ndarray, cluster_count: int) -> np.ndarray:
    if len(values) == 0 or cluster_count <= 0:
        return np.zeros((0, values.shape[1] if values.ndim > 1 else 1), dtype=np.float32)
    value_f = values.astype(np.float64, copy=False)
    weight_f = np.asarray(weights, dtype=np.float64).reshape((-1,))
    if len(weight_f) != len(value_f):
        raise ValueError("weights must align with values")
    totals = np.zeros((cluster_count, value_f.shape[1]), dtype=np.float64)
    np.add.at(totals, inverse, value_f * weight_f[:, None])
    weight_totals = np.bincount(inverse, weights=weight_f, minlength=cluster_count).astype(np.float64)
    weight_totals[weight_totals == 0] = 1.0
    return (totals / weight_totals[:, None]).astype(np.float32)


def nearest_palette_indices(colors: np.ndarray, palette: np.ndarray) -> np.ndarray:
    if colors.size == 0:
        return np.zeros((0,), dtype=np.int32)
    color_f = np.asarray(colors, dtype=np.float32)
    palette_f = np.asarray(palette, dtype=np.float32)
    distances = ((color_f[:, None, :] - palette_f[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distances, axis=1).astype(np.int32)


def weighted_kmeans_palette(colors: np.ndarray, weights: np.ndarray, max_colors: int) -> tuple[np.ndarray, np.ndarray]:
    if colors.size == 0:
        return np.zeros((0, 3), dtype=np.uint8), np.zeros((0,), dtype=np.int32)
    sample = np.asarray(colors, dtype=np.float32)
    sample_weights = np.asarray(weights, dtype=np.float32).reshape((-1,))
    if len(sample_weights) != len(sample):
        raise ValueError("weights must align with colors")
    unique_colors = np.unique(np.clip(np.rint(sample), 0, 255).astype(np.uint8), axis=0)
    if len(unique_colors) <= max_colors:
        return unique_colors, nearest_palette_indices(sample, unique_colors)

    luminance = sample @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    order = np.argsort(luminance, kind="stable")
    ordered_sample = sample[order]
    ordered_weights = np.maximum(sample_weights[order], 1.0)
    cumulative = np.cumsum(ordered_weights)
    total_weight = float(cumulative[-1]) if len(cumulative) else 0.0
    if total_weight <= 0.0:
        seed_indexes = np.linspace(0, len(sample) - 1, num=max_colors, dtype=int)
        centers = sample[seed_indexes].copy()
    else:
        targets = np.linspace(0.0, total_weight, num=max_colors + 2, dtype=np.float32)[1:-1]
        seed_indexes = np.searchsorted(cumulative, targets, side="left")
        seed_indexes = np.clip(seed_indexes, 0, len(ordered_sample) - 1)
        centers = ordered_sample[seed_indexes].copy()

    for _ in range(24):
        distances = ((sample[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for idx in range(len(centers)):
            members = labels == idx
            if not np.any(members):
                continue
            new_centers[idx] = np.average(sample[members], axis=0, weights=np.maximum(sample_weights[members], 1.0))
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    palette = np.clip(np.rint(centers), 0, 255).astype(np.uint8)
    unique_palette = np.unique(palette, axis=0)
    labels = nearest_palette_indices(sample, unique_palette)
    return unique_palette, labels


def weighted_feature_kmeans_labels(features: np.ndarray, weights: np.ndarray, max_groups: int) -> np.ndarray:
    if features.size == 0:
        return np.zeros((0,), dtype=np.int32)
    sample = np.asarray(features, dtype=np.float32)
    sample_weights = np.maximum(np.asarray(weights, dtype=np.float32).reshape((-1,)), 1.0)
    if len(sample_weights) != len(sample):
        raise ValueError("weights must align with features")
    if len(sample) <= max_groups:
        return np.arange(len(sample), dtype=np.int32)

    luminance = sample[:, 0]
    order = np.argsort(luminance, kind="stable")
    ordered_sample = sample[order]
    ordered_weights = sample_weights[order]
    cumulative = np.cumsum(ordered_weights)
    total_weight = float(cumulative[-1]) if len(cumulative) else 0.0
    if total_weight <= 0.0:
        seed_indexes = np.linspace(0, len(sample) - 1, num=max_groups, dtype=int)
        centers = sample[seed_indexes].copy()
    else:
        targets = np.linspace(0.0, total_weight, num=max_groups + 2, dtype=np.float32)[1:-1]
        seed_indexes = np.searchsorted(cumulative, targets, side="left")
        seed_indexes = np.clip(seed_indexes, 0, len(ordered_sample) - 1)
        centers = ordered_sample[seed_indexes].copy()

    for _ in range(28):
        distances = ((sample[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        for idx in range(len(centers)):
            members = labels == idx
            if not np.any(members):
                continue
            new_centers[idx] = np.average(sample[members], axis=0, weights=sample_weights[members])
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers
    distances = ((sample[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distances, axis=1).astype(np.int32)


def compute_face_normals(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    face_vertices = np.asarray(positions, dtype=np.float32)[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-8
    if np.any(valid):
        normals[valid] /= lengths[valid][:, None]
    return normals.astype(np.float32, copy=False)


def face_centroids(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(positions, dtype=np.float32)[np.asarray(faces, dtype=np.int64)].mean(axis=1).astype(np.float32)


def face_areas(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros((0,), dtype=np.float32)
    face_vertices = np.asarray(positions, dtype=np.float32)[np.asarray(faces, dtype=np.int64)]
    cross = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
    return (0.5 * np.linalg.norm(cross, axis=1)).astype(np.float32, copy=False)


def build_face_adjacency(faces: np.ndarray) -> list[list[int]]:
    adjacency: list[set[int]] = [set() for _ in range(len(faces))]
    edge_map: dict[tuple[int, int], int] = {}
    for face_index, face in enumerate(np.asarray(faces, dtype=np.int64)):
        a, b, c = (int(face[0]), int(face[1]), int(face[2]))
        for edge in ((a, b), (b, c), (c, a)):
            key = (edge[0], edge[1]) if edge[0] < edge[1] else (edge[1], edge[0])
            other = edge_map.get(key)
            if other is None:
                edge_map[key] = face_index
                continue
            adjacency[face_index].add(other)
            adjacency[other].add(face_index)
    return [sorted(items) for items in adjacency]


def build_connected_face_components(face_labels: np.ndarray, faces: np.ndarray) -> np.ndarray:
    labels = np.asarray(face_labels, dtype=np.int32)
    if len(labels) == 0 or len(faces) == 0:
        return np.zeros((len(labels),), dtype=np.int32)
    adjacency = build_face_adjacency(faces)
    components = np.full(len(labels), -1, dtype=np.int32)
    next_component = 0
    for start in range(len(labels)):
        if components[start] >= 0:
            continue
        target_label = int(labels[start])
        stack = [start]
        components[start] = next_component
        while stack:
            face_index = stack.pop()
            for neighbor in adjacency[face_index]:
                if components[neighbor] >= 0 or int(labels[neighbor]) != target_label:
                    continue
                components[neighbor] = next_component
                stack.append(neighbor)
        next_component += 1
    return components


def transfer_face_region_ownership(
    *,
    source_positions: np.ndarray,
    source_faces: np.ndarray,
    source_face_labels: np.ndarray,
    target_positions: np.ndarray,
    target_faces: np.ndarray,
    neighbors: int = 48,
    chunk_size: int = 1024,
    distance_power: float = 1.0,
    normal_power: float = 1.0,
    return_label_scores: bool = False,
) -> dict[str, np.ndarray | int]:
    source_face_labels = np.asarray(source_face_labels, dtype=np.int32)
    source_faces = np.asarray(source_faces, dtype=np.int64)
    target_faces = np.asarray(target_faces, dtype=np.int64)
    source_face_count = int(len(source_faces))
    target_face_count = int(len(target_faces))
    if source_face_count == 0 or target_face_count == 0:
        return {
            "target_face_labels": np.zeros((target_face_count,), dtype=np.int32),
            "source_component_count": 0,
            "source_component_labels": np.zeros((0,), dtype=np.int32),
            "target_component_ids": np.zeros((target_face_count,), dtype=np.int32),
        }

    source_component_ids = build_connected_face_components(source_face_labels, source_faces)
    source_component_count = int(source_component_ids.max()) + 1 if len(source_component_ids) else 0
    source_component_labels = np.zeros((source_component_count,), dtype=np.int32)
    label_count = int(source_face_labels.max()) + 1 if len(source_face_labels) else 0
    for component_id in range(source_component_count):
        member_indexes = np.flatnonzero(source_component_ids == component_id)
        if len(member_indexes) == 0:
            continue
        component_labels = source_face_labels[member_indexes]
        unique_labels, counts = np.unique(component_labels, return_counts=True)
        source_component_labels[component_id] = int(unique_labels[int(np.argmax(counts))])

    source_centroids = normalize_positions(face_centroids(source_positions, source_faces))
    target_centroids = normalize_positions(face_centroids(target_positions, target_faces))
    source_normals = compute_face_normals(source_positions, source_faces)
    target_normals = compute_face_normals(target_positions, target_faces)
    source_area_weights = np.sqrt(np.maximum(face_areas(source_positions, source_faces), 1e-8))
    max_area_weight = float(source_area_weights.max()) if len(source_area_weights) else 1.0
    source_area_weights /= max(max_area_weight, 1e-8)
    source_area_weights = np.clip(0.5 + source_area_weights, 0.5, 1.5)

    neighbor_count = max(1, min(int(neighbors), source_face_count))
    source_tree = cKDTree(source_centroids.astype(np.float32, copy=False))
    target_component_ids = np.zeros((target_face_count,), dtype=np.int32)
    label_scores = np.zeros((target_face_count, label_count), dtype=np.float32) if return_label_scores and label_count > 0 else None
    for start in range(0, target_face_count, chunk_size):
        stop = min(start + chunk_size, target_face_count)
        centroid_chunk = target_centroids[start:stop]
        normal_chunk = target_normals[start:stop]
        nearest_distances, nearest = source_tree.query(centroid_chunk.astype(np.float32, copy=False), k=neighbor_count)
        if neighbor_count == 1:
            nearest_distances = nearest_distances[:, None]
            nearest = nearest[:, None]
        nearest_distances = np.asarray(nearest_distances, dtype=np.float32) ** 2
        nearest = np.asarray(nearest, dtype=np.int64)
        nearest_components = source_component_ids[nearest]
        nearest_normals = source_normals[nearest]
        normal_alignment = np.clip((nearest_normals * normal_chunk[:, None, :]).sum(axis=2), -1.0, 1.0)
        normal_weights = np.power(0.15 + 0.85 * np.clip(normal_alignment, 0.0, 1.0), max(float(normal_power), 0.0))
        distance_weights = np.power(1.0 / np.maximum(nearest_distances, 1e-8), max(float(distance_power), 0.0))
        area_weights = source_area_weights[nearest]
        combined_weights = distance_weights * normal_weights * area_weights
        for row_index in range(stop - start):
            component_votes: dict[int, float] = {}
            label_vote_row = label_scores[start + row_index] if label_scores is not None else None
            for candidate_index, component_id in enumerate(nearest_components[row_index].tolist()):
                component_int = int(component_id)
                weight = float(combined_weights[row_index, candidate_index])
                component_votes[component_int] = component_votes.get(component_int, 0.0) + weight
                if label_vote_row is not None and 0 <= component_int < len(source_component_labels):
                    label_int = int(source_component_labels[component_int])
                    if 0 <= label_int < len(label_vote_row):
                        label_vote_row[label_int] += weight
            target_component_ids[start + row_index] = int(max(component_votes.items(), key=lambda item: item[1])[0])
    result = {
        "target_face_labels": source_component_labels[target_component_ids],
        "source_component_count": int(source_component_count),
        "source_component_labels": source_component_labels,
        "target_component_ids": target_component_ids,
    }
    if label_scores is not None:
        result["target_label_scores"] = label_scores
    return result


def refine_face_labels_with_graph_smoothing(
    face_labels: np.ndarray,
    label_scores: np.ndarray,
    faces: np.ndarray,
    positions: np.ndarray,
    *,
    iterations: int = 4,
    smoothness_weight: float = 0.35,
    boundary_power: float = 1.5,
) -> np.ndarray:
    labels = np.asarray(face_labels, dtype=np.int32).copy()
    scores = np.asarray(label_scores, dtype=np.float32)
    if len(labels) == 0 or len(faces) == 0 or scores.size == 0:
        return labels
    adjacency = build_face_adjacency(faces)
    normals = compute_face_normals(positions, faces)
    for _ in range(max(int(iterations), 0)):
        updated = labels.copy()
        for face_index, neighbors in enumerate(adjacency):
            if not neighbors:
                continue
            local_scores = scores[face_index].copy()
            current_normal = normals[face_index]
            for neighbor in neighbors:
                neighbor_label = int(labels[neighbor])
                alignment = max(float(np.dot(current_normal, normals[neighbor])), 0.0)
                edge_weight = float(np.power(alignment, max(float(boundary_power), 0.0)))
                if 0 <= neighbor_label < len(local_scores):
                    local_scores[neighbor_label] += float(smoothness_weight) * edge_weight
            updated[face_index] = int(np.argmax(local_scores))
        if np.array_equal(updated, labels):
            break
        labels = updated
    return labels


def build_face_regions(
    face_colors: np.ndarray,
    faces: np.ndarray,
    positions: np.ndarray,
    *,
    color_threshold: float = 34.0,
    normal_threshold_degrees: float = 34.0,
) -> np.ndarray:
    face_count = int(len(faces))
    if face_count == 0:
        return np.zeros((0,), dtype=np.int32)
    adjacency = build_face_adjacency(faces)
    normals = compute_face_normals(positions, faces)
    colors_f = np.asarray(face_colors, dtype=np.float32)
    labels = np.full(face_count, -1, dtype=np.int32)
    cosine_threshold = math.cos(math.radians(float(normal_threshold_degrees)))
    next_label = 0
    for start in range(face_count):
        if labels[start] >= 0:
            continue
        labels[start] = next_label
        stack = [start]
        while stack:
            face_index = stack.pop()
            face_color = colors_f[face_index]
            face_normal = normals[face_index]
            for neighbor in adjacency[face_index]:
                if labels[neighbor] >= 0:
                    continue
                normal_similarity = float(np.dot(face_normal, normals[neighbor]))
                if normal_similarity < cosine_threshold:
                    continue
                color_distance = float(np.linalg.norm(face_color - colors_f[neighbor]))
                if color_distance > color_threshold:
                    continue
                labels[neighbor] = next_label
                stack.append(neighbor)
        next_label += 1
    return labels


def compact_palette(palette: np.ndarray, face_palette_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(palette) == 0 or len(face_palette_indices) == 0:
        return palette, face_palette_indices
    used = np.unique(face_palette_indices.astype(np.int32))
    remap = {int(old): new for new, old in enumerate(used.tolist())}
    compacted = palette[used]
    remapped = np.array([remap[int(value)] for value in face_palette_indices.tolist()], dtype=np.int32)
    return compacted, remapped


def merge_small_palette_islands(
    face_palette_indices: np.ndarray,
    face_colors: np.ndarray,
    palette: np.ndarray,
    faces: np.ndarray,
    *,
    min_component_size: int = 96,
) -> np.ndarray:
    labels = np.asarray(face_palette_indices, dtype=np.int32).copy()
    if len(labels) == 0 or len(faces) == 0:
        return labels
    adjacency = build_face_adjacency(faces)
    visited = np.zeros(len(labels), dtype=bool)
    face_colors_f = np.asarray(face_colors, dtype=np.float32)
    palette_f = np.asarray(palette, dtype=np.float32)

    for start in range(len(labels)):
        if visited[start]:
            continue
        label = int(labels[start])
        stack = [start]
        component: list[int] = []
        visited[start] = True
        while stack:
            face_index = stack.pop()
            component.append(face_index)
            for neighbor in adjacency[face_index]:
                if visited[neighbor] or int(labels[neighbor]) != label:
                    continue
                visited[neighbor] = True
                stack.append(neighbor)
        if len(component) >= min_component_size:
            continue

        neighbor_border_counts: dict[int, int] = {}
        for face_index in component:
            for neighbor in adjacency[face_index]:
                neighbor_label = int(labels[neighbor])
                if neighbor_label == label:
                    continue
                neighbor_border_counts[neighbor_label] = neighbor_border_counts.get(neighbor_label, 0) + 1
        if not neighbor_border_counts:
            continue

        component_color = face_colors_f[np.array(component, dtype=np.int64)].mean(axis=0)
        best_label = label
        best_score: float | None = None
        for candidate_label, border_count in neighbor_border_counts.items():
            color_distance = float(np.sum((component_color - palette_f[candidate_label]) ** 2))
            score = color_distance / max(border_count, 1)
            if best_score is None or score < best_score:
                best_score = score
                best_label = candidate_label
        labels[np.array(component, dtype=np.int64)] = int(best_label)
    return labels


def smooth_face_palette_indices(
    face_palette_indices: np.ndarray,
    face_colors: np.ndarray,
    palette: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int = 2,
) -> np.ndarray:
    labels = np.asarray(face_palette_indices, dtype=np.int32).copy()
    if len(labels) == 0 or len(faces) == 0:
        return labels
    adjacency = build_face_adjacency(faces)
    palette_f = np.asarray(palette, dtype=np.float32)
    face_colors_f = np.asarray(face_colors, dtype=np.float32)

    for _ in range(max(int(iterations), 0)):
        updated = labels.copy()
        for face_index, neighbors in enumerate(adjacency):
            if len(neighbors) < 2:
                continue
            neighbor_labels = labels[np.array(neighbors, dtype=np.int64)]
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            dominant_idx = int(np.argmax(counts))
            dominant_label = int(unique_labels[dominant_idx])
            dominant_count = int(counts[dominant_idx])
            current_label = int(labels[face_index])
            if dominant_label == current_label or dominant_count < 2:
                continue
            current_distance = float(np.sum((face_colors_f[face_index] - palette_f[current_label]) ** 2))
            dominant_distance = float(np.sum((face_colors_f[face_index] - palette_f[dominant_label]) ** 2))
            if dominant_distance <= current_distance * 1.2:
                updated[face_index] = dominant_label
        labels = updated
    return labels


def build_region_first_face_palette(
    *,
    positions: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
    max_colors: int,
) -> dict[str, np.ndarray | int]:
    face_colors = np.asarray(face_colors, dtype=np.uint8)
    faces = np.asarray(faces, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.float32)

    region_labels = build_face_regions(face_colors, faces, positions)
    region_count = int(region_labels.max()) + 1 if len(region_labels) else 0
    region_colors = average_by_cluster(face_colors.astype(np.float32), region_labels, region_count)
    region_weights = np.bincount(region_labels, minlength=region_count).astype(np.float32) if region_count else np.zeros((0,), dtype=np.float32)
    region_centroids = average_by_cluster(face_centroids(positions, faces), region_labels, region_count)
    region_centroids = normalize_positions(region_centroids)
    region_normals = average_by_cluster(compute_face_normals(positions, faces), region_labels, region_count)
    region_normals = np.clip(region_normals, -1.0, 1.0)

    effective_max_colors = min(int(max_colors), 12 if region_count > 512 else max(8, min(14, region_count or int(max_colors))))
    feature_vectors = np.concatenate(
        [
            region_colors / 255.0 * 1.35,
            region_centroids * 0.85,
            region_normals * 0.35,
        ],
        axis=1,
    ) if region_count else np.zeros((0, 9), dtype=np.float32)
    semantic_group_labels = weighted_feature_kmeans_labels(feature_vectors, region_weights, max_groups=effective_max_colors)
    semantic_group_count = int(semantic_group_labels.max()) + 1 if len(semantic_group_labels) else 0
    semantic_group_colors = weighted_average_by_cluster(region_colors, semantic_group_labels, region_weights, semantic_group_count)
    semantic_group_weights = np.bincount(semantic_group_labels, weights=region_weights, minlength=semantic_group_count).astype(np.float32) if semantic_group_count else np.zeros((0,), dtype=np.float32)
    palette, semantic_palette_indices = weighted_kmeans_palette(semantic_group_colors, semantic_group_weights, max_colors=effective_max_colors)
    region_palette_indices = semantic_palette_indices[semantic_group_labels] if semantic_group_count else np.zeros((0,), dtype=np.int32)
    face_palette_indices = region_palette_indices[region_labels] if region_count else np.zeros((0,), dtype=np.int32)
    face_palette_indices = merge_small_palette_islands(face_palette_indices, face_colors, palette, faces)
    face_palette_indices = smooth_face_palette_indices(face_palette_indices, face_colors, palette, faces)
    palette, face_palette_indices = compact_palette(palette, face_palette_indices)

    return {
        "palette": palette,
        "face_palette_indices": face_palette_indices,
        "region_count": int(region_count),
        "semantic_group_count": int(semantic_group_count),
        "effective_max_colors": int(effective_max_colors),
    }
