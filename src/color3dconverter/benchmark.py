from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .bake import (
    bake_texture_to_corner_colors,
    bake_texture_to_vertex_colors,
    collapse_vertex_colors_by_position,
    face_colors_from_corner_colors,
)
from .color_adjustments import (
    apply_brightness_contrast,
    apply_hue_saturation,
    apply_levels,
    posterize,
    remap,
)
from .face_regions import build_connected_face_components
from .fixtures import BenchmarkFixture, list_benchmark_fixtures, load_benchmark_fixture
from .model_io import LoadedTexturedMesh, load_textured_model
from .pipeline import (
    convert_color_transferred_mesh_to_assets,
    convert_model_to_color_assets,
    convert_loaded_mesh_to_color_assets,
)
from .validation import write_bambu_validation_bundle, write_source_export_comparison


@dataclass
class BenchmarkLaneResult:
    fixture_name: str
    lane: str
    strategy: str
    report_path: str
    preview_path: str
    comparison_path: str
    face_accuracy: float
    palette_size: int
    component_count: int
    tiny_island_count: int
    largest_component_share: float
    score: float
    passed: bool


@dataclass
class CurvedTransferExperimentResult:
    fixture_name: str
    case_name: str
    strategy: str
    report_path: str
    preview_path: str
    comparison_path: str
    face_accuracy: float
    palette_size: int
    component_count: int
    tiny_island_count: int
    largest_component_share: float
    score: float
    passed: bool


@dataclass
class SurfaceBakeExperimentResult:
    experiment_name: str
    fixture_name: str
    representation: str
    sampling_mode: str
    preview_path: str
    comparison_path: str
    face_accuracy: float
    unique_face_color_count: int
    preserved_dark_face_count: int
    expected_dark_face_count: int
    passed: bool


@dataclass
class RealCaseAblationResult:
    case_name: str
    variant_label: str
    source_path: str
    strategy: str
    n_regions: int
    report_path: str
    preview_path: str
    vertex_color_obj_path: str
    comparison_path: str
    mean_pixel_drift: float
    assessment: str


@dataclass
class IterativeSearchRound:
    round_index: int
    candidate_count: int
    best_variant_label: str
    best_mean_pixel_drift: float
    improved_best: bool
    round_dir: str


@dataclass
class IterativeSearchBest:
    variant_label: str
    strategy: str
    n_regions: int
    texture_transform: dict[str, Any]
    mean_pixel_drift: float
    assessment: str
    report_path: str
    preview_path: str
    comparison_path: str


@dataclass
class CrossCaseSearchRound:
    round_index: int
    candidate_count: int
    best_variant_label: str
    best_fail_count: int
    best_max_drift: float
    best_mean_drift: float
    improved_best: bool
    round_dir: str


@dataclass
class CrossCaseSearchBest:
    variant_label: str
    strategy: str
    n_regions: int
    texture_transform: dict[str, Any]
    fail_count: int
    pass_count: int
    max_drift: float
    mean_drift: float
    case_results: list[dict[str, Any]]


def _write_face_color_preview(
    path: Path,
    positions: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
) -> None:
    import math

    if len(positions) == 0 or len(faces) == 0:
        Image.new("RGB", (960, 720), (245, 241, 234)).save(path)
        return
    points = np.asarray(positions, dtype=np.float32)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extents = np.maximum(bbox_max - bbox_min, 1e-6)
    scale = float(np.max(extents))
    normalized = (points - center) / scale
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
    face_points = projected[np.asarray(faces, dtype=np.int64)]
    face_depth = transformed[np.asarray(faces, dtype=np.int64)][:, :, 2].mean(axis=1)
    draw_order = np.argsort(face_depth)
    image = Image.new("RGB", (width, height), (244, 240, 233))
    draw = ImageDraw.Draw(image, "RGBA")
    colors = np.asarray(face_colors, dtype=np.uint8)
    for face_index in draw_order.tolist():
        polygon = face_points[face_index]
        color = tuple(int(channel) for channel in colors[face_index].tolist()) + (255,)
        outline = tuple(int(channel) for channel in (colors[face_index].astype(np.float32) * 0.72).clip(0, 255).astype(np.uint8).tolist()) + (220,)
        draw.polygon([tuple(point.tolist()) for point in polygon], fill=color, outline=outline)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _sample_triangle_texture_colors(texture_rgb: np.ndarray, texcoords: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if texcoords.size == 0 or len(faces) == 0:
        return np.full((len(faces), 3), 255, dtype=np.uint8)
    face_uv = np.asarray(texcoords[faces], dtype=np.float32)
    weights = np.array(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.60, 0.20, 0.20],
            [0.20, 0.60, 0.20],
            [0.20, 0.20, 0.60],
        ],
        dtype=np.float32,
    )
    samples = np.tensordot(weights, face_uv, axes=(1, 1))
    samples = np.transpose(samples, (1, 0, 2))
    samples = samples - np.floor(samples)
    height, width = texture_rgb.shape[:2]
    sample_x = np.clip(np.rint(samples[:, :, 0] * (width - 1)).astype(np.int64), 0, width - 1)
    sample_y = np.clip(np.rint((1.0 - samples[:, :, 1]) * (height - 1)).astype(np.int64), 0, height - 1)
    sampled = texture_rgb[sample_y, sample_x]
    return np.clip(np.rint(sampled.mean(axis=1)), 0, 255).astype(np.uint8)


def _write_export_preview(
    path: Path,
    positions: np.ndarray,
    faces: np.ndarray,
    palette: np.ndarray,
    triangle_palette_indices: np.ndarray,
) -> None:
    import math

    if len(positions) == 0 or len(faces) == 0:
        Image.new("RGB", (960, 720), (245, 241, 234)).save(path)
        return

    centered = np.asarray(positions, dtype=np.float32, copy=False)
    bbox_min = centered.min(axis=0)
    bbox_max = centered.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extents = np.maximum(bbox_max - bbox_min, 1e-6)
    scale = float(np.max(extents))
    normalized = (centered - center) / scale

    angle_y = math.radians(-32.0)
    angle_x = math.radians(20.0)
    rot_y = np.array(
        [
            [math.cos(angle_y), 0.0, math.sin(angle_y)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle_y), 0.0, math.cos(angle_y)],
        ],
        dtype=np.float32,
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle_x), -math.sin(angle_x)],
            [0.0, math.sin(angle_x), math.cos(angle_x)],
        ],
        dtype=np.float32,
    )
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
    usable_w = width - margin * 2.0
    usable_h = height - margin * 2.0
    scale_2d = min(usable_w / float(span[0]), usable_h / float(span[1]))
    projected = (projected - (min_xy + max_xy) / 2.0) * scale_2d
    projected[:, 0] += width / 2.0
    projected[:, 1] += height / 2.0

    face_points = projected[np.asarray(faces, dtype=np.int64)]
    face_depth = transformed[np.asarray(faces, dtype=np.int64)][:, :, 2].mean(axis=1)
    face_vertices = transformed[np.asarray(faces, dtype=np.int64)]
    face_normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
    normal_lengths = np.linalg.norm(face_normals, axis=1)
    valid_normals = normal_lengths > 1e-8
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
        if np.isnan(polygon).any():
            continue
        color = np.asarray(palette[int(triangle_palette_indices[face_index])], dtype=np.float32)
        lit = np.clip(np.rint(color * float(lighting[face_index])), 0, 255).astype(np.uint8)
        outline = tuple(int(channel) for channel in np.clip(lit * 0.72, 0, 255).tolist()) + (220,)
        fill = tuple(int(channel) for channel in lit.tolist()) + (255,)
        draw.polygon([tuple(point.tolist()) for point in polygon], fill=fill, outline=outline)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _write_texture_source_preview(
    path: Path,
    *,
    positions: np.ndarray,
    faces: np.ndarray,
    texcoords: np.ndarray,
    texture_rgb: np.ndarray,
) -> None:
    triangle_colors = _sample_triangle_texture_colors(texture_rgb, texcoords, faces)
    if len(triangle_colors) == 0 or len(faces) == 0:
        Image.new("RGB", (960, 720), (245, 241, 234)).save(path)
        return
    palette, triangle_palette_indices = np.unique(np.asarray(triangle_colors, dtype=np.uint8), axis=0, return_inverse=True)
    _write_export_preview(
        path,
        positions,
        faces,
        np.asarray(palette, dtype=np.uint8),
        np.asarray(triangle_palette_indices, dtype=np.int32),
    )


def _load_report_palette_and_labels(report: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.load(Path(report["face_palette_indices_path"]).expanduser().resolve())
    palette = np.load(Path(report["palette_npy_path"]).expanduser().resolve())
    return np.asarray(palette, dtype=np.uint8), np.asarray(labels, dtype=np.int32)


def _face_accuracy(expected_face_colors: np.ndarray, palette: np.ndarray, labels: np.ndarray, tolerance: int = 16) -> float:
    if len(expected_face_colors) == 0:
        return 1.0
    predicted = np.asarray(palette, dtype=np.uint8)[np.asarray(labels, dtype=np.int32)]
    expected = np.asarray(expected_face_colors, dtype=np.uint8)
    diff = np.abs(predicted.astype(np.int16) - expected.astype(np.int16))
    matches = np.all(diff <= int(tolerance), axis=1)
    return round(float(matches.mean()), 4)


def _labels_from_face_colors(face_colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(face_colors, dtype=np.uint8)
    if len(colors) == 0:
        return np.zeros((0,), dtype=np.int32)
    _, inverse = np.unique(colors, axis=0, return_inverse=True)
    return np.asarray(inverse, dtype=np.int32)


def _component_stats(labels: np.ndarray, faces: np.ndarray) -> tuple[int, int, float]:
    if len(labels) == 0:
        return 0, 0, 0.0
    component_ids = build_connected_face_components(np.asarray(labels, dtype=np.int32), np.asarray(faces, dtype=np.int64))
    if len(component_ids) == 0:
        return 0, 0, 0.0
    component_sizes = np.bincount(component_ids)
    tiny_threshold = max(4, min(64, max(1, len(labels) // 500)))
    tiny_island_count = int(np.sum(component_sizes < tiny_threshold))
    largest_component_share = round(float(component_sizes.max()) / float(max(len(labels), 1)), 4)
    return int(len(component_sizes)), tiny_island_count, largest_component_share


def _face_colors_from_vertex_colors(faces: np.ndarray, vertex_colors: np.ndarray) -> np.ndarray:
    face_array = np.asarray(faces, dtype=np.int64)
    colors = np.asarray(vertex_colors, dtype=np.uint8)
    if len(face_array) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    face_colors = np.zeros((len(face_array), 3), dtype=np.uint8)
    for index, face in enumerate(face_array):
        unique, counts = np.unique(colors[face], axis=0, return_counts=True)
        face_colors[index] = np.asarray(unique[int(np.argmax(counts))], dtype=np.uint8)
    return face_colors


def _face_accuracy_from_colors(predicted_face_colors: np.ndarray, expected_face_colors: np.ndarray, tolerance: int = 16) -> float:
    expected = np.asarray(expected_face_colors, dtype=np.uint8)
    predicted = np.asarray(predicted_face_colors, dtype=np.uint8)
    if len(expected) == 0:
        return 1.0
    diff = np.abs(predicted.astype(np.int16) - expected.astype(np.int16))
    matches = np.all(diff <= int(tolerance), axis=1)
    return round(float(matches.mean()), 4)


def _dark_face_count(face_colors: np.ndarray) -> int:
    colors = np.asarray(face_colors, dtype=np.uint8)
    if len(colors) == 0:
        return 0
    return int(np.sum(np.all(colors <= np.array([48, 48, 48], dtype=np.uint8), axis=1)))


def _face_colors_from_loaded_mesh(loaded: LoadedTexturedMesh) -> np.ndarray:
    if len(loaded.faces) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    face_uv = np.asarray(loaded.texcoords, dtype=np.float32)[np.asarray(loaded.faces, dtype=np.int64)]
    weights = np.array(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.60, 0.20, 0.20],
            [0.20, 0.60, 0.20],
            [0.20, 0.20, 0.60],
        ],
        dtype=np.float32,
    )
    samples = np.tensordot(weights, face_uv, axes=(1, 1))
    samples = np.transpose(samples, (1, 0, 2))
    samples = samples - np.floor(samples)
    height, width = loaded.texture_rgb.shape[:2]
    sample_x = np.clip(np.rint(samples[:, :, 0] * (width - 1)).astype(np.int64), 0, width - 1)
    sample_y = np.clip(np.rint((1.0 - samples[:, :, 1]) * (height - 1)).astype(np.int64), 0, height - 1)
    sampled = loaded.texture_rgb[sample_y, sample_x]
    return np.clip(np.rint(sampled.mean(axis=1)), 0, 255).astype(np.uint8)


def _apply_texture_transform(texture_rgb: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    rgb_image = np.asarray(texture_rgb, dtype=np.float32) / 255.0
    if rgb_image.size == 0:
        return np.zeros_like(texture_rgb, dtype=np.uint8)
    original_shape = rgb_image.shape
    rgb = rgb_image.reshape((-1, 3))

    levels = transform.get("levels") or {}
    if levels:
        rgb = apply_levels(
            rgb,
            float(levels.get("in_black", 0.0)),
            float(levels.get("in_white", 1.0)),
            float(levels.get("gamma", 1.0)),
            float(levels.get("out_black", 0.0)),
            float(levels.get("out_white", 1.0)),
        )

    brightness = float(transform.get("brightness", 0.0))
    contrast = float(transform.get("contrast", 0.0))
    if brightness != 0.0 or contrast != 0.0:
        rgb = apply_brightness_contrast(rgb, brightness=brightness, contrast=contrast)

    hue_shift = transform.get("hue_shift")
    saturation = transform.get("saturation")
    value = transform.get("value")
    if hue_shift is not None or saturation is not None or value is not None:
        rgb = apply_hue_saturation(
            rgb,
            hue_shift=float(hue_shift if hue_shift is not None else 0.5),
            saturation=float(saturation if saturation is not None else 1.0),
            value=float(value if value is not None else 1.0),
        )

    shadow_lift = float(transform.get("shadow_lift", 0.0))
    if shadow_lift > 0.0:
        luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        lifted = remap(luminance, 0.0, 1.0, shadow_lift, 1.0)
        scale = np.divide(lifted, np.maximum(luminance, 1e-6), out=np.ones_like(lifted), where=luminance > 1e-6)
        rgb = np.clip(rgb * scale[:, None], 0.0, 1.0)

    posterize_levels = transform.get("posterize_levels")
    if posterize_levels is not None:
        rgb = posterize(np.clip(rgb, 0.0, 1.0), int(posterize_levels))

    rgb = rgb.reshape(original_shape)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _prune_neutral_texture_transform(transform: dict[str, Any] | None) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    payload = dict(transform or {})
    for key, value in payload.items():
        if key == "levels":
            levels = {str(level_key): float(level_value) for level_key, level_value in dict(value or {}).items()}
            neutral_levels = {
                "in_black": 0.0,
                "in_white": 1.0,
                "gamma": 1.0,
                "out_black": 0.0,
                "out_white": 1.0,
            }
            normalized_levels = {**neutral_levels, **levels}
            if any(abs(float(normalized_levels[level_key]) - neutral_levels[level_key]) > 1e-9 for level_key in neutral_levels):
                cleaned["levels"] = normalized_levels
            continue
        if value is None:
            continue
        if key in {"brightness", "contrast", "shadow_lift", "hue_shift"} and abs(float(value)) <= 1e-9:
            continue
        if key in {"saturation", "value"} and abs(float(value) - 1.0) <= 1e-9:
            continue
        cleaned[str(key)] = float(value) if isinstance(value, (int, float)) else value
    return cleaned


def _normalize_iterative_candidate(candidate: dict[str, Any], *, default_strategy: str, default_n_regions: int) -> dict[str, Any]:
    normalized = {
        "strategy": str(candidate.get("strategy") or default_strategy),
        "n_regions": int(candidate.get("n_regions") or default_n_regions),
        "texture_transform": _prune_neutral_texture_transform(candidate.get("texture_transform")),
    }
    return normalized


def _candidate_signature(candidate: dict[str, Any]) -> str:
    return json.dumps(candidate, sort_keys=True, separators=(",", ":"))


def _candidate_label(candidate: dict[str, Any]) -> str:
    strategy = str(candidate["strategy"]).replace("_", "-")
    transform = dict(candidate.get("texture_transform") or {})
    parts = [strategy, f"r{int(candidate['n_regions'])}"]
    if "posterize_levels" in transform:
        parts.append(f"p{int(transform['posterize_levels'])}")
    if "saturation" in transform:
        parts.append(f"sat{int(round(float(transform['saturation']) * 100))}")
    if "value" in transform:
        parts.append(f"val{int(round(float(transform['value']) * 100))}")
    if "brightness" in transform:
        parts.append(f"b{int(round(float(transform['brightness']) * 1000))}")
    if "contrast" in transform:
        parts.append(f"c{int(round(float(transform['contrast']) * 1000))}")
    if "shadow_lift" in transform:
        parts.append(f"lift{int(round(float(transform['shadow_lift']) * 1000))}")
    if "levels" in transform:
        levels = transform["levels"]
        parts.append(
            "lv"
            + "-".join(
                [
                    str(int(round(float(levels.get("in_black", 0.0)) * 100))),
                    str(int(round(float(levels.get("in_white", 1.0)) * 100))),
                    str(int(round(float(levels.get("gamma", 1.0)) * 100))),
                ]
            )
        )
    digest = hashlib.sha1(_candidate_signature(candidate).encode("utf-8")).hexdigest()[:8]
    raw = "_".join(parts)
    sanitized = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in raw).strip("_")
    return f"{sanitized[:72]}_{digest}"


def _iterative_neighbors(best_candidate: dict[str, Any], search_space: dict[str, Any]) -> list[dict[str, Any]]:
    neighbors: list[dict[str, Any]] = []
    for field in ("strategy", "n_regions"):
        values = list(search_space.get(field) or [])
        for value in values:
            candidate = json.loads(json.dumps(best_candidate))
            candidate[field] = value
            neighbors.append(candidate)

    texture_space = dict(search_space.get("texture_transform") or {})
    transform_keys = [
        "posterize_levels",
        "saturation",
        "value",
        "brightness",
        "contrast",
        "shadow_lift",
        "hue_shift",
        "levels",
    ]
    for key in transform_keys:
        values = list(texture_space.get(key) or [])
        for value in values:
            candidate = json.loads(json.dumps(best_candidate))
            texture_transform = dict(candidate.get("texture_transform") or {})
            if value is None:
                texture_transform.pop(key, None)
            else:
                texture_transform[key] = value
            candidate["texture_transform"] = texture_transform
            neighbors.append(candidate)
    return neighbors


def _cross_case_sort_key(payload: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(payload.get("fail_count") or 0),
        float(payload.get("max_drift") or 1.0),
        float(payload.get("mean_drift") or 1.0),
    )


def _cross_case_meaningful_improvement(
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    *,
    improvement_epsilon: float,
) -> bool:
    if previous is None:
        return True
    current_fail = int(current.get("fail_count") or 0)
    previous_fail = int(previous.get("fail_count") or 0)
    if current_fail < previous_fail:
        return True
    if current_fail > previous_fail:
        return False
    current_max = float(current.get("max_drift") or 1.0)
    previous_max = float(previous.get("max_drift") or 1.0)
    if (previous_max - current_max) > improvement_epsilon:
        return True
    if abs(previous_max - current_max) <= improvement_epsilon:
        current_mean = float(current.get("mean_drift") or 1.0)
        previous_mean = float(previous.get("mean_drift") or 1.0)
        return (previous_mean - current_mean) > improvement_epsilon
    return False


def _lane_score(
    face_accuracy: float,
    tiny_island_count: int,
    palette_size: int,
    expected_palette_size: int,
    component_count: int,
    expected_component_count: int,
    expected_tiny_island_count: int,
) -> float:
    palette_penalty = abs(int(palette_size) - int(expected_palette_size)) * 0.03
    component_penalty = min(abs(int(component_count) - int(expected_component_count)) * 0.01, 0.15)
    extra_tiny_islands = max(0, int(tiny_island_count) - int(expected_tiny_island_count))
    island_penalty = min(float(extra_tiny_islands) * 0.005, 0.25)
    score = face_accuracy - palette_penalty - component_penalty - island_penalty
    return round(score, 4)


def _write_lane_summary(
    output_dir: Path,
    *,
    fixture: BenchmarkFixture,
    lane: str,
    expected_face_colors: np.ndarray,
    report: dict[str, Any],
    pass_threshold: float,
) -> BenchmarkLaneResult:
    palette, labels = _load_report_palette_and_labels(report)
    face_accuracy = _face_accuracy(expected_face_colors, palette, labels)
    component_count, tiny_island_count, largest_component_share = _component_stats(labels, fixture.same_mesh.faces if lane == "same_mesh" else fixture.repaired_mesh.faces)
    expected_labels = _labels_from_face_colors(expected_face_colors)
    expected_component_count, expected_tiny_island_count, _ = _component_stats(
        expected_labels,
        fixture.same_mesh.faces if lane == "same_mesh" else fixture.repaired_mesh.faces,
    )
    expected_preview_path = output_dir / f"{lane}_expected_preview.png"
    expected_positions = fixture.same_mesh.positions if lane == "same_mesh" else fixture.repaired_mesh.positions
    expected_faces = fixture.same_mesh.faces if lane == "same_mesh" else fixture.repaired_mesh.faces
    _write_face_color_preview(expected_preview_path, expected_positions, expected_faces, expected_face_colors)
    comparison = write_source_export_comparison(
        source_preview_path=expected_preview_path,
        export_preview_path=report["preview_path"],
        comparison_path=output_dir / f"{lane}_expected_vs_export.png",
        source_mode=fixture.name,
        simplify_applied=(lane == "repaired_transfer"),
        color_transfer_applied=(lane == "repaired_transfer"),
    )
    expected_palette_size = int(len(np.unique(np.asarray(expected_face_colors, dtype=np.uint8), axis=0)))
    score = _lane_score(
        face_accuracy,
        tiny_island_count,
        len(palette),
        expected_palette_size,
        component_count,
        expected_component_count,
        expected_tiny_island_count,
    )
    return BenchmarkLaneResult(
        fixture_name=fixture.name,
        lane=lane,
        strategy=str(report["strategy"]),
        report_path=str(report["report_path"]),
        preview_path=str(report["preview_path"]),
        comparison_path=str(comparison["comparison_path"]),
        face_accuracy=face_accuracy,
        palette_size=int(len(palette)),
        component_count=component_count,
        tiny_island_count=tiny_island_count,
        largest_component_share=largest_component_share,
        score=score,
        passed=bool(face_accuracy >= float(pass_threshold)),
    )


def choose_preferred_lane(results: list[BenchmarkLaneResult]) -> BenchmarkLaneResult | None:
    if not results:
        return None
    return max(
        results,
        key=lambda item: (
            float(item.score),
            float(item.face_accuracy),
            -int(item.tiny_island_count),
            float(item.largest_component_share),
        ),
    )


def _run_transfer_case(
    *,
    fixture: BenchmarkFixture,
    case_name: str,
    target_mesh,
    expected_face_colors: np.ndarray,
    output_dir: Path,
    strategy: str,
    pass_threshold: float,
) -> CurvedTransferExperimentResult:
    report = convert_color_transferred_mesh_to_assets(
        target_loaded=target_mesh,
        color_source_loaded=fixture.same_mesh,
        out_dir=output_dir,
        max_colors=fixture.suggested_regions,
        strategy=strategy,
        object_name=f"{fixture.name}_{case_name}",
    )
    palette, labels = _load_report_palette_and_labels(report)
    face_accuracy = _face_accuracy(expected_face_colors, palette, labels)
    component_count, tiny_island_count, largest_component_share = _component_stats(labels, target_mesh.faces)
    expected_preview_path = output_dir / f"{case_name}_{strategy}_expected_preview.png"
    _write_face_color_preview(expected_preview_path, target_mesh.positions, target_mesh.faces, expected_face_colors)
    comparison = write_source_export_comparison(
        source_preview_path=expected_preview_path,
        export_preview_path=report["preview_path"],
        comparison_path=output_dir / f"{case_name}_{strategy}_expected_vs_export.png",
        source_mode=f"{fixture.name}:{case_name}",
        simplify_applied=(case_name != "identity"),
        color_transfer_applied=True,
    )
    expected_palette_size = int(len(np.unique(np.asarray(expected_face_colors, dtype=np.uint8), axis=0)))
    expected_labels = _labels_from_face_colors(expected_face_colors)
    expected_component_count, expected_tiny_island_count, _ = _component_stats(expected_labels, target_mesh.faces)
    score = _lane_score(
        face_accuracy,
        tiny_island_count,
        len(palette),
        expected_palette_size,
        component_count,
        expected_component_count,
        expected_tiny_island_count,
    )
    return CurvedTransferExperimentResult(
        fixture_name=fixture.name,
        case_name=case_name,
        strategy=strategy,
        report_path=str(report["report_path"]),
        preview_path=str(report["preview_path"]),
        comparison_path=str(comparison["comparison_path"]),
        face_accuracy=face_accuracy,
        palette_size=int(len(palette)),
        component_count=component_count,
        tiny_island_count=tiny_island_count,
        largest_component_share=largest_component_share,
        score=score,
        passed=bool(face_accuracy >= float(pass_threshold)),
    )


def run_curved_transfer_experiments(
    *,
    out_dir: str | Path,
    fixture_names: list[str] | None = None,
    strategies: list[str] | None = None,
) -> dict[str, Any]:
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_fixtures = fixture_names or ["banded_sphere", "deformed_banded_sphere"]
    selected_strategies = strategies or [
        "geometry_transfer_texture_regions",
        "legacy_face_regions",
        "geometry_transfer_legacy_face_regions_graph",
    ]
    suite_rows: list[dict[str, Any]] = []
    for fixture_name in selected_fixtures:
        fixture = load_benchmark_fixture(fixture_name)
        fixture_dir = output_dir / fixture.name
        fixture_dir.mkdir(parents=True, exist_ok=True)
        case_specs = [
            ("identity", fixture.same_mesh, np.asarray(fixture.expected_same_face_colors, dtype=np.uint8), float(fixture.pass_threshold_same_mesh)),
        ]
        if fixture.repaired_mesh is not None and fixture.expected_repaired_face_colors is not None:
            case_name = "deformation" if "deformed" in fixture.name else "subdivision"
            case_specs.append((case_name, fixture.repaired_mesh, np.asarray(fixture.expected_repaired_face_colors, dtype=np.uint8), float(fixture.pass_threshold_repaired)))
        fixture_summary: dict[str, Any] = {
            "fixture_name": fixture.name,
            "description": fixture.description,
            "cases": [],
        }
        for case_name, target_mesh, expected_face_colors, pass_threshold in case_specs:
            case_dir = fixture_dir / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            case_results: list[CurvedTransferExperimentResult] = []
            for strategy in selected_strategies:
                case_results.append(
                    _run_transfer_case(
                        fixture=fixture,
                        case_name=case_name,
                        target_mesh=target_mesh,
                        expected_face_colors=expected_face_colors,
                        output_dir=case_dir,
                        strategy=strategy,
                        pass_threshold=pass_threshold,
                    )
                )
            best = max(case_results, key=lambda item: (float(item.score), float(item.face_accuracy), -int(item.tiny_island_count)))
            case_summary = {
                "case_name": case_name,
                "results": [asdict(item) for item in case_results],
                "best": asdict(best),
            }
            fixture_summary["cases"].append(case_summary)
            suite_rows.extend(
                [
                    {
                        "fixture_name": fixture.name,
                        "case_name": case_name,
                        **asdict(item),
                    }
                    for item in case_results
                ]
            )
        (fixture_dir / "curved_transfer_summary.json").write_text(json.dumps(fixture_summary, indent=2), encoding="utf-8")
        markdown = [f"# Curved Transfer Experiments: {fixture.name}", "", fixture.description, ""]
        for case in fixture_summary["cases"]:
            markdown.append(f"## {case['case_name'].title()}")
            for item in case["results"]:
                markdown.append(
                    f"- `{item['strategy']}`: accuracy `{item['face_accuracy']:.3f}`, score `{item['score']:.3f}`, tiny islands `{item['tiny_island_count']}`, passed `{item['passed']}`"
                )
            markdown.append(f"- Best: `{case['best']['strategy']}`")
            markdown.append("")
        (fixture_dir / "curved_transfer_summary.md").write_text("\n".join(markdown).strip() + "\n", encoding="utf-8")
    suite = {
        "fixture_count": int(len(selected_fixtures)),
        "fixtures": selected_fixtures,
        "strategies": selected_strategies,
        "results": suite_rows,
    }
    (output_dir / "curved_transfer_suite.json").write_text(json.dumps(suite, indent=2), encoding="utf-8")
    return suite


def run_surface_bake_experiments(
    *,
    out_dir: str | Path,
    experiment_names: list[str] | None = None,
) -> dict[str, Any]:
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_specs = [
        ("01_seam_split_quad_collapsed_nearest", "seam_split_quad", "collapsed_shared_vertex", "nearest"),
        ("02_seam_split_quad_collapsed_bilinear", "seam_split_quad", "collapsed_shared_vertex", "bilinear"),
        ("03_seam_split_quad_corner_bilinear", "seam_split_quad", "corner", "bilinear"),
        ("04_checker_quad_shared_nearest", "checker_quad", "shared_vertex", "nearest"),
        ("05_checker_quad_shared_bilinear", "checker_quad", "shared_vertex", "bilinear"),
        ("06_checker_quad_corner_bilinear", "checker_quad", "corner", "bilinear"),
        ("07_smiley_cube_shared_nearest", "smiley_cube", "shared_vertex", "nearest"),
        ("08_smiley_cube_shared_bilinear", "smiley_cube", "shared_vertex", "bilinear"),
        ("09_smiley_cube_corner_bilinear", "smiley_cube", "corner", "bilinear"),
        ("10_banded_sphere_corner_bilinear", "banded_sphere", "corner", "bilinear"),
    ]
    selected_specs = [
        spec for spec in experiment_specs if not experiment_names or spec[0] in set(experiment_names)
    ]
    results: list[SurfaceBakeExperimentResult] = []
    for experiment_name, fixture_name, representation, sampling_mode in selected_specs:
        fixture = load_benchmark_fixture(fixture_name)
        loaded = fixture.same_mesh
        experiment_dir = output_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        if representation == "corner":
            corner_colors, metadata = bake_texture_to_corner_colors(
                loaded.texture_rgb,
                loaded.texcoords,
                loaded.faces,
                pad_pixels=4,
                sampling_mode=sampling_mode,
            )
            predicted_face_colors = face_colors_from_corner_colors(corner_colors)
        else:
            vertex_colors, metadata = bake_texture_to_vertex_colors(
                loaded.texture_rgb,
                loaded.texcoords,
                loaded.faces,
                pad_pixels=4,
                sampling_mode=sampling_mode,
            )
            if representation == "collapsed_shared_vertex":
                vertex_colors, collapse_metadata = collapse_vertex_colors_by_position(
                    loaded.positions,
                    vertex_colors,
                )
                metadata = {**metadata, **collapse_metadata}
            predicted_face_colors = _face_colors_from_vertex_colors(loaded.faces, vertex_colors)
        preview_path = experiment_dir / "predicted_preview.png"
        _write_face_color_preview(preview_path, loaded.positions, loaded.faces, predicted_face_colors)
        expected_preview_path = experiment_dir / "expected_preview.png"
        expected_face_colors = np.asarray(fixture.expected_same_face_colors, dtype=np.uint8)
        _write_face_color_preview(expected_preview_path, loaded.positions, loaded.faces, expected_face_colors)
        comparison = write_source_export_comparison(
            source_preview_path=expected_preview_path,
            export_preview_path=preview_path,
            comparison_path=experiment_dir / "expected_vs_predicted.png",
            source_mode=f"{fixture.name}:{representation}",
            simplify_applied=False,
            color_transfer_applied=False,
        )
        accuracy = _face_accuracy_from_colors(predicted_face_colors, expected_face_colors)
        result = SurfaceBakeExperimentResult(
            experiment_name=experiment_name,
            fixture_name=fixture.name,
            representation=representation,
            sampling_mode=sampling_mode,
            preview_path=str(preview_path),
            comparison_path=str(comparison["comparison_path"]),
            face_accuracy=accuracy,
            unique_face_color_count=int(len(np.unique(np.asarray(predicted_face_colors, dtype=np.uint8), axis=0))),
            preserved_dark_face_count=_dark_face_count(predicted_face_colors),
            expected_dark_face_count=_dark_face_count(expected_face_colors),
            passed=bool(accuracy >= float(fixture.pass_threshold_same_mesh)),
        )
        summary = {
            **asdict(result),
            "metadata": metadata,
        }
        (experiment_dir / "surface_bake_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        markdown = [
            f"# Surface Bake Experiment: {experiment_name}",
            "",
            f"- fixture: `{fixture.name}`",
            f"- representation: `{representation}`",
            f"- sampling_mode: `{sampling_mode}`",
            f"- face_accuracy: `{accuracy:.4f}`",
            f"- unique_face_color_count: `{summary['unique_face_color_count']}`",
            f"- dark_faces: `{summary['preserved_dark_face_count']}` / `{summary['expected_dark_face_count']}`",
        ]
        (experiment_dir / "surface_bake_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
        results.append(result)
    suite = {
        "experiment_count": int(len(results)),
        "results": [asdict(item) for item in results],
    }
    (output_dir / "surface_bake_suite.json").write_text(json.dumps(suite, indent=2), encoding="utf-8")
    markdown = ["# Surface Bake Suite", ""]
    for item in results:
        markdown.append(
            f"- `{item.experiment_name}`: accuracy `{item.face_accuracy:.4f}`, dark faces `{item.preserved_dark_face_count}/{item.expected_dark_face_count}`, passed `{item.passed}`"
        )
    (output_dir / "surface_bake_suite.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return suite


def run_fixture_benchmark(
    fixture: BenchmarkFixture,
    *,
    out_dir: str | Path,
    same_mesh_strategy: str = "legacy_fast_face_labels",
    repaired_strategy: str = "geometry_transfer_legacy_face_regions",
) -> dict[str, Any]:
    output_dir = Path(out_dir).expanduser().resolve() / fixture.name
    output_dir.mkdir(parents=True, exist_ok=True)
    same_mesh_report = convert_loaded_mesh_to_color_assets(
        fixture.same_mesh,
        out_dir=output_dir / "same_mesh",
        n_regions=fixture.suggested_regions,
        strategy=same_mesh_strategy,
        object_name=fixture.name,
    )
    lane_results = [
        _write_lane_summary(
            output_dir,
            fixture=fixture,
            lane="same_mesh",
            expected_face_colors=np.asarray(fixture.expected_same_face_colors, dtype=np.uint8),
            report=same_mesh_report,
            pass_threshold=fixture.pass_threshold_same_mesh,
        )
    ]
    if fixture.repaired_mesh is not None and fixture.expected_repaired_face_colors is not None:
        repaired_report = convert_color_transferred_mesh_to_assets(
            target_loaded=fixture.repaired_mesh,
            color_source_loaded=fixture.same_mesh,
            out_dir=output_dir / "repaired_transfer",
            max_colors=fixture.suggested_regions,
            strategy=repaired_strategy,
            object_name=f"{fixture.name}_repaired",
        )
        lane_results.append(
            _write_lane_summary(
                output_dir,
                fixture=fixture,
                lane="repaired_transfer",
                expected_face_colors=np.asarray(fixture.expected_repaired_face_colors, dtype=np.uint8),
                report=repaired_report,
                pass_threshold=fixture.pass_threshold_repaired,
            )
        )
    preferred = choose_preferred_lane(lane_results)
    summary = {
        "fixture_name": fixture.name,
        "description": fixture.description,
        "suggested_regions": int(fixture.suggested_regions),
        "lanes": [asdict(item) for item in lane_results],
        "preferred_lane": asdict(preferred) if preferred else None,
    }
    (output_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown = [
        f"# Benchmark: {fixture.name}",
        "",
        fixture.description,
        "",
        "## Lanes",
    ]
    for item in lane_results:
        markdown.extend(
            [
                f"- {item.lane}: strategy `{item.strategy}`, accuracy `{item.face_accuracy:.3f}`, score `{item.score:.3f}`, tiny islands `{item.tiny_island_count}`, passed `{item.passed}`",
                f"  preview: {item.preview_path}",
                f"  comparison: {item.comparison_path}",
            ]
        )
    if preferred:
        markdown.extend(["", "## Preferred Lane", f"- {preferred.lane} via `{preferred.strategy}`"])
    (output_dir / "benchmark_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return summary


def run_benchmark_suite(
    *,
    out_dir: str | Path,
    fixture_names: list[str] | None = None,
    same_mesh_strategy: str = "legacy_fast_face_labels",
    repaired_strategy: str = "geometry_transfer_legacy_face_regions",
) -> dict[str, Any]:
    names = fixture_names or list_benchmark_fixtures()
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for name in names:
        fixture = load_benchmark_fixture(name)
        summaries.append(
            run_fixture_benchmark(
                fixture,
                out_dir=output_dir,
                same_mesh_strategy=same_mesh_strategy,
                repaired_strategy=repaired_strategy,
            )
        )
    suite = {
        "fixture_count": int(len(summaries)),
        "fixtures": summaries,
    }
    (output_dir / "suite_summary.json").write_text(json.dumps(suite, indent=2), encoding="utf-8")
    return suite


def run_real_case_ablation(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
    strategy: str | None = None,
    n_regions: int | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).expanduser().resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    case_name = str(config["case_name"])
    default_source_preview_path = Path(config["source_preview_path"]).expanduser().resolve()
    default_source_mode = str(config.get("source_mode") or "single_image")
    strategy_value = str(strategy or config.get("strategy") or "legacy_fast_face_labels")
    n_regions_value = int(n_regions if n_regions is not None else (config.get("n_regions") or 8))
    output_dir = Path(out_dir).expanduser().resolve() if out_dir else config_file.parent / f"{case_name}_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_exports: list[dict[str, Any]] = []
    probe_exports_source_path = config.get("probe_exports_source_path")
    if probe_exports_source_path:
        probe_source = Path(str(probe_exports_source_path)).expanduser().resolve()
        probe_exports = list(json.loads(probe_source.read_text(encoding="utf-8")).get("probe_exports") or [])

    results: list[RealCaseAblationResult] = []
    variants = list(config.get("variants") or [])
    if not variants:
        raise ValueError("real ablation config must include at least one variant")

    for variant in variants:
        label = str(variant["label"])
        source_path = Path(str(variant["source_path"])).expanduser().resolve()
        variant_source_preview_path = Path(
            str(variant.get("source_preview_path") or default_source_preview_path)
        ).expanduser().resolve()
        variant_source_mode = str(variant.get("source_mode") or default_source_mode)
        variant_strategy = str(variant.get("strategy") or strategy_value)
        variant_n_regions = int(variant.get("n_regions") or n_regions_value)
        variant_probe_exports = probe_exports
        variant_probe_exports_source_path = variant.get("probe_exports_source_path")
        if variant_probe_exports_source_path:
            variant_probe_source = Path(str(variant_probe_exports_source_path)).expanduser().resolve()
            variant_probe_exports = list(json.loads(variant_probe_source.read_text(encoding="utf-8")).get("probe_exports") or [])
        variant_dir = output_dir / label
        texture_transform = variant.get("texture_transform")
        if texture_transform:
            loaded = load_textured_model(source_path)
            transformed_texture = _apply_texture_transform(loaded.texture_rgb, dict(texture_transform))
            (variant_dir / "transformed_texture.png").parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(transformed_texture, mode="RGB").save(variant_dir / "transformed_texture.png")
            loaded = LoadedTexturedMesh(
                mesh=loaded.mesh,
                positions=loaded.positions,
                faces=loaded.faces,
                texcoords=loaded.texcoords,
                texture_rgb=transformed_texture,
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
            transformed_source_preview_path = variant_dir / "transformed_source_preview.png"
            _write_texture_source_preview(
                transformed_source_preview_path,
                positions=loaded.positions,
                faces=loaded.faces,
                texcoords=loaded.texcoords,
                texture_rgb=loaded.texture_rgb,
            )
            variant_source_preview_path = transformed_source_preview_path
            report = convert_loaded_mesh_to_color_assets(
                loaded,
                out_dir=variant_dir,
                n_regions=variant_n_regions,
                strategy=variant_strategy,
                object_name=f"{case_name}_{label}",
                obj_filename="local_bambu_vertex_colors.obj",
                threemf_filename="local_bambu_palette.3mf",
                preview_filename="local_bambu_export_preview.png",
                swatch_filename="local_bambu_palette_swatches.png",
                palette_csv_filename="local_bambu_palette.csv",
                report_filename="local_bambu_converter_report.json",
            )
        else:
            report = convert_model_to_color_assets(
                source_path,
                out_dir=variant_dir,
                n_regions=variant_n_regions,
                strategy=variant_strategy,
                object_name=f"{case_name}_{label}",
                obj_filename="local_bambu_vertex_colors.obj",
                threemf_filename="local_bambu_palette.3mf",
                preview_filename="local_bambu_export_preview.png",
                swatch_filename="local_bambu_palette_swatches.png",
                palette_csv_filename="local_bambu_palette.csv",
                report_filename="local_bambu_converter_report.json",
            )
        if variant_probe_exports:
            comparison = write_bambu_validation_bundle(
                output_dir=variant_dir,
                source_preview_path=variant_source_preview_path,
                export_preview_path=Path(report["preview_path"]),
                threemf_path=Path(report["threemf_path"]),
                obj_path=Path(report["obj_path"]),
                probe_exports=variant_probe_exports,
                source_mode=variant_source_mode,
                simplify_applied=False,
                color_transfer_applied=False,
            )
            mean_pixel_drift = float(comparison["mean_pixel_drift"])
            assessment = str(comparison["assessment"])
            comparison_path = str(comparison["comparison_path"])
        else:
            comparison = write_source_export_comparison(
                source_preview_path=variant_source_preview_path,
                export_preview_path=Path(report["preview_path"]),
                comparison_path=variant_dir / f"{label}_comparison.png",
                source_mode=variant_source_mode,
                simplify_applied=False,
                color_transfer_applied=False,
            )
            mean_pixel_drift = float(comparison["mean_pixel_drift"])
            assessment = str(comparison["assessment"])
            comparison_path = str(comparison["comparison_path"])
        results.append(
            RealCaseAblationResult(
                case_name=case_name,
                variant_label=label,
                source_path=str(source_path),
                strategy=variant_strategy,
                n_regions=variant_n_regions,
                report_path=str(report["report_path"]),
                preview_path=str(report["preview_path"]),
                vertex_color_obj_path=str(report["vertex_color_obj_path"]),
                comparison_path=comparison_path,
                mean_pixel_drift=mean_pixel_drift,
                assessment=assessment,
            )
        )

    results.sort(key=lambda item: float(item.mean_pixel_drift))
    rows = [asdict(item) for item in results]
    summary = {
        "case_name": case_name,
        "strategy": strategy_value,
        "n_regions": n_regions_value,
        "source_preview_path": str(default_source_preview_path),
        "variant_count": int(len(rows)),
        "results": rows,
        "best_variant": rows[0] if rows else None,
    }
    (output_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown = [f"# Real Case Ablation: {case_name}", "", f"- strategy: `{strategy}`", f"- regions: `{n_regions}`", ""]
    for row in rows:
        markdown.append(
            f"- `{row['variant_label']}`: drift `{row['mean_pixel_drift']:.4f}`, assessment `{row['assessment']}`"
        )
    (output_dir / "ablation_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    thumb = (260, 260)
    cols = min(3, max(1, len(rows)))
    rows_count = int(np.ceil(len(rows) / cols))
    board = Image.new("RGB", (cols * (thumb[0] + 16) + 16, rows_count * (thumb[1] + 44) + 16), (245, 241, 234))
    draw = ImageDraw.Draw(board)
    for index, row in enumerate(rows):
        image = Image.open(row["preview_path"]).convert("RGB")
        fitted = ImageOps.fit(image, thumb, method=Image.Resampling.BICUBIC)
        x = 16 + (index % cols) * (thumb[0] + 16)
        y = 16 + (index // cols) * (thumb[1] + 44)
        board.paste(fitted, (x, y + 18))
        draw.text((x, y), row["variant_label"], fill=(30, 30, 30))
        draw.text((x, y + 22 + thumb[1]), f"drift {row['mean_pixel_drift']:.3f}", fill=(30, 30, 30))
    board_path = output_dir / "ablation_board.png"
    board.save(board_path)
    summary["board_path"] = str(board_path)
    (output_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_iterative_real_case_search(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).expanduser().resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    case_name = str(config["case_name"])
    default_strategy = str(config.get("strategy") or "legacy_fast_face_labels")
    default_n_regions = int(config.get("n_regions") or 8)
    output_dir = Path(out_dir).expanduser().resolve() if out_dir else config_file.parent / f"{case_name}_iterative"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_value = float(config.get("target_value", 0.01))
    improvement_epsilon = float(config.get("improvement_epsilon", 0.002))
    patience = max(1, int(config.get("patience", 3)))
    max_iterations = max(1, int(config.get("max_iterations", 10)))
    max_runtime_minutes = max(1.0, float(config.get("max_runtime_minutes", 60.0)))
    search_space = dict(config.get("search_space") or {})

    seed_candidates_raw = list(config.get("seed_candidates") or [])
    if not seed_candidates_raw:
        seed_candidates_raw = [dict(config.get("base_candidate") or {})]
    frontier = [
        _normalize_iterative_candidate(candidate, default_strategy=default_strategy, default_n_regions=default_n_regions)
        for candidate in seed_candidates_raw
    ]

    seen_signatures: set[str] = set()
    rounds: list[IterativeSearchRound] = []
    best_payload: dict[str, Any] | None = None
    no_improve_rounds = 0
    stop_reason = "max_iterations_reached"
    started = perf_counter()

    shared_config: dict[str, Any] = {
        "case_name": case_name,
        "source_preview_path": config["source_preview_path"],
    }
    for optional_key in ("source_mode", "probe_exports_source_path"):
        if config.get(optional_key) is not None:
            shared_config[optional_key] = config.get(optional_key)

    for iteration in range(1, max_iterations + 1):
        elapsed_minutes = (perf_counter() - started) / 60.0
        if elapsed_minutes >= max_runtime_minutes:
            stop_reason = "runtime_limit_reached"
            break

        unique_frontier: list[dict[str, Any]] = []
        for candidate in frontier:
            signature = _candidate_signature(candidate)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            unique_frontier.append(candidate)
        if not unique_frontier:
            stop_reason = "search_exhausted"
            break

        round_dir = output_dir / f"round_{iteration:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        round_variants = []
        candidate_lookup: dict[str, dict[str, Any]] = {}
        for candidate in unique_frontier:
            label = _candidate_label(candidate)
            candidate_lookup[label] = candidate
            variant: dict[str, Any] = {
                "label": label,
                "source_path": config["source_path"],
                "strategy": candidate["strategy"],
                "n_regions": candidate["n_regions"],
            }
            if candidate.get("texture_transform"):
                variant["texture_transform"] = candidate["texture_transform"]
            round_variants.append(variant)

        round_config = {**shared_config, "variants": round_variants}
        round_config_path = round_dir / "round_config.json"
        round_config_path.write_text(json.dumps(round_config, indent=2), encoding="utf-8")

        round_summary = run_real_case_ablation(
            config_path=round_config_path,
            out_dir=round_dir,
        )

        round_results: list[dict[str, Any]] = []
        for result in list(round_summary.get("results") or []):
            candidate = candidate_lookup[str(result["variant_label"])]
            round_results.append(
                {
                    "candidate": candidate,
                    "variant_label": str(result["variant_label"]),
                    "strategy": str(result["strategy"]),
                    "n_regions": int(result["n_regions"]),
                    "texture_transform": dict(candidate.get("texture_transform") or {}),
                    "mean_pixel_drift": float(result["mean_pixel_drift"]),
                    "assessment": str(result["assessment"]),
                    "report_path": str(result["report_path"]),
                    "preview_path": str(result["preview_path"]),
                    "comparison_path": str(result["comparison_path"]),
                }
            )
        if not round_results:
            stop_reason = "no_round_results"
            break

        round_best = min(round_results, key=lambda item: item["mean_pixel_drift"])
        improved = best_payload is None or round_best["mean_pixel_drift"] < float(best_payload["mean_pixel_drift"])
        meaningful_improvement = best_payload is None or (float(best_payload["mean_pixel_drift"]) - round_best["mean_pixel_drift"]) > improvement_epsilon

        if improved:
            best_payload = round_best
        if meaningful_improvement:
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        rounds.append(
            IterativeSearchRound(
                round_index=iteration,
                candidate_count=len(round_results),
                best_variant_label=str(round_best["variant_label"]),
                best_mean_pixel_drift=float(round_best["mean_pixel_drift"]),
                improved_best=bool(improved),
                round_dir=str(round_dir),
            )
        )

        if best_payload is not None and float(best_payload["mean_pixel_drift"]) <= target_value:
            stop_reason = "target_reached"
            break
        if no_improve_rounds >= patience:
            stop_reason = "patience_exhausted"
            break

        frontier = _iterative_neighbors(best_payload["candidate"], search_space) if best_payload is not None else []

    summary = {
        "case_name": case_name,
        "objective_metric": "mean_pixel_drift",
        "target_value": target_value,
        "improvement_epsilon": improvement_epsilon,
        "patience": patience,
        "max_iterations": max_iterations,
        "max_runtime_minutes": max_runtime_minutes,
        "stop_reason": stop_reason,
        "round_count": len(rounds),
        "rounds": [asdict(item) for item in rounds],
        "best_result": asdict(
            IterativeSearchBest(
                variant_label=str(best_payload["variant_label"]),
                strategy=str(best_payload["strategy"]),
                n_regions=int(best_payload["n_regions"]),
                texture_transform=dict(best_payload.get("texture_transform") or {}),
                mean_pixel_drift=float(best_payload["mean_pixel_drift"]),
                assessment=str(best_payload["assessment"]),
                report_path=str(best_payload["report_path"]),
                preview_path=str(best_payload["preview_path"]),
                comparison_path=str(best_payload["comparison_path"]),
            )
        )
        if best_payload is not None
        else None,
    }
    (output_dir / "iterative_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown = [
        f"# Iterative Search: {case_name}",
        "",
        f"- objective_metric: `mean_pixel_drift`",
        f"- target_value: `{target_value}`",
        f"- stop_reason: `{stop_reason}`",
        f"- rounds: `{len(rounds)}`",
    ]
    if best_payload is not None:
        markdown.extend(
            [
                "",
                "## Best Result",
                f"- variant: `{best_payload['variant_label']}`",
                f"- strategy: `{best_payload['strategy']}`",
                f"- n_regions: `{best_payload['n_regions']}`",
                f"- mean_pixel_drift: `{best_payload['mean_pixel_drift']}`",
                f"- assessment: `{best_payload['assessment']}`",
                f"- preview: {best_payload['preview_path']}",
                f"- comparison: {best_payload['comparison_path']}",
            ]
        )
    if rounds:
        markdown.extend(["", "## Rounds"])
        for item in rounds:
            markdown.append(
                f"- round {item.round_index}: `{item.candidate_count}` candidates, best `{item.best_variant_label}` at drift `{item.best_mean_pixel_drift}`"
            )
    (output_dir / "iterative_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return summary


def run_cross_case_iterative_search(
    *,
    config_path: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).expanduser().resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    suite_name = str(config.get("suite_name") or config.get("case_name") or config_file.stem)
    cases = list(config.get("cases") or [])
    if not cases:
        raise ValueError("Cross-case iterative search requires at least one case.")

    default_strategy = str(config.get("strategy") or "legacy_fast_face_labels")
    default_n_regions = int(config.get("n_regions") or 8)
    output_dir = Path(out_dir).expanduser().resolve() if out_dir else config_file.parent / f"{suite_name}_cross_case"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_value = float(config.get("target_value", 0.05))
    improvement_epsilon = float(config.get("improvement_epsilon", 0.002))
    patience = max(1, int(config.get("patience", 3)))
    max_iterations = max(1, int(config.get("max_iterations", 8)))
    max_runtime_minutes = max(1.0, float(config.get("max_runtime_minutes", 60.0)))
    search_space = dict(config.get("search_space") or {})

    seed_candidates_raw = list(config.get("seed_candidates") or [])
    if not seed_candidates_raw:
        seed_candidates_raw = [dict(config.get("base_candidate") or {})]
    frontier = [
        _normalize_iterative_candidate(candidate, default_strategy=default_strategy, default_n_regions=default_n_regions)
        for candidate in seed_candidates_raw
    ]

    seen_signatures: set[str] = set()
    rounds: list[CrossCaseSearchRound] = []
    best_payload: dict[str, Any] | None = None
    no_improve_rounds = 0
    stop_reason = "max_iterations_reached"
    started = perf_counter()

    for iteration in range(1, max_iterations + 1):
        elapsed_minutes = (perf_counter() - started) / 60.0
        if elapsed_minutes >= max_runtime_minutes:
            stop_reason = "runtime_limit_reached"
            break

        unique_frontier: list[dict[str, Any]] = []
        for candidate in frontier:
            signature = _candidate_signature(candidate)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            unique_frontier.append(candidate)
        if not unique_frontier:
            stop_reason = "search_exhausted"
            break

        round_dir = output_dir / f"round_{iteration:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        round_results: list[dict[str, Any]] = []

        for candidate in unique_frontier:
            label = _candidate_label(candidate)
            candidate_dir = round_dir / label
            candidate_dir.mkdir(parents=True, exist_ok=True)
            case_results: list[dict[str, Any]] = []

            for case in cases:
                case_name = str(case["case_name"])
                case_source_path = Path(str(case["source_path"])).expanduser().resolve()
                case_source_preview_path = Path(str(case["source_preview_path"])).expanduser().resolve()
                case_source_mode = str(case.get("source_mode") or "")
                case_probe_exports_source_path = case.get("probe_exports_source_path")
                case_target_value = float(case.get("target_value", target_value))
                case_strategy = str(case.get("strategy") or candidate["strategy"])
                case_n_regions = int(case.get("n_regions") or candidate["n_regions"])
                case_out_dir = candidate_dir / case_name
                case_config_path = candidate_dir / f"{case_name}_config.json"
                case_variant: dict[str, Any] = {
                    "label": label,
                    "source_path": str(case_source_path),
                    "strategy": case_strategy,
                    "n_regions": case_n_regions,
                }
                if candidate.get("texture_transform"):
                    case_variant["texture_transform"] = candidate["texture_transform"]
                case_config: dict[str, Any] = {
                    "case_name": case_name,
                    "source_preview_path": str(case_source_preview_path),
                    "strategy": case_strategy,
                    "n_regions": case_n_regions,
                    "variants": [case_variant],
                }
                if case_source_mode:
                    case_config["source_mode"] = case_source_mode
                if case_probe_exports_source_path is not None:
                    case_config["probe_exports_source_path"] = str(case_probe_exports_source_path)
                case_config_path.write_text(json.dumps(case_config, indent=2), encoding="utf-8")
                case_summary = run_real_case_ablation(config_path=case_config_path, out_dir=case_out_dir)
                case_row = dict((case_summary.get("results") or [])[0])
                case_drift = float(case_row["mean_pixel_drift"])
                case_results.append(
                    {
                        "case_name": case_name,
                        "target_value": case_target_value,
                        "mean_pixel_drift": case_drift,
                        "assessment": str(case_row["assessment"]),
                        "report_path": str(case_row["report_path"]),
                        "preview_path": str(case_row["preview_path"]),
                        "comparison_path": str(case_row["comparison_path"]),
                        "passed": case_drift <= case_target_value,
                    }
                )

            fail_count = int(sum(1 for item in case_results if not bool(item["passed"])))
            pass_count = int(len(case_results) - fail_count)
            max_drift = float(max(float(item["mean_pixel_drift"]) for item in case_results))
            mean_drift = float(sum(float(item["mean_pixel_drift"]) for item in case_results) / float(len(case_results)))
            round_results.append(
                {
                    "candidate": candidate,
                    "variant_label": label,
                    "strategy": str(candidate["strategy"]),
                    "n_regions": int(candidate["n_regions"]),
                    "texture_transform": dict(candidate.get("texture_transform") or {}),
                    "fail_count": fail_count,
                    "pass_count": pass_count,
                    "max_drift": max_drift,
                    "mean_drift": mean_drift,
                    "case_results": case_results,
                }
            )

        if not round_results:
            stop_reason = "no_round_results"
            break

        round_best = min(round_results, key=_cross_case_sort_key)
        improved = best_payload is None or _cross_case_sort_key(round_best) < _cross_case_sort_key(best_payload)
        meaningful_improvement = _cross_case_meaningful_improvement(
            round_best,
            best_payload,
            improvement_epsilon=improvement_epsilon,
        )

        if improved:
            best_payload = round_best
        if meaningful_improvement:
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        rounds.append(
            CrossCaseSearchRound(
                round_index=iteration,
                candidate_count=len(round_results),
                best_variant_label=str(round_best["variant_label"]),
                best_fail_count=int(round_best["fail_count"]),
                best_max_drift=float(round_best["max_drift"]),
                best_mean_drift=float(round_best["mean_drift"]),
                improved_best=bool(improved),
                round_dir=str(round_dir),
            )
        )

        if best_payload is not None and int(best_payload["fail_count"]) == 0 and float(best_payload["max_drift"]) <= target_value:
            stop_reason = "target_reached"
            break
        if no_improve_rounds >= patience:
            stop_reason = "patience_exhausted"
            break

        frontier = _iterative_neighbors(best_payload["candidate"], search_space) if best_payload is not None else []

    summary = {
        "suite_name": suite_name,
        "case_count": len(cases),
        "case_names": [str(case["case_name"]) for case in cases],
        "objective_metric": "cross_case_mean_pixel_drift",
        "target_value": target_value,
        "improvement_epsilon": improvement_epsilon,
        "patience": patience,
        "max_iterations": max_iterations,
        "max_runtime_minutes": max_runtime_minutes,
        "stop_reason": stop_reason,
        "round_count": len(rounds),
        "rounds": [asdict(item) for item in rounds],
        "best_result": asdict(
            CrossCaseSearchBest(
                variant_label=str(best_payload["variant_label"]),
                strategy=str(best_payload["strategy"]),
                n_regions=int(best_payload["n_regions"]),
                texture_transform=dict(best_payload.get("texture_transform") or {}),
                fail_count=int(best_payload["fail_count"]),
                pass_count=int(best_payload["pass_count"]),
                max_drift=float(best_payload["max_drift"]),
                mean_drift=float(best_payload["mean_drift"]),
                case_results=list(best_payload["case_results"]),
            )
        )
        if best_payload is not None
        else None,
    }
    (output_dir / "cross_case_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    markdown = [
        f"# Cross-Case Iterative Search: {suite_name}",
        "",
        f"- cases: `{len(cases)}`",
        f"- target_value: `{target_value}`",
        f"- stop_reason: `{stop_reason}`",
        f"- rounds: `{len(rounds)}`",
    ]
    if best_payload is not None:
        markdown.extend(
            [
                "",
                "## Best Result",
                f"- variant: `{best_payload['variant_label']}`",
                f"- strategy: `{best_payload['strategy']}`",
                f"- n_regions: `{best_payload['n_regions']}`",
                f"- fail_count: `{best_payload['fail_count']}`",
                f"- max_drift: `{best_payload['max_drift']:.4f}`",
                f"- mean_drift: `{best_payload['mean_drift']:.4f}`",
                "",
                "## Case Results",
            ]
        )
        for case_result in best_payload["case_results"]:
            markdown.append(
                f"- `{case_result['case_name']}`: drift `{float(case_result['mean_pixel_drift']):.4f}`, target `{float(case_result['target_value']):.4f}`, assessment `{case_result['assessment']}`"
            )
    if rounds:
        markdown.extend(["", "## Rounds"])
        for item in rounds:
            markdown.append(
                f"- round {item.round_index}: `{item.candidate_count}` candidates, best `{item.best_variant_label}`, fail_count `{item.best_fail_count}`, max_drift `{item.best_max_drift:.4f}`, mean_drift `{item.best_mean_drift:.4f}`"
            )
    (output_dir / "cross_case_summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    if best_payload is not None:
        thumb = (260, 260)
        case_results = list(best_payload["case_results"])
        cols = min(3, max(1, len(case_results)))
        rows_count = int(np.ceil(len(case_results) / cols))
        board = Image.new("RGB", (cols * (thumb[0] + 16) + 16, rows_count * (thumb[1] + 44) + 16), (245, 241, 234))
        draw = ImageDraw.Draw(board)
        for index, case_result in enumerate(case_results):
            image = Image.open(case_result["preview_path"]).convert("RGB")
            fitted = ImageOps.fit(image, thumb, method=Image.Resampling.BICUBIC)
            x = 16 + (index % cols) * (thumb[0] + 16)
            y = 16 + (index // cols) * (thumb[1] + 44)
            board.paste(fitted, (x, y + 18))
            draw.text((x, y), str(case_result["case_name"]), fill=(30, 30, 30))
            draw.text((x, y + 22 + thumb[1]), f"drift {float(case_result['mean_pixel_drift']):.3f}", fill=(30, 30, 30))
        board_path = output_dir / "cross_case_best_board.png"
        board.save(board_path)
        summary["best_board_path"] = str(board_path)
        (output_dir / "cross_case_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary
