from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, distance_transform_edt


def build_uv_island_mask(
    texture_shape: tuple[int, int] | tuple[int, int, int],
    texcoords: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    height = int(texture_shape[0])
    width = int(texture_shape[1])
    if height <= 0 or width <= 0:
        return np.zeros((0, 0), dtype=bool)
    if len(faces) == 0 or len(texcoords) == 0:
        return np.zeros((height, width), dtype=bool)
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    uv = np.asarray(texcoords, dtype=np.float32)
    uv = uv - np.floor(uv)
    face_indices = np.asarray(faces, dtype=np.int64)
    for face in face_indices:
        polygon: list[tuple[float, float]] = []
        for vertex_index in face.tolist():
            u, v = uv[int(vertex_index)]
            x = float(np.clip(u * (width - 1), 0.0, width - 1))
            y = float(np.clip((1.0 - v) * (height - 1), 0.0, height - 1))
            polygon.append((x, y))
        draw.polygon(polygon, fill=1)
    return np.asarray(mask_image, dtype=np.uint8) > 0


def seam_pad_texture(
    texture_rgb: np.ndarray,
    texcoords: np.ndarray,
    faces: np.ndarray,
    *,
    pad_pixels: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    texture = np.asarray(texture_rgb, dtype=np.uint8)
    mask = build_uv_island_mask(texture.shape, texcoords, faces)
    if texture.size == 0 or mask.size == 0:
        return texture.copy(), mask, {"pad_pixels": int(pad_pixels), "valid_uv_pixels": 0, "padded_uv_pixels": 0}
    if not np.any(mask) or pad_pixels <= 0:
        valid_pixel_count = int(mask.sum())
        return texture.copy(), mask, {"pad_pixels": int(max(pad_pixels, 0)), "valid_uv_pixels": valid_pixel_count, "padded_uv_pixels": valid_pixel_count}

    dilated = binary_dilation(mask, iterations=int(pad_pixels))
    gutter = np.logical_and(dilated, np.logical_not(mask))
    padded = texture.copy()
    nearest_indices = distance_transform_edt(np.logical_not(mask), return_distances=False, return_indices=True)
    gutter_y, gutter_x = np.nonzero(gutter)
    source_y = nearest_indices[0][gutter_y, gutter_x]
    source_x = nearest_indices[1][gutter_y, gutter_x]
    padded[gutter_y, gutter_x] = texture[source_y, source_x]
    padded_mask = np.logical_or(mask, gutter)
    return padded, padded_mask, {
        "pad_pixels": int(pad_pixels),
        "valid_uv_pixels": int(mask.sum()),
        "padded_uv_pixels": int(padded_mask.sum()),
    }


def sample_texture_bilinear(texture_rgb: np.ndarray, texcoords: np.ndarray) -> np.ndarray:
    if texcoords.size == 0:
        return np.full((0, 3), 255, dtype=np.uint8)
    texture = np.asarray(texture_rgb, dtype=np.float32)
    height, width = texture.shape[:2]
    uv = np.asarray(texcoords, dtype=np.float32)
    uv = uv - np.floor(uv)
    x = np.clip(uv[:, 0] * (width - 1), 0.0, width - 1)
    y = np.clip((1.0 - uv[:, 1]) * (height - 1), 0.0, height - 1)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)

    wx = (x - x0).astype(np.float32)[:, None]
    wy = (y - y0).astype(np.float32)[:, None]

    top_left = texture[y0, x0]
    top_right = texture[y0, x1]
    bottom_left = texture[y1, x0]
    bottom_right = texture[y1, x1]

    top = top_left * (1.0 - wx) + top_right * wx
    bottom = bottom_left * (1.0 - wx) + bottom_right * wx
    sampled = top * (1.0 - wy) + bottom * wy
    return np.clip(np.rint(sampled), 0, 255).astype(np.uint8)


def collapse_vertex_colors_by_position(
    positions: np.ndarray,
    vertex_colors: np.ndarray,
    *,
    position_decimals: int = 6,
) -> tuple[np.ndarray, dict[str, Any]]:
    points = np.asarray(positions, dtype=np.float32)
    colors = np.asarray(vertex_colors, dtype=np.float32)
    if len(points) == 0 or len(colors) == 0:
        return np.asarray(vertex_colors, dtype=np.uint8).copy(), {
            "collapsed_group_count": 0,
            "max_group_size": 0,
            "position_decimals": int(position_decimals),
        }
    rounded = np.round(points, decimals=int(position_decimals))
    groups: dict[tuple[float, float, float], list[int]] = {}
    for index, point in enumerate(rounded.tolist()):
        key = tuple(float(value) for value in point)
        groups.setdefault(key, []).append(int(index))
    collapsed = colors.copy()
    max_group_size = 0
    for group_indices in groups.values():
        max_group_size = max(max_group_size, len(group_indices))
        if len(group_indices) <= 1:
            continue
        mean_color = colors[group_indices].mean(axis=0)
        collapsed[group_indices] = mean_color
    return np.clip(np.rint(collapsed), 0, 255).astype(np.uint8), {
        "collapsed_group_count": int(sum(1 for group in groups.values() if len(group) > 1)),
        "max_group_size": int(max_group_size),
        "position_decimals": int(position_decimals),
    }


def bake_texture_to_corner_colors(
    texture_rgb: np.ndarray,
    texcoords: np.ndarray,
    faces: np.ndarray,
    *,
    pad_pixels: int = 4,
    sampling_mode: str = "bilinear",
) -> tuple[np.ndarray, dict[str, Any]]:
    if sampling_mode not in {"bilinear", "nearest"}:
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}")
    face_array = np.asarray(faces, dtype=np.int64)
    if len(face_array) == 0:
        return np.zeros((0, 3, 3), dtype=np.uint8), {
            "sampling_mode": sampling_mode,
            "corner_count": 0,
            "face_count": 0,
            "pad_pixels": int(pad_pixels),
        }
    padded_texture, padded_mask, seam_info = seam_pad_texture(
        texture_rgb,
        texcoords,
        faces,
        pad_pixels=pad_pixels,
    )
    corner_uvs = np.asarray(texcoords, dtype=np.float32)[face_array].reshape(-1, 2)
    if sampling_mode == "bilinear":
        corner_colors = sample_texture_bilinear(padded_texture, corner_uvs)
    else:
        from .face_regions import sample_texture

        corner_colors = sample_texture(padded_texture, corner_uvs)
    reshaped = corner_colors.reshape(len(face_array), 3, 3)
    return reshaped, {
        "sampling_mode": sampling_mode,
        "corner_count": int(len(corner_uvs)),
        "face_count": int(len(face_array)),
        "padded_mask_coverage": float(padded_mask.mean()) if padded_mask.size else 0.0,
        **seam_info,
    }


def face_colors_from_corner_colors(corner_colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(corner_colors, dtype=np.uint8)
    if len(colors) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    face_colors = np.zeros((len(colors), 3), dtype=np.uint8)
    for index, triplet in enumerate(colors):
        unique, counts = np.unique(np.asarray(triplet, dtype=np.uint8), axis=0, return_counts=True)
        face_colors[index] = np.asarray(unique[int(np.argmax(counts))], dtype=np.uint8)
    return face_colors


def bake_texture_to_vertex_colors(
    texture_rgb: np.ndarray,
    texcoords: np.ndarray,
    faces: np.ndarray,
    *,
    pad_pixels: int = 4,
    sampling_mode: str = "bilinear",
) -> tuple[np.ndarray, dict[str, Any]]:
    if sampling_mode not in {"bilinear", "nearest"}:
        raise ValueError(f"Unsupported sampling mode: {sampling_mode}")
    padded_texture, padded_mask, seam_info = seam_pad_texture(
        texture_rgb,
        texcoords,
        faces,
        pad_pixels=pad_pixels,
    )
    if sampling_mode == "bilinear":
        baked = sample_texture_bilinear(padded_texture, texcoords)
    else:
        from .face_regions import sample_texture

        baked = sample_texture(padded_texture, texcoords)
    metadata = {
        "sampling_mode": sampling_mode,
        "texture_width": int(np.asarray(texture_rgb).shape[1]) if np.asarray(texture_rgb).ndim >= 2 else 0,
        "texture_height": int(np.asarray(texture_rgb).shape[0]) if np.asarray(texture_rgb).ndim >= 2 else 0,
        "vertex_count": int(len(np.asarray(texcoords))),
        "has_uv_faces": bool(len(np.asarray(faces)) > 0),
        "padded_mask_coverage": float(padded_mask.mean()) if padded_mask.size else 0.0,
        **seam_info,
    }
    return baked, metadata
