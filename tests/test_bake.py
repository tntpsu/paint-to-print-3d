from __future__ import annotations

import numpy as np

from color3dconverter.bake import (
    bake_texture_to_corner_colors,
    bake_texture_to_vertex_colors,
    build_uv_island_mask,
    collapse_vertex_colors_by_position,
    face_colors_from_corner_colors,
    seam_pad_texture,
)


def test_build_uv_island_mask_marks_triangle_pixels() -> None:
    texcoords = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    mask = build_uv_island_mask((8, 8, 3), texcoords, faces)
    assert mask.shape == (8, 8)
    assert int(mask.sum()) > 0


def test_seam_pad_texture_expands_valid_pixels() -> None:
    texture = np.zeros((8, 8, 3), dtype=np.uint8)
    texture[:4, :4] = np.array([255, 0, 0], dtype=np.uint8)
    texcoords = np.array(
        [
            [0.0, 1.0],
            [0.5, 1.0],
            [0.0, 0.5],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    padded, padded_mask, info = seam_pad_texture(texture, texcoords, faces, pad_pixels=2)
    assert padded.shape == texture.shape
    assert int(info["padded_uv_pixels"]) > int(info["valid_uv_pixels"])
    assert padded_mask.sum() > build_uv_island_mask(texture.shape, texcoords, faces).sum()


def test_bake_texture_to_vertex_colors_bilinear_preserves_palette() -> None:
    texture = np.zeros((16, 16, 3), dtype=np.uint8)
    texture[:, :8] = np.array([255, 0, 0], dtype=np.uint8)
    texture[:, 8:] = np.array([0, 0, 255], dtype=np.uint8)
    texcoords = np.array(
        [
            [0.10, 0.80],
            [0.90, 0.80],
            [0.10, 0.20],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    baked, metadata = bake_texture_to_vertex_colors(texture, texcoords, faces, pad_pixels=2, sampling_mode="bilinear")
    assert baked.shape == (3, 3)
    assert tuple(baked[0].tolist()) == (255, 0, 0)
    assert baked[1, 2] > 200
    assert baked[1, 2] > baked[1, 0]
    assert tuple(baked[2].tolist()) == (255, 0, 0)
    assert metadata["sampling_mode"] == "bilinear"
    assert metadata["padded_uv_pixels"] >= metadata["valid_uv_pixels"]


def test_bake_texture_to_corner_colors_returns_face_corner_shape() -> None:
    texture = np.zeros((16, 16, 3), dtype=np.uint8)
    texture[:, :8] = np.array([255, 0, 0], dtype=np.uint8)
    texture[:, 8:] = np.array([0, 0, 255], dtype=np.uint8)
    texcoords = np.array(
        [
            [0.10, 0.80],
            [0.90, 0.80],
            [0.10, 0.20],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    corner_colors, metadata = bake_texture_to_corner_colors(texture, texcoords, faces, pad_pixels=2, sampling_mode="bilinear")
    assert corner_colors.shape == (1, 3, 3)
    assert metadata["corner_count"] == 3
    face_colors = face_colors_from_corner_colors(corner_colors)
    assert face_colors.shape == (1, 3)


def test_collapse_vertex_colors_by_position_blends_shared_positions() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    colors = np.array(
        [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ],
        dtype=np.uint8,
    )
    collapsed, metadata = collapse_vertex_colors_by_position(positions, colors)
    assert tuple(collapsed[0].tolist()) == tuple(collapsed[2].tolist())
    assert metadata["collapsed_group_count"] == 1
