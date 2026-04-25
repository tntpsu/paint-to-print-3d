from __future__ import annotations

from pathlib import Path

import numpy as np

from color3dconverter.model_io import LoadedTexturedMesh
from color3dconverter.surface_transfer import (
    barycentric_weights,
    interpolate_triangle_colors,
    transfer_face_colors_nearest_surface,
    transfer_face_colors_raycast,
)


def _simple_loaded_mesh(tmp_path: Path) -> LoadedTexturedMesh:
    return LoadedTexturedMesh(
        mesh=None,
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        texcoords=np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        texture_rgb=np.array(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [255, 255, 255]],
            ],
            dtype=np.uint8,
        ),
        source_path=tmp_path / "simple.glb",
        texture_path=None,
        source_format="glb",
    )


def test_barycentric_weights_sum_to_one() -> None:
    triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    point = np.array([0.25, 0.25, 0.0], dtype=np.float32)
    weights = barycentric_weights(point, triangle)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= -1e-5)


def test_interpolate_triangle_colors_weighted() -> None:
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.float32)
    weights = np.array([0.5, 0.25, 0.25], dtype=np.float32)
    out = interpolate_triangle_colors(colors, weights)
    assert np.allclose(out, [127.5, 63.75, 63.75], atol=1e-3)


def test_transfer_face_colors_nearest_surface_identity(tmp_path: Path) -> None:
    loaded = _simple_loaded_mesh(tmp_path)
    face_points = np.array([[1.0 / 3.0, 1.0 / 3.0, 0.0]], dtype=np.float32)
    colors = transfer_face_colors_nearest_surface(
        source_loaded=loaded,
        target_face_points=face_points,
        candidate_count=1,
        sampling_mode="nearest",
    )
    assert colors.shape == (1, 3)
    assert np.all(colors[0] >= 0)
    assert np.all(colors[0] <= 255)


def test_transfer_face_colors_raycast_identity(tmp_path: Path) -> None:
    loaded = _simple_loaded_mesh(tmp_path)
    face_points = np.array([[1.0 / 3.0, 1.0 / 3.0, 0.25]], dtype=np.float32)
    face_normals = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
    colors = transfer_face_colors_raycast(
        source_loaded=loaded,
        target_face_points=face_points,
        target_face_normals=face_normals,
        candidate_count=1,
        sampling_mode="nearest",
    )
    assert colors.shape == (1, 3)
    assert np.all(colors[0] >= 0)
    assert np.all(colors[0] <= 255)
