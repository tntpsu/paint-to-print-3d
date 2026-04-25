from __future__ import annotations

import numpy as np

from color3dconverter.regions import assign_faces_to_texture_regions, build_texture_regions


def test_assign_faces_to_texture_regions_returns_expected_labels() -> None:
    texture = np.array(
        [
            [[255, 0, 0], [255, 0, 0]],
            [[0, 255, 0], [0, 255, 0]],
        ],
        dtype=np.uint8,
    )
    region_labels = np.array([[0, 0], [1, 1]], dtype=np.int32)
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    texcoords = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    face_labels = assign_faces_to_texture_regions(faces, texcoords, region_labels)

    assert face_labels.shape == (2,)
    assert set(face_labels.tolist()) <= {0, 1}
    palette_labels, palette = build_texture_regions(texture, n_regions=2)
    assert palette_labels.shape == (2, 2)
    assert palette.shape == (2, 3)
