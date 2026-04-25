from __future__ import annotations

from pathlib import Path

import numpy as np

from color3dconverter.export_obj_vertex_colors import write_obj_with_per_vertex_colors, write_obj_with_vertex_colors


def test_write_obj_with_vertex_colors_writes_color_vertices_and_uvs(tmp_path: Path) -> None:
    path = write_obj_with_vertex_colors(
        tmp_path / "vertex_color.obj",
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        face_colors=np.array([[1.0, 0.5, 0.0]], dtype=np.float32),
        texcoords=np.array(
            [
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        object_name="TestMesh",
    )
    text = path.read_text(encoding="utf-8")
    assert "o TestMesh" in text
    assert "v 0.000000 0.000000 0.000000 1.000000 0.500000 0.000000" in text
    assert "vt 0.000000 1.000000" in text
    assert "f 1/1 2/2 3/3" in text


def test_write_obj_with_per_vertex_colors_writes_shared_vertices(tmp_path: Path) -> None:
    path = write_obj_with_per_vertex_colors(
        tmp_path / "per_vertex_color.obj",
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        vertex_colors=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        object_name="PerVertexMesh",
    )
    text = path.read_text(encoding="utf-8")
    assert "o PerVertexMesh" in text
    assert text.count("\nv ") == 3
    assert "v 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000" in text
    assert "v 1.000000 0.000000 0.000000 0.000000 1.000000 0.000000" in text
    assert "f 1 2 3" in text
