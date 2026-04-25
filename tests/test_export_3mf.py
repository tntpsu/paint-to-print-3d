from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np

from color3dconverter.export_3mf import write_colorgroup_3mf


def test_write_colorgroup_3mf_writes_color_group(tmp_path: Path) -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    palette = np.array([[255, 0, 0]], dtype=np.uint8)
    face_labels = np.array([0], dtype=np.int32)

    output_path = write_colorgroup_3mf(tmp_path / "triangle.3mf", positions, faces, palette, face_labels, object_name="Triangle")

    with ZipFile(output_path) as archive:
        xml_text = archive.read("3D/3dmodel.model").decode("utf-8")

    assert "colorgroup" in xml_text.lower()
    assert "#FF0000FF" in xml_text
    assert "Triangle" in xml_text
