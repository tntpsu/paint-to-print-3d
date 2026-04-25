from __future__ import annotations

from pathlib import Path

import numpy as np

from color3dconverter.paint_cleanup import cleanup_paint_region_labels, paint_component_metrics
from color3dconverter.pipeline import write_labeled_mesh_to_assets


def _grid_mesh(grid_size: int = 7) -> tuple[np.ndarray, np.ndarray]:
    positions: list[list[float]] = []
    for y in range(grid_size + 1):
        for x in range(grid_size + 1):
            positions.append([float(x), float(y), 0.0])

    faces: list[list[int]] = []
    for y in range(grid_size):
        for x in range(grid_size):
            v0 = y * (grid_size + 1) + x
            v1 = v0 + 1
            v2 = v0 + (grid_size + 1)
            v3 = v2 + 1
            faces.append([v0, v1, v3])
            faces.append([v0, v3, v2])
    return np.asarray(positions, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def test_cleanup_paint_region_labels_absorbs_noise_but_keeps_protected_detail() -> None:
    positions, faces = _grid_mesh()
    labels = np.zeros((len(faces),), dtype=np.int32)
    noise_indexes = np.array([0, 10, 20, 32, 46, 58, 70, 82], dtype=np.int64)
    labels[noise_indexes] = 1
    labels[84:86] = 2
    palette = np.array(
        [
            [230, 190, 52],
            [78, 92, 130],
            [245, 245, 238],
        ],
        dtype=np.uint8,
    )

    before = paint_component_metrics(labels, faces)
    cleaned, report = cleanup_paint_region_labels(
        face_labels=labels,
        palette=palette,
        positions=positions,
        faces=faces,
        min_component_size=4,
        max_passes=4,
        protected_labels={2},
        enable_semantic_protection=False,
    )
    after = paint_component_metrics(cleaned, faces)

    assert report["status"] == "improved"
    assert after["component_count"] < before["component_count"]
    assert after["tiny_island_count"] < before["tiny_island_count"]
    assert not np.any(cleaned[noise_indexes] == 1)
    assert np.all(cleaned[84:86] == 2)


def test_write_labeled_mesh_to_assets_uses_cleaned_labels(tmp_path: Path) -> None:
    positions, faces = _grid_mesh(grid_size=3)
    labels = np.zeros((len(faces),), dtype=np.int32)
    labels[0] = 1
    palette = np.array([[240, 180, 50], [20, 80, 220]], dtype=np.uint8)
    cleaned, _ = cleanup_paint_region_labels(
        face_labels=labels,
        palette=palette,
        positions=positions,
        faces=faces,
        min_component_size=3,
        max_passes=2,
        enable_semantic_protection=False,
        max_absorb_face_share=0.20,
    )

    report = write_labeled_mesh_to_assets(
        positions=positions,
        faces=faces,
        face_labels=cleaned,
        palette=palette,
        source_path=tmp_path / "source.glb",
        out_dir=tmp_path / "labeled",
        strategy="paint_region_cleanup",
    )

    assert Path(report["obj_path"]).exists()
    assert Path(report["threemf_path"]).exists()
    assert report["strategy"] == "paint_region_cleanup"
    assert report["component_count"] == 1
    assert report["tiny_island_count"] == 0
