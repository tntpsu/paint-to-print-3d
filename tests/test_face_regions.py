from __future__ import annotations

import numpy as np

from color3dconverter.face_regions import (
    build_connected_face_components,
    refine_face_labels_with_graph_smoothing,
    transfer_face_region_ownership,
)


def test_build_connected_face_components_splits_disconnected_same_label_regions() -> None:
    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
            [4, 5, 6],
        ],
        dtype=np.int64,
    )
    face_labels = np.array([0, 0, 0], dtype=np.int32)
    components = build_connected_face_components(face_labels, faces)
    assert components.tolist() == [0, 0, 1]


def test_transfer_face_region_ownership_preserves_source_face_regions() -> None:
    source_positions = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    source_faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],
            [1, 4, 3],
            [4, 5, 3],
        ],
        dtype=np.int64,
    )
    source_face_labels = np.array([0, 0, 1, 1], dtype=np.int32)

    target_positions = np.array(
        [
            [-1.0, 0.0, 0.05],
            [0.0, 0.0, 0.05],
            [-1.0, 1.0, 0.05],
            [0.0, 1.0, 0.05],
            [1.0, 0.0, 0.05],
            [1.0, 1.0, 0.05],
        ],
        dtype=np.float32,
    )
    target_faces = source_faces.copy()

    transferred = transfer_face_region_ownership(
        source_positions=source_positions,
        source_faces=source_faces,
        source_face_labels=source_face_labels,
        target_positions=target_positions,
        target_faces=target_faces,
        neighbors=4,
        chunk_size=2,
    )

    assert transferred["source_component_count"] == 2
    assert np.asarray(transferred["target_face_labels"]).tolist() == [0, 0, 1, 1]


def test_transfer_face_region_ownership_can_return_label_scores() -> None:
    source_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    source_faces = np.array([[0, 1, 2]], dtype=np.int64)
    source_face_labels = np.array([2], dtype=np.int32)

    transferred = transfer_face_region_ownership(
        source_positions=source_positions,
        source_faces=source_faces,
        source_face_labels=source_face_labels,
        target_positions=source_positions,
        target_faces=source_faces,
        return_label_scores=True,
    )

    scores = np.asarray(transferred["target_label_scores"])
    assert scores.shape == (1, 3)
    assert int(np.argmax(scores[0])) == 2


def test_refine_face_labels_with_graph_smoothing_uses_neighbor_support() -> None:
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [2, 1, 3],
            [0, 2, 4],
        ],
        dtype=np.int64,
    )
    face_labels = np.array([0, 1, 1], dtype=np.int32)
    label_scores = np.array(
        [
            [0.51, 0.49],
            [0.1, 0.9],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )

    refined = refine_face_labels_with_graph_smoothing(
        face_labels,
        label_scores,
        faces,
        positions,
        iterations=4,
        smoothness_weight=0.6,
    )

    assert refined.tolist() == [1, 1, 1]
