from __future__ import annotations

import numpy as np

from color3dconverter.fixtures import list_benchmark_fixtures, load_benchmark_fixture


def test_list_benchmark_fixtures_contains_learning_ladder() -> None:
    names = list_benchmark_fixtures()
    assert "checker_quad" in names
    assert "seam_split_quad" in names
    assert "six_color_cube" in names
    assert "smiley_cube" in names
    assert "banded_sphere" in names
    assert "simple_duck" in names


def test_cube_fixture_has_expected_shapes() -> None:
    fixture = load_benchmark_fixture("six_color_cube")
    assert fixture.same_mesh.faces.shape == (12, 3)
    assert fixture.expected_same_face_colors.shape == (12, 3)
    assert fixture.repaired_mesh is not None
    assert fixture.expected_repaired_face_colors is not None
    assert fixture.repaired_mesh.faces.shape[0] > fixture.same_mesh.faces.shape[0]
    unique_expected = np.unique(fixture.expected_same_face_colors, axis=0)
    assert len(unique_expected) == 6


def test_smiley_cube_fixture_preserves_small_dark_regions() -> None:
    fixture = load_benchmark_fixture("smiley_cube")
    unique_expected = np.unique(fixture.expected_same_face_colors, axis=0)
    assert any(np.all(color <= np.array([32, 32, 32], dtype=np.uint8)) for color in unique_expected)
    assert fixture.same_mesh.faces.shape[0] > 100


def test_seam_split_quad_fixture_uses_duplicated_positions_for_uv_seams() -> None:
    fixture = load_benchmark_fixture("seam_split_quad")
    assert fixture.same_mesh.faces.shape == (2, 3)
    unique_positions = np.unique(fixture.same_mesh.positions, axis=0)
    assert len(unique_positions) < len(fixture.same_mesh.positions)
