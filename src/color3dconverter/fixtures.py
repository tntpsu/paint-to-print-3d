from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw
import trimesh

from .model_io import LoadedTexturedMesh
from .face_regions import sample_texture


@dataclass
class BenchmarkFixture:
    name: str
    description: str
    same_mesh: LoadedTexturedMesh
    expected_same_face_colors: np.ndarray
    repaired_mesh: LoadedTexturedMesh | None = None
    expected_repaired_face_colors: np.ndarray | None = None
    suggested_regions: int = 6
    pass_threshold_same_mesh: float = 0.98
    pass_threshold_repaired: float = 0.94


def _uv_rect(col: int, row: int, *, cols: int, rows: int, inset: float = 0.08) -> tuple[float, float, float, float]:
    cell_w = 1.0 / float(cols)
    cell_h = 1.0 / float(rows)
    u0 = col * cell_w + cell_w * inset
    v0 = row * cell_h + cell_h * inset
    u1 = (col + 1) * cell_w - cell_w * inset
    v1 = (row + 1) * cell_h - cell_h * inset
    return (u0, v0, u1, v1)


def _append_grid_face(
    positions: list[list[float]],
    texcoords: list[list[float]],
    faces: list[list[int]],
    *,
    origin: tuple[float, float, float],
    u_vec: tuple[float, float, float],
    v_vec: tuple[float, float, float],
    steps_u: int,
    steps_v: int,
    uv_rect: tuple[float, float, float, float],
) -> None:
    base_index = len(positions)
    origin_v = np.asarray(origin, dtype=np.float32)
    u_step = np.asarray(u_vec, dtype=np.float32)
    v_step = np.asarray(v_vec, dtype=np.float32)
    u0, v0, u1, v1 = uv_rect
    grid_indices = np.zeros((steps_v + 1, steps_u + 1), dtype=np.int64)
    for iy in range(steps_v + 1):
        fy = iy / float(max(steps_v, 1))
        for ix in range(steps_u + 1):
            fx = ix / float(max(steps_u, 1))
            position = origin_v + u_step * fx + v_step * fy
            uv = [u0 + (u1 - u0) * fx, v0 + (v1 - v0) * fy]
            positions.append(position.tolist())
            texcoords.append(uv)
            grid_indices[iy, ix] = base_index
            base_index += 1
    for iy in range(steps_v):
        for ix in range(steps_u):
            a = int(grid_indices[iy, ix])
            b = int(grid_indices[iy, ix + 1])
            c = int(grid_indices[iy + 1, ix])
            d = int(grid_indices[iy + 1, ix + 1])
            faces.append([a, b, d])
            faces.append([a, d, c])


def _image_from_cells(
    *,
    cols: int,
    rows: int,
    cell_px: int = 128,
    fill_fn: Callable[[Image.Image, ImageDraw.ImageDraw, int, int, tuple[int, int, int, int]], None],
) -> np.ndarray:
    image = Image.new("RGB", (cols * cell_px, rows * cell_px), (245, 241, 233))
    draw = ImageDraw.Draw(image)
    for row in range(rows):
        for col in range(cols):
            box = (col * cell_px, row * cell_px, (col + 1) * cell_px, (row + 1) * cell_px)
            fill_fn(image, draw, col, row, box)
    return np.array(image, dtype=np.uint8)


def _flat_cell_texture(
    *,
    cols: int,
    rows: int,
    color_map: dict[tuple[int, int], tuple[int, int, int]],
    cell_px: int = 128,
) -> np.ndarray:
    def fill_fn(image: Image.Image, draw: ImageDraw.ImageDraw, col: int, row: int, box: tuple[int, int, int, int]) -> None:
        draw.rectangle(box, fill=color_map.get((col, row), (230, 230, 230)))

    return _image_from_cells(cols=cols, rows=rows, cell_px=cell_px, fill_fn=fill_fn)


def _majority_face_colors(loaded: LoadedTexturedMesh) -> np.ndarray:
    vertex_colors = sample_texture(loaded.texture_rgb, loaded.texcoords)
    result = np.zeros((len(loaded.faces), 3), dtype=np.uint8)
    for face_index, face in enumerate(np.asarray(loaded.faces, dtype=np.int64)):
        color_counts: dict[tuple[int, int, int], int] = {}
        for vertex_index in face.tolist():
            color = tuple(int(channel) for channel in vertex_colors[int(vertex_index)].tolist())
            color_counts[color] = color_counts.get(color, 0) + 1
        result[face_index] = np.array(max(color_counts.items(), key=lambda item: item[1])[0], dtype=np.uint8)
    return result


def _centroid_face_colors(loaded: LoadedTexturedMesh) -> np.ndarray:
    if len(loaded.faces) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    uv_centroids = np.asarray(loaded.texcoords, dtype=np.float32)[np.asarray(loaded.faces, dtype=np.int64)].mean(axis=1)
    return sample_texture(loaded.texture_rgb, uv_centroids)


def _subdivide_loaded_mesh(loaded: LoadedTexturedMesh, *, iterations: int = 1, source_stem_suffix: str = "_repaired") -> LoadedTexturedMesh:
    positions = np.asarray(loaded.positions, dtype=np.float32)
    texcoords = np.asarray(loaded.texcoords, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    for _ in range(max(int(iterations), 0)):
        edge_midpoints: dict[tuple[int, int], int] = {}
        new_positions = positions.tolist()
        new_texcoords = texcoords.tolist()
        new_faces: list[list[int]] = []

        def midpoint_index(i: int, j: int) -> int:
            key = (i, j) if i < j else (j, i)
            existing = edge_midpoints.get(key)
            if existing is not None:
                return existing
            midpoint = ((positions[i] + positions[j]) * 0.5).astype(np.float32)
            uv_midpoint = ((texcoords[i] + texcoords[j]) * 0.5).astype(np.float32)
            index = len(new_positions)
            new_positions.append(midpoint.tolist())
            new_texcoords.append(uv_midpoint.tolist())
            edge_midpoints[key] = index
            return index

        for face in faces.tolist():
            a, b, c = (int(face[0]), int(face[1]), int(face[2]))
            ab = midpoint_index(a, b)
            bc = midpoint_index(b, c)
            ca = midpoint_index(c, a)
            new_faces.extend(
                [
                    [a, ab, ca],
                    [ab, b, bc],
                    [ca, bc, c],
                    [ab, bc, ca],
                ]
            )
        positions = np.asarray(new_positions, dtype=np.float32)
        texcoords = np.asarray(new_texcoords, dtype=np.float32)
        faces = np.asarray(new_faces, dtype=np.int64)
    return LoadedTexturedMesh(
        mesh=None,
        positions=positions,
        faces=faces,
        texcoords=texcoords,
        texture_rgb=np.asarray(loaded.texture_rgb, dtype=np.uint8),
        source_path=loaded.source_path.with_name(f"{loaded.source_path.stem}{source_stem_suffix}{loaded.source_path.suffix or '.glb'}"),
        texture_path=None,
        source_format=loaded.source_format,
    )


def _deform_loaded_mesh(
    loaded: LoadedTexturedMesh,
    *,
    twist_degrees: float = 28.0,
    squash_y: float = 0.84,
    bulge_z: float = 1.08,
    source_stem_suffix: str = "_deformed",
) -> LoadedTexturedMesh:
    positions = np.asarray(loaded.positions, dtype=np.float32).copy()
    if len(positions) > 0:
        normalized_y = positions[:, 1].astype(np.float32, copy=False)
        max_abs_y = float(np.max(np.abs(normalized_y))) if len(normalized_y) else 1.0
        max_abs_y = max(max_abs_y, 1e-6)
        angles = np.deg2rad(float(twist_degrees)) * (normalized_y / max_abs_y)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        x = positions[:, 0].copy()
        z = positions[:, 2].copy()
        positions[:, 0] = x * cos_a - z * sin_a
        positions[:, 2] = x * sin_a + z * cos_a
        positions[:, 1] *= float(squash_y)
        positions[:, 2] *= float(bulge_z)
    return LoadedTexturedMesh(
        mesh=None,
        positions=positions,
        faces=np.asarray(loaded.faces, dtype=np.int64),
        texcoords=np.asarray(loaded.texcoords, dtype=np.float32),
        texture_rgb=np.asarray(loaded.texture_rgb, dtype=np.uint8),
        source_path=loaded.source_path.with_name(f"{loaded.source_path.stem}{source_stem_suffix}{loaded.source_path.suffix or '.glb'}"),
        texture_path=None,
        source_format=loaded.source_format,
    )


def _loaded_mesh(
    *,
    name: str,
    positions: list[list[float]] | np.ndarray,
    faces: list[list[int]] | np.ndarray,
    texcoords: list[list[float]] | np.ndarray,
    texture_rgb: np.ndarray,
    source_format: str = "synthetic",
) -> LoadedTexturedMesh:
    return LoadedTexturedMesh(
        mesh=None,
        positions=np.asarray(positions, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        texcoords=np.asarray(texcoords, dtype=np.float32),
        texture_rgb=np.asarray(texture_rgb, dtype=np.uint8),
        source_path=Path(f"/synthetic/{name}.{source_format if source_format != 'synthetic' else 'glb'}"),
        texture_path=None,
        source_format=source_format,
    )


def build_six_color_cube_fixture() -> BenchmarkFixture:
    colors = {
        (0, 0): (224, 68, 55),
        (1, 0): (47, 125, 214),
        (2, 0): (248, 200, 44),
        (0, 1): (56, 166, 84),
        (1, 1): (145, 87, 214),
        (2, 1): (241, 132, 36),
    }
    texture = _flat_cell_texture(cols=3, rows=2, color_map=colors, cell_px=96)
    positions: list[list[float]] = []
    texcoords: list[list[float]] = []
    faces: list[list[int]] = []
    face_specs = [
        ((-1.0, -1.0, 1.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), _uv_rect(0, 0, cols=3, rows=2)),
        ((1.0, -1.0, -1.0), (-2.0, 0.0, 0.0), (0.0, 2.0, 0.0), _uv_rect(1, 0, cols=3, rows=2)),
        ((1.0, -1.0, 1.0), (0.0, 0.0, -2.0), (0.0, 2.0, 0.0), _uv_rect(2, 0, cols=3, rows=2)),
        ((-1.0, -1.0, -1.0), (0.0, 0.0, 2.0), (0.0, 2.0, 0.0), _uv_rect(0, 1, cols=3, rows=2)),
        ((-1.0, 1.0, 1.0), (2.0, 0.0, 0.0), (0.0, 0.0, -2.0), _uv_rect(1, 1, cols=3, rows=2)),
        ((-1.0, -1.0, -1.0), (2.0, 0.0, 0.0), (0.0, 0.0, 2.0), _uv_rect(2, 1, cols=3, rows=2)),
    ]
    for origin, u_vec, v_vec, rect in face_specs:
        _append_grid_face(positions, texcoords, faces, origin=origin, u_vec=u_vec, v_vec=v_vec, steps_u=1, steps_v=1, uv_rect=rect)
    same_mesh = _loaded_mesh(name="six_color_cube", positions=positions, faces=faces, texcoords=texcoords, texture_rgb=texture)
    repaired_mesh = _subdivide_loaded_mesh(same_mesh, iterations=1)
    return BenchmarkFixture(
        name="six_color_cube",
        description="Simple cube with one distinct flat color per face.",
        same_mesh=same_mesh,
        expected_same_face_colors=_centroid_face_colors(same_mesh),
        repaired_mesh=repaired_mesh,
        expected_repaired_face_colors=_centroid_face_colors(repaired_mesh),
        suggested_regions=6,
        pass_threshold_same_mesh=0.99,
        pass_threshold_repaired=0.97,
    )


def build_smiley_cube_fixture() -> BenchmarkFixture:
    base_colors = {
        (0, 0): (244, 210, 66),
        (1, 0): (52, 122, 208),
        (2, 0): (56, 166, 84),
        (0, 1): (240, 132, 36),
        (1, 1): (155, 90, 219),
        (2, 1): (218, 70, 64),
    }

    def fill_fn(image: Image.Image, draw: ImageDraw.ImageDraw, col: int, row: int, box: tuple[int, int, int, int]) -> None:
        draw.rectangle(box, fill=base_colors[(col, row)])
        if (col, row) != (0, 0):
            return
        x0, y0, x1, y1 = box
        eye_r = (x1 - x0) * 0.08
        left_eye = (x0 + (x1 - x0) * 0.28, y0 + (y1 - y0) * 0.34)
        right_eye = (x0 + (x1 - x0) * 0.72, y0 + (y1 - y0) * 0.34)
        for cx, cy in (left_eye, right_eye):
            draw.ellipse((cx - eye_r, cy - eye_r, cx + eye_r, cy + eye_r), fill=(16, 16, 16))
        smile_box = (x0 + (x1 - x0) * 0.25, y0 + (y1 - y0) * 0.34, x0 + (x1 - x0) * 0.75, y0 + (y1 - y0) * 0.78)
        draw.arc(smile_box, start=20, end=160, fill=(16, 16, 16), width=max(2, int((x1 - x0) * 0.05)))

    texture = _image_from_cells(cols=3, rows=2, cell_px=160, fill_fn=fill_fn)
    positions: list[list[float]] = []
    texcoords: list[list[float]] = []
    faces: list[list[int]] = []
    face_specs = [
        ((-1.0, -1.0, 1.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0), 24, 24, _uv_rect(0, 0, cols=3, rows=2, inset=0.02)),
        ((1.0, -1.0, -1.0), (-2.0, 0.0, 0.0), (0.0, 2.0, 0.0), 1, 1, _uv_rect(1, 0, cols=3, rows=2)),
        ((1.0, -1.0, 1.0), (0.0, 0.0, -2.0), (0.0, 2.0, 0.0), 1, 1, _uv_rect(2, 0, cols=3, rows=2)),
        ((-1.0, -1.0, -1.0), (0.0, 0.0, 2.0), (0.0, 2.0, 0.0), 1, 1, _uv_rect(0, 1, cols=3, rows=2)),
        ((-1.0, 1.0, 1.0), (2.0, 0.0, 0.0), (0.0, 0.0, -2.0), 1, 1, _uv_rect(1, 1, cols=3, rows=2)),
        ((-1.0, -1.0, -1.0), (2.0, 0.0, 0.0), (0.0, 0.0, 2.0), 1, 1, _uv_rect(2, 1, cols=3, rows=2)),
    ]
    for origin, u_vec, v_vec, steps_u, steps_v, rect in face_specs:
        _append_grid_face(
            positions,
            texcoords,
            faces,
            origin=origin,
            u_vec=u_vec,
            v_vec=v_vec,
            steps_u=steps_u,
            steps_v=steps_v,
            uv_rect=rect,
        )
    same_mesh = _loaded_mesh(name="smiley_cube", positions=positions, faces=faces, texcoords=texcoords, texture_rgb=texture)
    repaired_mesh = _subdivide_loaded_mesh(same_mesh, iterations=1)
    return BenchmarkFixture(
        name="smiley_cube",
        description="Cube with small black smiley details on one face to test tiny high-contrast region preservation.",
        same_mesh=same_mesh,
        expected_same_face_colors=_centroid_face_colors(same_mesh),
        repaired_mesh=repaired_mesh,
        expected_repaired_face_colors=_centroid_face_colors(repaired_mesh),
        suggested_regions=6,
        pass_threshold_same_mesh=0.95,
        pass_threshold_repaired=0.9,
    )


def build_seam_split_quad_fixture() -> BenchmarkFixture:
    colors = {
        (0, 0): (226, 74, 60),
        (1, 0): (58, 116, 218),
    }
    texture = _flat_cell_texture(cols=2, rows=1, color_map=colors, cell_px=128)
    positions = [
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ]
    texcoords = [
        [0.08, 0.90],
        [0.42, 0.90],
        [0.42, 0.10],
        [0.58, 0.90],
        [0.92, 0.10],
        [0.58, 0.10],
    ]
    faces = [[0, 1, 2], [3, 4, 5]]
    same_mesh = _loaded_mesh(
        name="seam_split_quad",
        positions=positions,
        faces=faces,
        texcoords=texcoords,
        texture_rgb=texture,
    )
    return BenchmarkFixture(
        name="seam_split_quad",
        description="Two triangles share identical geometric positions but sample different UV islands, which makes seam-smearing easy to spot when colors are collapsed by geometric vertex.",
        same_mesh=same_mesh,
        expected_same_face_colors=_centroid_face_colors(same_mesh),
        suggested_regions=2,
        pass_threshold_same_mesh=0.99,
    )


def build_checker_quad_fixture() -> BenchmarkFixture:
    cell_px = 256
    texture = Image.new("RGB", (cell_px, cell_px), (245, 209, 78))
    draw = ImageDraw.Draw(texture)
    square = cell_px // 8
    start_x = int(cell_px * 0.60)
    start_y = int(cell_px * 0.18)
    for row in range(3):
        for col in range(3):
            if (row + col) % 2 == 0:
                x0 = start_x + col * square
                y0 = start_y + row * square
                draw.rectangle((x0, y0, x0 + square - 1, y0 + square - 1), fill=(18, 18, 18))
    positions: list[list[float]] = []
    texcoords: list[list[float]] = []
    faces: list[list[int]] = []
    _append_grid_face(
        positions,
        texcoords,
        faces,
        origin=(-1.0, -1.0, 0.0),
        u_vec=(2.0, 0.0, 0.0),
        v_vec=(0.0, 2.0, 0.0),
        steps_u=36,
        steps_v=36,
        uv_rect=(0.02, 0.02, 0.98, 0.98),
    )
    same_mesh = _loaded_mesh(
        name="checker_quad",
        positions=positions,
        faces=faces,
        texcoords=texcoords,
        texture_rgb=np.asarray(texture, dtype=np.uint8),
    )
    return BenchmarkFixture(
        name="checker_quad",
        description="Single flat quad with a tiny black checker patch to measure how well different bake representations preserve small high-contrast details.",
        same_mesh=same_mesh,
        expected_same_face_colors=_centroid_face_colors(same_mesh),
        suggested_regions=2,
        pass_threshold_same_mesh=0.95,
    )


def build_banded_sphere_fixture() -> BenchmarkFixture:
    lat_steps = 18
    lon_steps = 36
    positions: list[list[float]] = []
    texcoords: list[list[float]] = []
    faces: list[list[int]] = []
    for iy in range(lat_steps + 1):
        v = iy / float(lat_steps)
        theta = np.pi * v
        y = np.cos(theta)
        r = np.sin(theta)
        for ix in range(lon_steps + 1):
            u = ix / float(lon_steps)
            phi = 2.0 * np.pi * u
            x = r * np.cos(phi)
            z = r * np.sin(phi)
            positions.append([float(x), float(y), float(z)])
            texcoords.append([float(u), float(v)])
    row_width = lon_steps + 1
    for iy in range(lat_steps):
        for ix in range(lon_steps):
            a = iy * row_width + ix
            b = a + 1
            c = a + row_width
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    band_colors = [
        (247, 217, 72),
        (233, 120, 48),
        (69, 145, 220),
        (72, 174, 96),
    ]

    def fill_fn(image: Image.Image, draw: ImageDraw.ImageDraw, col: int, row: int, box: tuple[int, int, int, int]) -> None:
        x0, y0, x1, y1 = box
        band_h = (y1 - y0) / len(band_colors)
        for index, color in enumerate(band_colors):
            top = int(y0 + band_h * index)
            bottom = int(y0 + band_h * (index + 1))
            draw.rectangle((x0, top, x1, bottom), fill=color)

    texture = _image_from_cells(cols=1, rows=1, cell_px=256, fill_fn=fill_fn)
    same_mesh = _loaded_mesh(name="banded_sphere", positions=positions, faces=faces, texcoords=texcoords, texture_rgb=texture)
    repaired_mesh = _subdivide_loaded_mesh(same_mesh, iterations=1)
    return BenchmarkFixture(
        name="banded_sphere",
        description="Rounded benchmark with clean horizontal color bands to test curved-surface region preservation.",
        same_mesh=same_mesh,
        expected_same_face_colors=_centroid_face_colors(same_mesh),
        repaired_mesh=repaired_mesh,
        expected_repaired_face_colors=_centroid_face_colors(repaired_mesh),
        suggested_regions=4,
        pass_threshold_same_mesh=0.96,
        pass_threshold_repaired=0.92,
    )


def build_deformed_banded_sphere_fixture() -> BenchmarkFixture:
    base = build_banded_sphere_fixture()
    repaired_mesh = _deform_loaded_mesh(_subdivide_loaded_mesh(base.same_mesh, iterations=1), source_stem_suffix="_deformed")
    return BenchmarkFixture(
        name="deformed_banded_sphere",
        description="Curved deformation benchmark that twists and squashes the banded sphere to test transfer tolerance under nontrivial geometry change.",
        same_mesh=base.same_mesh,
        expected_same_face_colors=base.expected_same_face_colors,
        repaired_mesh=repaired_mesh,
        expected_repaired_face_colors=_centroid_face_colors(repaired_mesh),
        suggested_regions=4,
        pass_threshold_same_mesh=0.96,
        pass_threshold_repaired=0.88,
    )


def _append_mesh_part(
    positions: list[list[float]],
    texcoords: list[list[float]],
    faces: list[list[int]],
    *,
    mesh: trimesh.Trimesh,
    uv_rect: tuple[float, float, float, float],
) -> None:
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    tris = np.asarray(mesh.faces, dtype=np.int64)
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    extents = bbox_max - bbox_min
    axis_order = np.argsort(extents)[::-1]
    ax0 = int(axis_order[0])
    ax1 = int(axis_order[1]) if extents[axis_order[1]] > 1e-6 else (ax0 + 1) % 3
    uv0, vv0, uv1, vv1 = uv_rect
    denom0 = max(float(extents[ax0]), 1e-6)
    denom1 = max(float(extents[ax1]), 1e-6)
    base_index = len(positions)
    for vertex in verts.tolist():
        u = (float(vertex[ax0]) - float(bbox_min[ax0])) / denom0
        v = (float(vertex[ax1]) - float(bbox_min[ax1])) / denom1
        positions.append(vertex)
        texcoords.append([uv0 + (uv1 - uv0) * u, vv0 + (vv1 - vv0) * v])
    for face in tris.tolist():
        faces.append([base_index + int(face[0]), base_index + int(face[1]), base_index + int(face[2])])


def build_simple_duck_fixture() -> BenchmarkFixture:
    atlas_colors = {
        (0, 0): (242, 210, 74),
        (1, 0): (227, 126, 42),
        (2, 0): (153, 96, 60),
        (0, 1): (183, 54, 48),
        (1, 1): (90, 62, 50),
        (2, 1): (226, 188, 120),
    }
    texture = _flat_cell_texture(cols=3, rows=2, color_map=atlas_colors, cell_px=128)
    positions: list[list[float]] = []
    texcoords: list[list[float]] = []
    faces: list[list[int]] = []

    body = trimesh.creation.icosphere(subdivisions=2, radius=0.62)
    body.vertices *= np.array([1.0, 0.82, 0.76], dtype=np.float32)
    head = trimesh.creation.icosphere(subdivisions=1, radius=0.28)
    head.vertices *= np.array([1.0, 0.95, 0.92], dtype=np.float32)
    head.vertices += np.array([0.58, 0.22, 0.0], dtype=np.float32)
    beak = trimesh.creation.box(extents=(0.24, 0.14, 0.18))
    beak.vertices += np.array([0.92, 0.16, 0.0], dtype=np.float32)
    bandana = trimesh.creation.box(extents=(0.34, 0.08, 0.28))
    bandana.vertices += np.array([0.38, -0.02, 0.0], dtype=np.float32)
    hat_brim = trimesh.creation.box(extents=(0.48, 0.05, 0.42))
    hat_brim.vertices += np.array([0.5, 0.58, 0.0], dtype=np.float32)
    hat_crown = trimesh.creation.box(extents=(0.26, 0.2, 0.24))
    hat_crown.vertices += np.array([0.5, 0.74, 0.0], dtype=np.float32)
    left_boot = trimesh.creation.box(extents=(0.18, 0.12, 0.18))
    left_boot.vertices += np.array([0.08, -0.56, -0.18], dtype=np.float32)
    right_boot = trimesh.creation.box(extents=(0.18, 0.12, 0.18))
    right_boot.vertices += np.array([0.08, -0.56, 0.18], dtype=np.float32)

    _append_mesh_part(positions, texcoords, faces, mesh=body, uv_rect=_uv_rect(0, 0, cols=3, rows=2))
    _append_mesh_part(positions, texcoords, faces, mesh=head, uv_rect=_uv_rect(0, 0, cols=3, rows=2))
    _append_mesh_part(positions, texcoords, faces, mesh=beak, uv_rect=_uv_rect(1, 0, cols=3, rows=2))
    _append_mesh_part(positions, texcoords, faces, mesh=hat_brim, uv_rect=_uv_rect(2, 0, cols=3, rows=2))
    _append_mesh_part(positions, texcoords, faces, mesh=hat_crown, uv_rect=_uv_rect(2, 1, cols=3, rows=2))
    _append_mesh_part(positions, texcoords, faces, mesh=bandana, uv_rect=_uv_rect(0, 1, cols=3, rows=2))
    _append_mesh_part(positions, texcoords, faces, mesh=left_boot, uv_rect=_uv_rect(1, 1, cols=3, rows=2))
    _append_mesh_part(positions, texcoords, faces, mesh=right_boot, uv_rect=_uv_rect(1, 1, cols=3, rows=2))

    same_mesh = _loaded_mesh(name="simple_duck", positions=positions, faces=faces, texcoords=texcoords, texture_rgb=texture)
    repaired_mesh = _subdivide_loaded_mesh(same_mesh, iterations=1)
    return BenchmarkFixture(
        name="simple_duck",
        description="Disconnected toy duck parts with flat semantic colors for body, beak, hat, bandana, and boots.",
        same_mesh=same_mesh,
        expected_same_face_colors=_centroid_face_colors(same_mesh),
        repaired_mesh=repaired_mesh,
        expected_repaired_face_colors=_centroid_face_colors(repaired_mesh),
        suggested_regions=6,
        pass_threshold_same_mesh=0.97,
        pass_threshold_repaired=0.93,
    )


_FIXTURE_BUILDERS: dict[str, Callable[[], BenchmarkFixture]] = {
    "checker_quad": build_checker_quad_fixture,
    "seam_split_quad": build_seam_split_quad_fixture,
    "six_color_cube": build_six_color_cube_fixture,
    "smiley_cube": build_smiley_cube_fixture,
    "banded_sphere": build_banded_sphere_fixture,
    "deformed_banded_sphere": build_deformed_banded_sphere_fixture,
    "simple_duck": build_simple_duck_fixture,
}


def list_benchmark_fixtures() -> list[str]:
    return sorted(_FIXTURE_BUILDERS)


def load_benchmark_fixture(name: str) -> BenchmarkFixture:
    try:
        builder = _FIXTURE_BUILDERS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown benchmark fixture: {name}") from exc
    return builder()
