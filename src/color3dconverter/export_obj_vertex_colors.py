from __future__ import annotations

from pathlib import Path

import numpy as np


def write_obj_with_vertex_colors(
    path: str | Path,
    positions: np.ndarray,
    faces: np.ndarray,
    face_colors: np.ndarray,
    *,
    texcoords: np.ndarray | None = None,
    object_name: str = "ColorMesh",
) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pos = np.asarray(positions, dtype=np.float32)
    face_array = np.asarray(faces, dtype=np.int64)
    colors = np.asarray(face_colors, dtype=np.float32)
    uv_array = None if texcoords is None else np.asarray(texcoords, dtype=np.float32)

    lines = [
        "# 3dcolorconverter vertex-color OBJ",
        f"o {object_name}",
    ]
    vertex_index = 1
    for face_index, face in enumerate(face_array):
        face_vertex_ids: list[int] = []
        face_color = np.clip(colors[int(face_index)], 0.0, 1.0)
        for vertex_id in face.tolist():
            position = pos[int(vertex_id)]
            lines.append(
                "v {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    float(face_color[0]),
                    float(face_color[1]),
                    float(face_color[2]),
                )
            )
            if uv_array is not None and len(uv_array) > int(vertex_id):
                uv = uv_array[int(vertex_id)]
                lines.append("vt {:.6f} {:.6f}".format(float(uv[0]), float(uv[1])))
            face_vertex_ids.append(vertex_index)
            vertex_index += 1
        if uv_array is not None:
            lines.append(
                "f "
                + " ".join(f"{idx}/{idx}" for idx in face_vertex_ids)
            )
        else:
            lines.append("f " + " ".join(str(idx) for idx in face_vertex_ids))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_obj_with_per_vertex_colors(
    path: str | Path,
    positions: np.ndarray,
    faces: np.ndarray,
    vertex_colors: np.ndarray,
    *,
    object_name: str = "ColorMesh",
) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pos = np.asarray(positions, dtype=np.float32)
    face_array = np.asarray(faces, dtype=np.int64)
    colors = np.asarray(vertex_colors, dtype=np.float32)
    if len(colors) != len(pos):
        raise ValueError("vertex_colors must align with positions")

    lines = [
        "# 3dcolorconverter per-vertex-color OBJ",
        f"o {object_name}",
    ]
    for index, position in enumerate(pos.tolist()):
        color = np.clip(colors[int(index)], 0.0, 1.0)
        lines.append(
            "v {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                float(position[0]),
                float(position[1]),
                float(position[2]),
                float(color[0]),
                float(color[1]),
                float(color[2]),
            )
        )
    for face in face_array.tolist():
        lines.append("f " + " ".join(str(int(vertex_id) + 1) for vertex_id in face))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
