from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from color3dconverter.bake import sample_texture_bilinear
from color3dconverter.model_io import load_textured_model


def _pad4(data: bytes) -> bytes:
    return data + (b" " * ((4 - (len(data) % 4)) % 4))


def _write_minimal_textured_glb(path: Path) -> None:
    texture = Image.new("RGB", (2, 2), (0, 0, 0))
    texture.putpixel((0, 0), (255, 0, 0))
    texture.putpixel((1, 0), (0, 255, 0))
    texture.putpixel((0, 1), (0, 0, 255))
    texture.putpixel((1, 1), (255, 255, 0))
    image_buffer = io.BytesIO()
    texture.save(image_buffer, format="PNG")
    image_bytes = image_buffer.getvalue()

    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="<f4").tobytes()
    texcoords = np.array([[0.0, 0.0], [0.99, 0.0], [0.0, 0.99]], dtype="<f4").tobytes()
    indices = np.array([0, 1, 2], dtype="<u2").tobytes()

    chunks: list[bytes] = []
    offsets: list[int] = []
    for payload in (positions, texcoords, indices, image_bytes):
        offsets.append(sum(len(chunk) for chunk in chunks))
        chunks.append(_pad4(payload))
    bin_blob = b"".join(chunks)
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": offsets[0], "byteLength": len(positions)},
            {"buffer": 0, "byteOffset": offsets[1], "byteLength": len(texcoords)},
            {"buffer": 0, "byteOffset": offsets[2], "byteLength": len(indices)},
            {"buffer": 0, "byteOffset": offsets[3], "byteLength": len(image_bytes)},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
            {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "images": [{"bufferView": 3, "mimeType": "image/png"}],
        "textures": [{"source": 0}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
    }
    json_blob = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"))
    total_length = 12 + 8 + len(json_blob) + 8 + len(bin_blob)
    path.write_bytes(
        b"glTF"
        + struct.pack("<II", 2, total_length)
        + struct.pack("<II", len(json_blob), 0x4E4F534A)
        + json_blob
        + struct.pack("<II", len(bin_blob), 0x004E4942)
        + bin_blob
    )


def test_load_textured_glb_normalizes_texture_origin(tmp_path: Path) -> None:
    glb_path = tmp_path / "sample.glb"
    _write_minimal_textured_glb(glb_path)

    loaded = load_textured_model(glb_path)
    vertex_colors = sample_texture_bilinear(loaded.texture_rgb, loaded.texcoords)

    assert np.allclose(vertex_colors[0], [255, 0, 0], atol=4)
    assert np.allclose(vertex_colors[1], [0, 255, 0], atol=4)
    assert np.allclose(vertex_colors[2], [0, 0, 255], atol=4)
