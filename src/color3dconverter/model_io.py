from __future__ import annotations

import io
import json
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import trimesh
from PIL import Image

_COMPONENT_DTYPES: dict[int, str] = {
    5120: "<i1",
    5121: "<u1",
    5122: "<i2",
    5123: "<u2",
    5125: "<u4",
    5126: "<f4",
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


@dataclass
class LoadedTexturedMesh:
    mesh: trimesh.Trimesh | None
    positions: np.ndarray
    faces: np.ndarray
    texcoords: np.ndarray
    texture_rgb: np.ndarray
    source_path: Path
    texture_path: Path | None
    source_format: str
    normal_texture_rgb: np.ndarray | None = None
    orm_texture_rgb: np.ndarray | None = None
    base_color_factor: np.ndarray | None = None
    metallic_factor: float = 1.0
    roughness_factor: float = 1.0
    normal_scale: float = 1.0


def _as_trimesh(mesh_like: object) -> trimesh.Trimesh:
    if isinstance(mesh_like, trimesh.Trimesh):
        return mesh_like
    if isinstance(mesh_like, trimesh.Scene):
        meshes = [geom for geom in mesh_like.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No mesh geometry found in scene.")
        return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
    raise TypeError(f"Unsupported mesh type: {type(mesh_like)!r}")


def _resolve_texture_from_mtl(obj_path: Path) -> Path:
    mtl_path = obj_path.with_suffix(".mtl")
    if not mtl_path.exists():
        raise FileNotFoundError(f"Missing MTL file for {obj_path.name}: {mtl_path}")
    for line in mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("map_Kd "):
            texture_ref = stripped.split(" ", 1)[1].strip()
            candidate = (mtl_path.parent / texture_ref).expanduser().resolve()
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"No diffuse texture reference found in {mtl_path.name}")


def load_textured_obj(obj_path: str | Path, texture_path: str | Path | None = None) -> LoadedTexturedMesh:
    source_path = Path(obj_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    mesh = _as_trimesh(trimesh.load(str(source_path), force="mesh"))
    texcoords = getattr(mesh.visual, "uv", None)
    if texcoords is None:
        raise ValueError(f"{source_path.name} does not contain UV coordinates.")
    resolved_texture_path = (
        Path(texture_path).expanduser().resolve()
        if texture_path
        else _resolve_texture_from_mtl(source_path)
    )
    texture_rgb = np.array(Image.open(resolved_texture_path).convert("RGB"), dtype=np.uint8)
    return LoadedTexturedMesh(
        mesh=mesh,
        positions=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        texcoords=np.asarray(texcoords, dtype=np.float32),
        texture_rgb=texture_rgb,
        normal_texture_rgb=None,
        orm_texture_rgb=None,
        base_color_factor=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        metallic_factor=1.0,
        roughness_factor=1.0,
        normal_scale=1.0,
        source_path=source_path,
        texture_path=resolved_texture_path,
        source_format="obj",
    )


def _glb_chunks(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    if len(raw) < 20:
        raise ValueError(f"{path.name} is too small to be a GLB.")
    magic, version, length = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67:
        raise ValueError(f"{path.name} is not a GLB file.")
    if version != 2:
        raise ValueError(f"{path.name} uses unsupported GLB version {version}.")
    if length > len(raw):
        raise ValueError(f"{path.name} has an invalid reported length.")
    offset = 12
    json_chunk = None
    bin_chunk = b""
    while offset + 8 <= len(raw):
        chunk_len, chunk_type = struct.unpack_from("<II", raw, offset)
        offset += 8
        chunk = raw[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:
            json_chunk = chunk
        elif chunk_type == 0x004E4942:
            bin_chunk = chunk
    if json_chunk is None:
        raise ValueError(f"{path.name} is missing a JSON chunk.")
    return json.loads(json_chunk.decode("utf-8")), bin_chunk


def _read_accessor(gltf: dict[str, Any], bin_blob: bytes, accessor_index: int) -> np.ndarray:
    accessor = (gltf.get("accessors") or [])[accessor_index]
    buffer_view = (gltf.get("bufferViews") or [])[int(accessor.get("bufferView") or 0)]
    component_type = int(accessor.get("componentType") or 0)
    dtype_name = _COMPONENT_DTYPES.get(component_type)
    if not dtype_name:
        raise ValueError(f"Unsupported GLTF component type: {component_type}")
    dtype = np.dtype(dtype_name)
    component_count = _TYPE_COMPONENTS.get(str(accessor.get("type") or "SCALAR"))
    if not component_count:
        raise ValueError(f"Unsupported GLTF accessor type: {accessor.get('type')}")
    count = int(accessor.get("count") or 0)
    byte_offset = int(buffer_view.get("byteOffset") or 0) + int(accessor.get("byteOffset") or 0)
    item_width = dtype.itemsize * component_count
    byte_stride = int(buffer_view.get("byteStride") or item_width)
    if byte_stride == item_width:
        arr = np.frombuffer(bin_blob, dtype=dtype, count=count * component_count, offset=byte_offset)
        shaped = arr.reshape((count, component_count))
    else:
        shaped = np.ndarray(
            shape=(count, component_count),
            dtype=dtype,
            buffer=bin_blob,
            offset=byte_offset,
            strides=(byte_stride, dtype.itemsize),
        ).copy()
    return shaped.reshape((count,)) if component_count == 1 else shaped


def _extract_embedded_image(gltf: dict[str, Any], bin_blob: bytes, image_index: int, *, flip_y: bool = False) -> np.ndarray:
    image_entry = (gltf.get("images") or [])[image_index]
    buffer_view = (gltf.get("bufferViews") or [])[int(image_entry.get("bufferView") or 0)]
    byte_offset = int(buffer_view.get("byteOffset") or 0)
    byte_length = int(buffer_view.get("byteLength") or 0)
    image_bytes = bin_blob[byte_offset : byte_offset + byte_length]
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if flip_y:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return np.array(image, dtype=np.uint8)


def _extract_optional_embedded_texture(
    gltf: dict[str, Any],
    bin_blob: bytes,
    texture_info: dict[str, Any] | None,
    *,
    flip_y: bool = False,
) -> np.ndarray | None:
    if not texture_info:
        return None
    textures = gltf.get("textures") or []
    texture_index = texture_info.get("index")
    if texture_index is None:
        return None
    texture = textures[int(texture_index)]
    image_index = int(texture.get("source") or 0)
    return _extract_embedded_image(gltf, bin_blob, image_index, flip_y=flip_y)


def load_textured_glb(glb_path: str | Path) -> LoadedTexturedMesh:
    source_path = Path(glb_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    gltf, bin_blob = _glb_chunks(source_path)
    primitive = ((gltf.get("meshes") or [{}])[0].get("primitives") or [{}])[0]
    attributes = primitive.get("attributes") or {}
    position_index = attributes.get("POSITION")
    texcoord_index = attributes.get("TEXCOORD_0")
    index_index = primitive.get("indices")
    material_index = primitive.get("material")
    if position_index is None or texcoord_index is None or index_index is None or material_index is None:
        raise ValueError(f"{source_path.name} does not contain POSITION, TEXCOORD_0, indices, and material data.")

    positions = _read_accessor(gltf, bin_blob, int(position_index)).astype(np.float32)
    texcoords = _read_accessor(gltf, bin_blob, int(texcoord_index)).astype(np.float32)
    indices = _read_accessor(gltf, bin_blob, int(index_index)).astype(np.int64)
    if indices.size % 3 != 0:
        raise ValueError(f"{source_path.name} has a non-triangle index count.")
    faces = indices.reshape((-1, 3))

    materials = gltf.get("materials") or []
    textures = gltf.get("textures") or []
    material = materials[int(material_index)]
    pbr = material.get("pbrMetallicRoughness") or {}
    base_color_texture = pbr.get("baseColorTexture") or {}
    texture_index = int(base_color_texture.get("index") or 0)
    texture = textures[texture_index]
    image_index = int(texture.get("source") or 0)
    # glTF loaders such as Three.js sample embedded textures with flipY disabled.
    # The converter's OBJ-era samplers use the opposite image-origin convention,
    # so normalize GLB textures at load time to keep every downstream path honest.
    texture_rgb = _extract_embedded_image(gltf, bin_blob, image_index, flip_y=True)
    orm_texture_rgb = _extract_optional_embedded_texture(gltf, bin_blob, pbr.get("metallicRoughnessTexture"), flip_y=True)
    normal_texture_rgb = _extract_optional_embedded_texture(gltf, bin_blob, material.get("normalTexture"), flip_y=True)
    base_color_factor = np.asarray(pbr.get("baseColorFactor") or [1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    metallic_factor = float(pbr.get("metallicFactor") or 1.0)
    roughness_factor = float(pbr.get("roughnessFactor") or 1.0)
    normal_scale = float((material.get("normalTexture") or {}).get("scale") or 1.0)

    return LoadedTexturedMesh(
        mesh=None,
        positions=positions,
        faces=faces,
        texcoords=texcoords,
        texture_rgb=texture_rgb,
        normal_texture_rgb=normal_texture_rgb,
        orm_texture_rgb=orm_texture_rgb,
        base_color_factor=base_color_factor,
        metallic_factor=metallic_factor,
        roughness_factor=roughness_factor,
        normal_scale=normal_scale,
        source_path=source_path,
        texture_path=None,
        source_format="glb",
    )


def load_textured_objzip(zip_path: str | Path) -> LoadedTexturedMesh:
    source_path = Path(zip_path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    with tempfile.TemporaryDirectory(prefix="color3dconverter_objzip_") as temp_dir:
        temp_root = Path(temp_dir)
        with ZipFile(source_path, "r") as archive:
            archive.extractall(temp_root)
        obj_candidates = sorted(temp_root.rglob("*.obj"))
        if not obj_candidates:
            raise FileNotFoundError(f"No OBJ file found inside {source_path.name}")
        loaded = load_textured_obj(obj_candidates[0])
        return LoadedTexturedMesh(
            mesh=loaded.mesh,
            positions=loaded.positions,
            faces=loaded.faces,
            texcoords=loaded.texcoords,
            texture_rgb=loaded.texture_rgb,
            normal_texture_rgb=loaded.normal_texture_rgb,
            orm_texture_rgb=loaded.orm_texture_rgb,
            base_color_factor=loaded.base_color_factor,
            metallic_factor=loaded.metallic_factor,
            roughness_factor=loaded.roughness_factor,
            normal_scale=loaded.normal_scale,
            source_path=source_path,
            texture_path=None,
            source_format="objzip",
        )


def _white_texture() -> np.ndarray:
    return np.full((1, 1, 3), 255, dtype=np.uint8)


def _fallback_texcoords(vertex_count: int) -> np.ndarray:
    return np.zeros((max(int(vertex_count), 0), 2), dtype=np.float32)


def load_geometry_model(source_path: str | Path, texture_path: str | Path | None = None) -> LoadedTexturedMesh:
    path = Path(source_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".glb":
        try:
            return load_textured_glb(path)
        except Exception:
            pass
    if suffix == ".obj":
        try:
            return load_textured_obj(path, texture_path=texture_path)
        except Exception:
            pass
    if suffix in {".zip", ".objzip"}:
        try:
            return load_textured_objzip(path)
        except Exception:
            pass

    mesh = _as_trimesh(trimesh.load(str(path), force="mesh"))
    texcoords = getattr(mesh.visual, "uv", None)
    if texcoords is None or len(texcoords) != len(mesh.vertices):
        texcoords = _fallback_texcoords(len(mesh.vertices))

    resolved_texture_path = Path(texture_path).expanduser().resolve() if texture_path else None
    texture_rgb = (
        np.array(Image.open(resolved_texture_path).convert("RGB"), dtype=np.uint8)
        if resolved_texture_path
        else _white_texture()
    )
    return LoadedTexturedMesh(
        mesh=mesh,
        positions=np.asarray(mesh.vertices, dtype=np.float32),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        texcoords=np.asarray(texcoords, dtype=np.float32),
        texture_rgb=texture_rgb,
        normal_texture_rgb=None,
        orm_texture_rgb=None,
        base_color_factor=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        metallic_factor=1.0,
        roughness_factor=1.0,
        normal_scale=1.0,
        source_path=path,
        texture_path=resolved_texture_path,
        source_format=(suffix.lstrip(".") or "mesh"),
    )


def load_textured_model(source_path: str | Path, texture_path: str | Path | None = None) -> LoadedTexturedMesh:
    path = Path(source_path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return load_textured_obj(path, texture_path=texture_path)
    if suffix == ".glb":
        return load_textured_glb(path)
    if suffix in {".zip", ".objzip"}:
        return load_textured_objzip(path)
    raise ValueError(f"Unsupported source format: {suffix}")
