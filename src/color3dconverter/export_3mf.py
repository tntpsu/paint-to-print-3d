from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as ET

import numpy as np

THREEMF_CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
THREEMF_MATERIAL_NS = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02"
THREEMF_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
XML_NS = "http://www.w3.org/XML/1998/namespace"


def _format_3mf_color(color: np.ndarray) -> str:
    return f"#{int(color[0]):02X}{int(color[1]):02X}{int(color[2]):02X}FF"


def build_colorgroup_3mf_xml(
    positions: np.ndarray,
    faces: np.ndarray,
    palette: np.ndarray,
    face_palette_indices: np.ndarray,
    *,
    object_name: str | None = None,
) -> bytes:
    ET.register_namespace("", THREEMF_CORE_NS)
    ET.register_namespace("m", THREEMF_MATERIAL_NS)
    model = ET.Element(
        f"{{{THREEMF_CORE_NS}}}model",
        attrib={
            "unit": "millimeter",
            f"{{{XML_NS}}}lang": "en-US",
            "requiredextensions": "m",
        },
    )
    resources = ET.SubElement(model, f"{{{THREEMF_CORE_NS}}}resources")
    color_group = ET.SubElement(resources, f"{{{THREEMF_MATERIAL_NS}}}colorgroup", attrib={"id": "1"})
    for color in np.asarray(palette, dtype=np.uint8):
        ET.SubElement(color_group, f"{{{THREEMF_MATERIAL_NS}}}color", attrib={"color": _format_3mf_color(color)})
    object_attrib = {"id": "2", "type": "model", "pid": "1", "pindex": "0"}
    if object_name:
        object_attrib["name"] = " ".join(str(object_name).split())[:120]
    obj = ET.SubElement(resources, f"{{{THREEMF_CORE_NS}}}object", attrib=object_attrib)
    mesh = ET.SubElement(obj, f"{{{THREEMF_CORE_NS}}}mesh")
    vertices = ET.SubElement(mesh, f"{{{THREEMF_CORE_NS}}}vertices")
    for vertex in np.asarray(positions, dtype=np.float32):
        ET.SubElement(
            vertices,
            f"{{{THREEMF_CORE_NS}}}vertex",
            attrib={"x": f"{float(vertex[0]):.6f}", "y": f"{float(vertex[1]):.6f}", "z": f"{float(vertex[2]):.6f}"},
        )
    triangles = ET.SubElement(mesh, f"{{{THREEMF_CORE_NS}}}triangles")
    for face, palette_index in zip(np.asarray(faces, dtype=np.int64), np.asarray(face_palette_indices, dtype=np.int32), strict=False):
        value = str(int(palette_index))
        ET.SubElement(
            triangles,
            f"{{{THREEMF_CORE_NS}}}triangle",
            attrib={
                "v1": str(int(face[0])),
                "v2": str(int(face[1])),
                "v3": str(int(face[2])),
                "pid": "1",
                "p1": value,
                "p2": value,
                "p3": value,
            },
        )
    build = ET.SubElement(model, f"{{{THREEMF_CORE_NS}}}build")
    ET.SubElement(build, f"{{{THREEMF_CORE_NS}}}item", attrib={"objectid": "2"})
    return ET.tostring(model, encoding="utf-8", xml_declaration=True)


def write_colorgroup_3mf(
    output_path: str | Path,
    positions: np.ndarray,
    faces: np.ndarray,
    palette: np.ndarray,
    face_palette_indices: np.ndarray,
    *,
    object_name: str | None = None,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    xml_bytes = build_colorgroup_3mf_xml(
        positions,
        faces,
        palette,
        face_palette_indices,
        object_name=object_name,
    )
    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"""
    relationships = f"""<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="{THREEMF_REL_NS}">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"""
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr("3D/3dmodel.model", xml_bytes)
    return path
