from __future__ import annotations

import colorsys
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from textwrap import wrap
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from .benchmark import _write_texture_source_preview
from .model_io import load_textured_model
from .production import run_repaired_production_conversion


HANDOFF_SCHEMA_VERSION = "duckagent.paint_to_print_handoff.v1"

COOL_COLOR_INTENT_TERMS = {
    "blue",
    "purple",
    "violet",
    "indigo",
    "navy",
    "aqua",
    "teal",
    "turquoise",
    "ice",
    "frozen",
    "galaxy",
    "space",
    "ocean",
    "water",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _existing_path(value: Any) -> str | None:
    if not value:
        return None
    path = Path(str(value)).expanduser().resolve()
    return str(path) if path.exists() else None


def _gate(
    gate_id: str,
    *,
    passed: bool,
    summary: str,
    required: bool = True,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "passed": bool(passed),
        "required": bool(required),
        "summary": summary,
        "details": details or {},
    }


def _load_json(path_value: Any) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(str(path_value)).expanduser().resolve()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _object_terms(object_name: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(object_name or "").lower()))


def _palette_rows_from_reports(
    *,
    production_report: dict[str, Any],
    conversion_report: dict[str, Any],
) -> list[dict[str, Any]]:
    palette = conversion_report.get("palette") or production_report.get("palette") or []
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(palette):
        if not isinstance(row, dict):
            continue
        rgb = row.get("rgb")
        if not isinstance(rgb, (list, tuple)) or len(rgb) < 3:
            continue
        try:
            color = [max(0, min(255, int(value))) for value in rgb[:3]]
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "palette_index": int(row.get("palette_index") if row.get("palette_index") is not None else index),
                "hex": str(row.get("hex") or f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"),
                "rgb": color,
                "face_count": max(0, int(row.get("face_count") or 0)),
            }
        )
    return rows


def _palette_color_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_faces = sum(int(row.get("face_count") or 0) for row in rows)
    if total_faces <= 0:
        total_faces = len(rows)

    cool_weight = 0
    warm_detail_weight = 0
    light_neutral_weight = 0
    hues: list[float] = []
    for row in rows:
        rgb = row.get("rgb") or [0, 0, 0]
        r, g, b = [float(value) / 255.0 for value in rgb[:3]]
        hue, saturation, value = colorsys.rgb_to_hsv(r, g, b)
        hue_degrees = hue * 360.0
        hues.append(hue_degrees)
        weight = int(row.get("face_count") or 0) or 1
        is_cool_wash = 185.0 <= hue_degrees <= 275.0 and b >= max(r, g) - 0.03 and value >= 0.62
        is_warm_detail = (
            (18.0 <= hue_degrees <= 78.0 and saturation >= 0.16 and value >= 0.28)
            or (r >= g >= b and (r - b) >= 0.18 and value >= 0.25)
        )
        is_light_neutral = saturation <= 0.16 and value >= 0.72
        if is_cool_wash:
            cool_weight += weight
        if is_warm_detail:
            warm_detail_weight += weight
        if is_light_neutral:
            light_neutral_weight += weight

    hue_span = max(hues) - min(hues) if hues else 0.0
    return {
        "palette_size": len(rows),
        "palette_hexes": [str(row.get("hex")) for row in rows],
        "cool_wash_share": round(float(cool_weight) / float(total_faces), 4) if total_faces else 0.0,
        "warm_detail_share": round(float(warm_detail_weight) / float(total_faces), 4) if total_faces else 0.0,
        "light_neutral_share": round(float(light_neutral_weight) / float(total_faces), 4) if total_faces else 0.0,
        "hue_span_degrees": round(hue_span, 2),
    }


def _assess_visual_color_confidence(
    *,
    production_report: dict[str, Any],
    conversion_report: dict[str, Any],
    object_name: str | None,
) -> dict[str, Any]:
    rows = _palette_rows_from_reports(production_report=production_report, conversion_report=conversion_report)
    profile = _palette_color_profile(rows)
    terms = _object_terms(object_name)
    has_duck_intent = "duck" in terms
    explicitly_cool = bool(terms & COOL_COLOR_INTENT_TERMS)
    duck_intent = conversion_report.get("duck_color_intent") or {}
    beak_missing = has_duck_intent and duck_intent and duck_intent.get("beak_label") is None

    reasons: list[str] = []
    if not rows:
        reasons.append("No palette rows were available for visual color confidence checks.")
    if (
        has_duck_intent
        and not explicitly_cool
        and profile["palette_size"] >= 3
        and profile["cool_wash_share"] >= 0.82
        and profile["warm_detail_share"] <= 0.05
    ):
        reasons.append("Palette is mostly cool blue/purple with almost no warm duck, beak, tan, brown, or orange detail.")
    if (
        has_duck_intent
        and not explicitly_cool
        and beak_missing
        and profile["warm_detail_share"] <= 0.03
    ):
        reasons.append("Duck color-intent could not identify a beak or warm detail region.")

    ready = not reasons
    return {
        "status": "ready" if ready else "review_required",
        "ready": ready,
        "object_name": object_name,
        "object_terms": sorted(terms),
        "explicitly_cool_intent": explicitly_cool,
        "beak_missing": bool(beak_missing),
        "reasons": reasons,
        "profile": profile,
    }


def _write_source_preview(source_path: str | Path, texture_path: str | Path | None, output_path: Path) -> str:
    loaded = load_textured_model(source_path, texture_path=texture_path)
    _write_texture_source_preview(
        output_path,
        positions=loaded.positions,
        faces=loaded.faces,
        texcoords=loaded.texcoords,
        texture_rgb=loaded.texture_rgb,
    )
    return str(output_path)


def _fit_panel(image_path: str | None, size: tuple[int, int], *, fallback: str) -> Image.Image:
    panel = Image.new("RGB", size, (238, 235, 228))
    draw = ImageDraw.Draw(panel)
    if not image_path:
        draw.text((18, size[1] // 2 - 8), fallback, fill=(78, 74, 66))
        return panel
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        draw.text((18, size[1] // 2 - 8), fallback, fill=(78, 74, 66))
        return panel
    image = Image.open(path).convert("RGB")
    fitted = ImageOps.contain(image, size, method=Image.Resampling.BICUBIC)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    panel.paste(fitted, (x, y))
    return panel


def _draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], *, width: int, fill: tuple[int, int, int]) -> int:
    x, y = xy
    for line in wrap(str(text), width=width):
        draw.text((x, y), line, fill=fill)
        y += 17
    return y


def _write_handoff_qa_board(
    path: Path,
    *,
    manifest: dict[str, Any],
    source_preview_path: str | None,
    export_preview_path: str | None,
    palette_swatch_path: str | None,
) -> str:
    width = 1280
    height = 760
    margin = 28
    panel_size = (370, 300)
    canvas = Image.new("RGB", (width, height), (248, 244, 236))
    draw = ImageDraw.Draw(canvas)

    status = str(manifest.get("status") or "unknown")
    status_color = (34, 116, 74) if status == "ready" else (150, 91, 18)
    draw.text((margin, 20), "DuckAgent Paint-to-Print Handoff", fill=(26, 24, 22))
    draw.rounded_rectangle((margin, 48, margin + 220, 82), radius=12, fill=status_color)
    draw.text((margin + 16, 58), status.upper(), fill=(255, 255, 255))

    panels = [
        ("Source texture preview", source_preview_path, "No source preview"),
        ("Bambu export preview", export_preview_path, "No export preview"),
        ("Palette swatches", palette_swatch_path, "No palette swatches"),
    ]
    x = margin
    y = 112
    for label, image_path, fallback in panels:
        draw.text((x, y - 24), label, fill=(36, 34, 31))
        panel = _fit_panel(image_path, panel_size, fallback=fallback)
        canvas.paste(panel, (x, y))
        x += panel_size[0] + 28

    summary = manifest.get("summary") or {}
    artifacts = manifest.get("artifacts") or {}
    gates = manifest.get("gates") or []
    failed_required = [gate for gate in gates if gate.get("required") is True and gate.get("passed") is not True]
    y_text = 455
    draw.text((margin, y_text), "Acceptance Snapshot", fill=(26, 24, 22))
    y_text += 26
    snapshot_lines = [
        f"Palette: {summary.get('palette_size')} colors",
        f"Components: {summary.get('component_count')}",
        f"Tiny islands: {summary.get('tiny_island_count')}",
        f"Flat bottom: {summary.get('bottom_flatness_status')}",
        f"3MF: {artifacts.get('bambu_3mf_path')}",
    ]
    for line in snapshot_lines:
        y_text = _draw_wrapped(draw, line, (margin, y_text), width=74, fill=(48, 45, 40))
        y_text += 2

    x_right = 720
    y_right = 455
    draw.text((x_right, y_right), "Required Gate Results", fill=(26, 24, 22))
    y_right += 26
    visible_gates = [gate for gate in gates if gate.get("required") is True][:8]
    for gate in visible_gates:
        marker = "PASS" if gate.get("passed") is True else "REVIEW"
        color = (34, 116, 74) if gate.get("passed") is True else (150, 91, 18)
        draw.text((x_right, y_right), marker, fill=color)
        y_right = _draw_wrapped(draw, str(gate.get("summary") or gate.get("id")), (x_right + 74, y_right), width=56, fill=(48, 45, 40))
        y_right += 5

    if failed_required:
        y_right += 8
        draw.text((x_right, y_right), "DuckAgent action: hold for review", fill=(150, 91, 18))
    else:
        y_right += 8
        draw.text((x_right, y_right), "DuckAgent action: attach Bambu-ready bundle", fill=(34, 116, 74))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)


def _write_handoff_markdown(path: Path, manifest: dict[str, Any]) -> str:
    summary = manifest.get("summary") or {}
    artifacts = manifest.get("artifacts") or {}
    gates = manifest.get("gates") or []
    lines = [
        "# DuckAgent Paint-to-Print Handoff",
        "",
        f"- Status: {manifest.get('status')}",
        f"- Ready for DuckAgent handoff: {manifest.get('ready_for_duckagent_handoff')}",
        f"- Palette size: {summary.get('palette_size')}",
        f"- Component count: {summary.get('component_count')}",
        f"- Tiny islands: {summary.get('tiny_island_count')}",
        f"- Flat bottom: {summary.get('bottom_flatness_status')}",
        f"- QA board: {artifacts.get('qa_board_path')}",
        f"- 3MF: {artifacts.get('bambu_3mf_path')}",
        f"- OBJ: {artifacts.get('grouped_obj_path')}",
        f"- MTL: {artifacts.get('grouped_mtl_path')}",
        "",
        "## Gates",
        "",
    ]
    for gate in gates:
        marker = "PASS" if gate.get("passed") is True else "REVIEW"
        required = "required" if gate.get("required") is True else "advisory"
        lines.append(f"- {marker} ({required}): {gate.get('summary')}")
    lines.extend(
        [
            "",
            "## DuckAgent Contract",
            "",
            "DuckAgent should read `handoff_manifest.json`, attach the stable artifact paths, and hold operator review when any required gate fails.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _build_handoff_gates(
    *,
    production_report: dict[str, Any],
    conversion_report: dict[str, Any],
    paint_intent_report: dict[str, Any],
    source_preview_path: str | None,
    object_name: str | None,
    min_colors: int,
    max_colors: int,
) -> list[dict[str, Any]]:
    artifacts = {
        "obj": conversion_report.get("obj_path") or production_report.get("obj_path"),
        "mtl": conversion_report.get("mtl_path") or production_report.get("mtl_path"),
        "threemf": conversion_report.get("threemf_path") or production_report.get("threemf_path"),
        "preview": conversion_report.get("preview_path") or production_report.get("preview_path"),
        "palette_swatches": conversion_report.get("palette_swatch_path"),
    }
    missing_artifacts = [name for name, value in artifacts.items() if _existing_path(value) is None]
    bambu_validation = production_report.get("bambu_material_validation") or conversion_report.get("bambu_material_validation") or {}
    transfer_assessment = production_report.get("transfer_assessment") or conversion_report.get("repaired_transfer_assessment") or {}
    bottom_flatness = (((paint_intent_report.get("geometry") or {}).get("bottom_flatness")) or {})
    palette_size = int(production_report.get("palette_size") or conversion_report.get("palette_size") or 0)
    component_count = int(production_report.get("component_count") or conversion_report.get("component_count") or 0)
    tiny_island_count = int(production_report.get("tiny_island_count") or conversion_report.get("tiny_island_count") or 0)
    largest_component_share = float(production_report.get("largest_component_share") or conversion_report.get("largest_component_share") or 0.0)
    max_components = max(palette_size * 48, palette_size + 8)
    visual_color_confidence = _assess_visual_color_confidence(
        production_report=production_report,
        conversion_report=conversion_report,
        object_name=object_name,
    )

    return [
        _gate(
            "required_artifacts_exist",
            passed=not missing_artifacts,
            summary="OBJ, MTL, 3MF, preview, and palette swatches were written.",
            details={"missing": missing_artifacts, "artifacts": artifacts},
        ),
        _gate(
            "bambu_material_bundle_valid",
            passed=bambu_validation.get("ready_for_bambu") is True,
            summary="Generated OBJ/MTL/3MF files pass Bambu material bundle validation.",
            details=bambu_validation,
        ),
        _gate(
            "repaired_transfer_policy_ready",
            passed=transfer_assessment.get("ready_for_auto") is True,
            summary="Repaired transfer policy marks the conversion safe for automatic use.",
            details=transfer_assessment,
        ),
        _gate(
            "flat_bottom_support_preserved",
            passed=bottom_flatness.get("ready") is True,
            summary="Model has enough flat bottom support for duck-style printing.",
            details=bottom_flatness,
        ),
        _gate(
            "palette_size_reasonable",
            passed=int(min_colors) <= palette_size <= int(max_colors),
            summary=f"Palette size {palette_size} is within the requested {int(min_colors)}-{int(max_colors)} color range.",
            details={"palette_size": palette_size, "min_colors": int(min_colors), "max_colors": int(max_colors)},
        ),
        _gate(
            "paint_regions_not_overfragmented",
            passed=component_count <= max_components and tiny_island_count <= 96 and largest_component_share >= 0.08,
            summary="Paint regions are not over-fragmented for a first Bambu import pass.",
            details={
                "component_count": component_count,
                "max_components": max_components,
                "tiny_island_count": tiny_island_count,
                "max_tiny_islands": 96,
                "largest_component_share": largest_component_share,
                "min_largest_component_share": 0.08,
            },
        ),
        _gate(
            "visual_color_confidence",
            passed=visual_color_confidence.get("ready") is True,
            summary=(
                "Palette color families look plausible for automated handoff."
                if visual_color_confidence.get("ready") is True
                else "Palette color families look suspicious; hold for visual review before printing."
            ),
            details=visual_color_confidence,
        ),
        _gate(
            "source_preview_available",
            passed=_existing_path(source_preview_path) is not None,
            required=False,
            summary="Source preview was written for visual QA.",
            details={"source_preview_path": source_preview_path},
        ),
    ]


def run_duckagent_handoff(
    source_path: str | Path,
    *,
    texture_path: str | Path | None = None,
    out_dir: str | Path,
    object_name: str | None = None,
    repair_backend: str = "voxel_marching_cubes",
    target_face_count: int | None = 250_000,
    max_colors: int = 8,
    min_colors: int = 2,
    transfer_strategy: str = "geometry_transfer_blender_like_bake_duck_intent",
    repair_smoothing_iterations: int | None = None,
    repair_voxel_divisions: int = 128,
    paint_cleanup: bool = True,
    paint_cleanup_min_component_size: int | None = None,
    paint_cleanup_passes: int = 4,
    fail_closed: bool = True,
) -> dict[str, Any]:
    """Build the stable bundle DuckAgent can consume after model approval.

    The lower conversion remains deterministic and package-owned. This layer is
    intentionally an integration contract: manifest, QA board, gates, and stable
    artifact names for DuckAgent orchestration.
    """
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_preview_path = _write_source_preview(
        source_path,
        texture_path,
        output_dir / "source_preview.png",
    )
    production_report = run_repaired_production_conversion(
        source_path,
        texture_path=texture_path,
        out_dir=output_dir,
        object_name=object_name,
        repair_backend=repair_backend,
        target_face_count=target_face_count,
        max_colors=max_colors,
        transfer_strategy=transfer_strategy,
        repair_smoothing_iterations=repair_smoothing_iterations,
        repair_voxel_divisions=repair_voxel_divisions,
        paint_cleanup=paint_cleanup,
        paint_cleanup_min_component_size=paint_cleanup_min_component_size,
        paint_cleanup_passes=paint_cleanup_passes,
        fail_closed=fail_closed,
    )
    conversion_report = _load_json(production_report.get("conversion_report_path"))
    paint_intent_report = _load_json(production_report.get("paint_intent_report_path"))
    gates = _build_handoff_gates(
        production_report=production_report,
        conversion_report=conversion_report,
        paint_intent_report=paint_intent_report,
        source_preview_path=source_preview_path,
        object_name=object_name,
        min_colors=min_colors,
        max_colors=max_colors,
    )

    bottom_flatness = (((paint_intent_report.get("geometry") or {}).get("bottom_flatness")) or {})
    artifacts = {
        "source_preview_path": source_preview_path,
        "export_preview_path": _existing_path(conversion_report.get("preview_path") or production_report.get("preview_path")),
        "palette_swatches_path": _existing_path(conversion_report.get("palette_swatch_path")),
        "grouped_obj_path": _existing_path(conversion_report.get("obj_path") or production_report.get("obj_path")),
        "grouped_mtl_path": _existing_path(conversion_report.get("mtl_path") or production_report.get("mtl_path")),
        "bambu_3mf_path": _existing_path(conversion_report.get("threemf_path") or production_report.get("threemf_path")),
        "conversion_report_path": _existing_path(production_report.get("conversion_report_path")),
        "production_report_path": _existing_path(production_report.get("production_report_path")),
        "acceptance_summary_path": _existing_path(production_report.get("acceptance_summary_path")),
        "paint_intent_report_path": _existing_path(production_report.get("paint_intent_report_path")),
        "paint_intent_markdown_path": _existing_path(production_report.get("paint_intent_markdown_path")),
        "intermediate_repaired_geometry_path": _existing_path(production_report.get("intermediate_repaired_geometry_path")),
    }
    required_gates_passed = all(gate.get("passed") is True for gate in gates if gate.get("required") is True)
    ready = bool(production_report.get("ready_for_production") is True and required_gates_passed)
    manifest: dict[str, Any] = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "ready" if ready else "review_required",
        "ready_for_duckagent_handoff": ready,
        "source": {
            "path": str(Path(source_path).expanduser().resolve()),
            "texture_path": None if texture_path is None else str(Path(texture_path).expanduser().resolve()),
            "object_name": object_name,
        },
        "configuration": {
            "repair_backend": repair_backend,
            "target_face_count": None if target_face_count is None else int(target_face_count),
            "max_colors": int(max_colors),
            "min_colors": int(min_colors),
            "transfer_strategy": transfer_strategy,
            "repair_smoothing_iterations": production_report.get("repair_smoothing_iterations"),
            "repair_voxel_divisions": production_report.get("repair_voxel_divisions"),
            "paint_cleanup_enabled": bool(paint_cleanup),
            "paint_cleanup_passes": int(paint_cleanup_passes),
            "fail_closed": bool(fail_closed),
        },
        "summary": {
            "palette_size": production_report.get("palette_size"),
            "component_count": production_report.get("component_count"),
            "tiny_island_count": production_report.get("tiny_island_count"),
            "largest_component_share": production_report.get("largest_component_share"),
            "bottom_flatness_status": bottom_flatness.get("status"),
            "bambu_validation_status": (production_report.get("bambu_material_validation") or {}).get("status"),
            "transfer_assessment_status": (production_report.get("transfer_assessment") or {}).get("status"),
            "visual_color_confidence_status": (
                next((gate.get("details") or {} for gate in gates if gate.get("id") == "visual_color_confidence"), {}).get("status")
            ),
        },
        "gates": gates,
        "artifacts": artifacts,
        "duckagent_contract": {
            "read": "handoff_manifest.json",
            "attach_when": "ready_for_duckagent_handoff is true",
            "stable_artifact_keys": [
                "bambu_3mf_path",
                "grouped_obj_path",
                "grouped_mtl_path",
                "export_preview_path",
                "palette_swatches_path",
                "qa_board_path",
                "handoff_markdown_path",
            ],
            "hold_for_operator_when": "any required gate has passed=false",
        },
        "operator_next_action": (
            "Attach the Bambu-ready 3MF/OBJ/MTL bundle to the DuckAgent run."
            if ready
            else "Show the QA board and failed gates before publishing or printing."
        ),
        "production_report": production_report,
    }
    qa_board_path = _write_handoff_qa_board(
        output_dir / "handoff_qa_board.png",
        manifest=manifest,
        source_preview_path=artifacts["source_preview_path"],
        export_preview_path=artifacts["export_preview_path"],
        palette_swatch_path=artifacts["palette_swatches_path"],
    )
    manifest["artifacts"]["qa_board_path"] = qa_board_path
    manifest["gates"].append(
        _gate(
            "qa_board_written",
            passed=_existing_path(qa_board_path) is not None,
            required=False,
            summary="Visual QA board was written for DuckAgent/operator review.",
            details={"qa_board_path": qa_board_path},
        )
    )
    markdown_path = _write_handoff_markdown(output_dir / "handoff_summary.md", manifest)
    manifest["artifacts"]["handoff_markdown_path"] = markdown_path
    manifest_path = output_dir / "handoff_manifest.json"
    manifest["artifacts"]["handoff_manifest_path"] = str(manifest_path)
    _write_json(manifest_path, manifest)
    return manifest
