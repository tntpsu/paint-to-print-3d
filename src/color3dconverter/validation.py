from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_source_export_comparison(
    *,
    source_preview_path: str | Path,
    export_preview_path: str | Path,
    comparison_path: str | Path,
    source_mode: str | None = None,
    simplify_applied: bool = False,
    color_transfer_applied: bool = False,
) -> dict[str, Any]:
    from PIL import Image, ImageChops, ImageDraw, ImageOps, ImageStat

    source_preview = Path(source_preview_path).expanduser().resolve()
    export_preview = Path(export_preview_path).expanduser().resolve()
    output_path = Path(comparison_path).expanduser().resolve()
    if not source_preview.exists() or not export_preview.exists():
        raise FileNotFoundError("Source preview and export preview must both exist for comparison.")

    source_image = Image.open(source_preview).convert("RGB")
    export_image = Image.open(export_preview).convert("RGB")
    compare_size = (400, 400)
    source_fit = ImageOps.fit(source_image, compare_size, method=Image.Resampling.BICUBIC)
    export_fit = ImageOps.fit(export_image, compare_size, method=Image.Resampling.BICUBIC)

    diff = ImageChops.difference(source_fit, export_fit)
    diff_stats = ImageStat.Stat(diff)
    mean_abs_diff = round(sum(diff_stats.mean) / (len(diff_stats.mean) * 255.0), 4)
    if mean_abs_diff <= 0.09:
        assessment = "close"
        assessment_label = "Close match"
    elif mean_abs_diff <= 0.18:
        assessment = "moderate_drift"
        assessment_label = "Moderate drift"
    else:
        assessment = "poor_match"
        assessment_label = "Poor match"

    notes: list[str] = []
    if source_mode == "single_image":
        notes.append("The faceted look can come directly from the single-image source mesh when that upstream asset is already low poly.")
    if simplify_applied:
        notes.append("Geometry simplification was applied before export, so some drift may come from shape cleanup as well as palette reduction.")
    if color_transfer_applied:
        notes.append("This export uses repaired geometry with transferred source color, so the comparison is testing the reusable transfer path rather than direct texture-region labeling.")
    if assessment != "close":
        notes.append("The export still needs cleaner printable regions before it will feel trustworthy in Bambu Studio.")

    canvas = Image.new("RGB", (860, 470), (245, 241, 234))
    draw = ImageDraw.Draw(canvas)
    canvas.paste(source_fit, (20, 48))
    canvas.paste(export_fit, (440, 48))
    draw.text((20, 16), "Source Preview", fill=(28, 28, 28))
    draw.text((440, 16), "Local Bambu Export Preview", fill=(28, 28, 28))
    draw.text((20, 452), f"Assessment: {assessment_label}", fill=(28, 28, 28))
    draw.text((220, 452), f"Mean pixel drift: {mean_abs_diff:.3f}", fill=(28, 28, 28))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

    return {
        "assessment": assessment,
        "assessment_label": assessment_label,
        "mean_pixel_drift": mean_abs_diff,
        "comparison_path": str(output_path),
        "source_preview_path": str(source_preview),
        "export_preview_path": str(export_preview),
        "summary": f"{assessment_label} between the source preview and the current Bambu export preview.",
        "notes": notes,
    }


def write_bambu_validation_bundle(
    *,
    output_dir: str | Path,
    source_preview_path: str | Path | None,
    export_preview_path: str | Path,
    threemf_path: str | Path,
    obj_path: str | Path,
    probe_exports: list[dict[str, Any]] | None = None,
    source_mode: str | None = None,
    simplify_applied: bool = False,
    color_transfer_applied: bool = False,
    comparison_filename: str = "local_bambu_source_comparison.png",
    report_filename: str = "local_bambu_validation_report.json",
    markdown_filename: str = "local_bambu_validation_report.md",
) -> dict[str, Any] | None:
    output_root = Path(output_dir).expanduser().resolve()
    source_preview_text = str(source_preview_path or "").strip()
    if not source_preview_text:
        return None
    source_preview = Path(source_preview_text).expanduser().resolve()
    export_preview = Path(export_preview_path).expanduser().resolve()
    if not source_preview.exists() or not export_preview.exists():
        return None

    comparison = write_source_export_comparison(
        source_preview_path=source_preview,
        export_preview_path=export_preview,
        comparison_path=output_root / comparison_filename,
        source_mode=source_mode,
        simplify_applied=simplify_applied,
        color_transfer_applied=color_transfer_applied,
    )
    report = {
        "status": "ok",
        "comparison": comparison,
        "threemf_path": str(Path(threemf_path).expanduser().resolve()),
        "obj_path": str(Path(obj_path).expanduser().resolve()),
        "probe_exports": probe_exports or [],
        "operator_checklist": [
            "Open the local 3MF in Bambu Studio first.",
            "Compare Bambu's imported color blocks against the source/export comparison image.",
            "If the mapping looks wrong, test the 4-color and 8-color probe files before changing the converter.",
            "Use the OBJ only as a fallback geometry/material inspection path.",
        ],
    }
    report_path = output_root / report_filename
    markdown_path = output_root / markdown_filename
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown = [
        "# Bambu Validation",
        "",
        f"- Assessment: {comparison['assessment_label']}",
        f"- Mean pixel drift: {comparison['mean_pixel_drift']:.3f}",
        f"- 3MF: {report['threemf_path']}",
        f"- OBJ: {report['obj_path']}",
        "",
        "## Checklist",
    ]
    markdown.extend([f"- {item}" for item in report["operator_checklist"]])
    if probe_exports:
        markdown.extend(["", "## Probe Files"])
        markdown.extend([f"- {item.get('label') or item.get('name') or 'Probe'}: {item.get('path') or ''}" for item in probe_exports])
    if comparison.get("notes"):
        markdown.extend(["", "## Notes"])
        markdown.extend([f"- {item}" for item in comparison["notes"]])
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return {
        **comparison,
        "validation_report_path": str(report_path),
        "validation_markdown_path": str(markdown_path),
    }
