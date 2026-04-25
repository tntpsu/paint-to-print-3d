from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LANE_PRIORITY: dict[str, int] = {
    "same_mesh_production": 0,
    "provider_baked_repaired_same_mesh": 1,
    "repaired_geometry_region_transfer": 2,
    "same_mesh_conversion_report": 3,
    "unknown": 99,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser().resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Lane report must be a JSON object: {report_path}")
    return {"path": str(report_path), "payload": payload}


def _artifact_paths(report: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "obj_path",
        "vertex_color_obj_path",
        "mtl_path",
        "threemf_path",
        "preview_path",
        "palette_swatch_path",
        "palette_csv_path",
        "report_path",
    ]
    return {key: report.get(key) for key in keys if report.get(key)}


def _production_candidate(report_path: str, report: dict[str, Any]) -> dict[str, Any]:
    selected = dict(report.get("selected_candidate") or {})
    ready = bool(report.get("ready_for_production"))
    reasons = []
    if not ready:
        message = str(report.get("message") or "").strip()
        reasons.append(message or "same-mesh production gate did not pass")

    return {
        "label": "same_mesh_production",
        "lane": "same_mesh_production",
        "report_path": report_path,
        "ready_for_auto": ready,
        "assessment_status": "ready_for_auto" if ready else "needs_review",
        "rejection_reasons": reasons,
        "priority": LANE_PRIORITY["same_mesh_production"],
        "mean_pixel_drift": selected.get("mean_pixel_drift"),
        "artifact_paths": {
            "selected_dir": selected.get("selected_dir"),
            "export_preview_path": selected.get("export_preview_path"),
            "comparison_path": selected.get("comparison_path"),
            "production_report_path": report.get("production_report_path") or report_path,
        },
        "summary": {
            "status": report.get("status"),
            "quality_threshold": report.get("quality_threshold"),
            "candidate_label": selected.get("label"),
            "assessment": selected.get("assessment"),
        },
    }


def _assessed_conversion_candidate(report_path: str, report: dict[str, Any], *, assessment_key: str) -> dict[str, Any]:
    lane = str(report.get("conversion_lane") or "unknown")
    assessment = dict(report.get(assessment_key) or {})
    ready = bool(assessment.get("ready_for_auto"))
    reasons = [str(item) for item in (assessment.get("reasons") or []) if str(item).strip()]
    if not ready and not reasons:
        reasons.append(f"{assessment_key} did not mark the lane ready for automatic selection")

    return {
        "label": lane,
        "lane": lane,
        "report_path": report_path,
        "ready_for_auto": ready,
        "assessment_status": str(assessment.get("status") or ("ready_for_auto" if ready else "needs_review")),
        "rejection_reasons": reasons,
        "priority": LANE_PRIORITY.get(lane, LANE_PRIORITY["unknown"]),
        "face_count": report.get("face_count"),
        "component_count": report.get("component_count"),
        "tiny_island_count": report.get("tiny_island_count"),
        "largest_component_share": report.get("largest_component_share"),
        "artifact_paths": _artifact_paths(report),
        "summary": {
            "strategy": report.get("strategy"),
            "palette_size": report.get("palette_size"),
            "texture_role": (report.get("provider_bake_texture_diagnostics") or {}).get("texture_role"),
            "recommendation": assessment.get("recommendation"),
        },
    }


def _raw_conversion_candidate(report_path: str, report: dict[str, Any]) -> dict[str, Any]:
    lane = str(report.get("conversion_lane") or "same_mesh_conversion_report")
    return {
        "label": lane,
        "lane": lane,
        "report_path": report_path,
        "ready_for_auto": False,
        "assessment_status": "needs_review",
        "rejection_reasons": [
            "conversion report has no production or lane assessment gate; run convert-production or an assessed lane first"
        ],
        "priority": LANE_PRIORITY.get(lane, LANE_PRIORITY["same_mesh_conversion_report"]),
        "face_count": report.get("face_count"),
        "component_count": report.get("component_count"),
        "tiny_island_count": report.get("tiny_island_count"),
        "largest_component_share": report.get("largest_component_share"),
        "artifact_paths": _artifact_paths(report),
        "summary": {
            "status": report.get("status"),
            "strategy": report.get("strategy"),
            "palette_size": report.get("palette_size"),
        },
    }


def normalize_lane_candidate(report_path: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    resolved_path = str(Path(report_path).expanduser().resolve())
    if "ready_for_production" in report:
        return _production_candidate(resolved_path, report)

    lane = str(report.get("conversion_lane") or "")
    if lane == "provider_baked_repaired_same_mesh":
        return _assessed_conversion_candidate(resolved_path, report, assessment_key="provider_bake_assessment")
    if lane == "repaired_geometry_region_transfer":
        return _assessed_conversion_candidate(resolved_path, report, assessment_key="repaired_transfer_assessment")
    return _raw_conversion_candidate(resolved_path, report)


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, float, int, int]:
    priority = _safe_int(candidate.get("priority"), LANE_PRIORITY["unknown"])
    drift = _safe_float(candidate.get("mean_pixel_drift"), 999.0)
    component_count = _safe_int(candidate.get("component_count"), 999_999_999)
    tiny_island_count = _safe_int(candidate.get("tiny_island_count"), 999_999_999)
    return (priority, drift, component_count, tiny_island_count)


def choose_conversion_lane(
    report_paths: list[str | Path],
    *,
    out_report: str | Path | None = None,
) -> dict[str, Any]:
    loaded_reports = [_load_report(path) for path in report_paths]
    candidates = [
        normalize_lane_candidate(item["path"], item["payload"])
        for item in loaded_reports
    ]
    candidates.sort(key=_candidate_sort_key)

    ready_candidates = [candidate for candidate in candidates if candidate.get("ready_for_auto") is True]
    selected_lane = ready_candidates[0] if ready_candidates else None
    rejected_lanes = [
        candidate
        for candidate in candidates
        if selected_lane is None or candidate.get("report_path") != selected_lane.get("report_path")
    ]

    report = {
        "status": "ready" if selected_lane else "needs_review",
        "mode": "propose_only",
        "ready_for_operator_approval": bool(selected_lane),
        "selected_lane": selected_lane,
        "rejected_lanes": rejected_lanes,
        "candidates": candidates,
        "selection_policy": {
            "priority_order": [
                "same_mesh_production",
                "provider_baked_repaired_same_mesh",
                "repaired_geometry_region_transfer",
            ],
            "notes": [
                "Same-mesh production must pass the visual drift gate before it can win.",
                "Provider-baked repaired output must pass texture and structure assessment before it can win.",
                "Local repaired-transfer output is only eligible after its repaired-transfer assessment passes.",
                "This chooser is report-only and does not mutate or publish assets.",
            ],
        },
    }

    if out_report:
        output_path = Path(out_report).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["lane_choice_report_path"] = str(output_path)

    return report
