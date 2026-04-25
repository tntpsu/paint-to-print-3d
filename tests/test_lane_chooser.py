from __future__ import annotations

import json
from pathlib import Path

from color3dconverter.lane_chooser import choose_conversion_lane, normalize_lane_candidate


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_choose_conversion_lane_prefers_ready_same_mesh_production(tmp_path: Path) -> None:
    same_mesh = _write_json(
        tmp_path / "production_report.json",
        {
            "status": "ready",
            "ready_for_production": True,
            "quality_threshold": 0.02,
            "selected_candidate": {
                "label": "posterize4_r8",
                "mean_pixel_drift": 0.01,
                "selected_dir": str(tmp_path / "selected"),
                "export_preview_path": str(tmp_path / "selected" / "region_preview.png"),
            },
        },
    )
    provider = _write_json(
        tmp_path / "provider_report.json",
        {
            "status": "ok",
            "conversion_lane": "provider_baked_repaired_same_mesh",
            "face_count": 24_000,
            "component_count": 18,
            "tiny_island_count": 2,
            "provider_bake_assessment": {
                "status": "ready_for_auto",
                "ready_for_auto": True,
                "reasons": [],
            },
        },
    )

    report = choose_conversion_lane([provider, same_mesh], out_report=tmp_path / "lane_choice_report.json")

    assert report["status"] == "ready"
    assert report["selected_lane"]["lane"] == "same_mesh_production"
    assert report["ready_for_operator_approval"] is True
    assert Path(report["lane_choice_report_path"]).exists()
    assert len(report["rejected_lanes"]) == 1
    assert report["rejected_lanes"][0]["lane"] == "provider_baked_repaired_same_mesh"


def test_choose_conversion_lane_uses_provider_bake_before_repaired_transfer(tmp_path: Path) -> None:
    provider = _write_json(
        tmp_path / "provider_report.json",
        {
            "status": "ok",
            "conversion_lane": "provider_baked_repaired_same_mesh",
            "face_count": 24_000,
            "component_count": 18,
            "tiny_island_count": 2,
            "provider_bake_assessment": {
                "status": "ready_for_auto",
                "ready_for_auto": True,
                "reasons": [],
            },
        },
    )
    repaired = _write_json(
        tmp_path / "repaired_report.json",
        {
            "status": "ok",
            "conversion_lane": "repaired_geometry_region_transfer",
            "face_count": 25_000,
            "component_count": 20,
            "tiny_island_count": 3,
            "repaired_transfer_assessment": {
                "status": "ready_for_auto",
                "ready_for_auto": True,
                "reasons": [],
            },
        },
    )

    report = choose_conversion_lane([repaired, provider])

    assert report["status"] == "ready"
    assert report["selected_lane"]["lane"] == "provider_baked_repaired_same_mesh"
    assert report["rejected_lanes"][0]["lane"] == "repaired_geometry_region_transfer"


def test_choose_conversion_lane_preserves_rejection_reasons_when_all_lanes_fail(tmp_path: Path) -> None:
    provider = _write_json(
        tmp_path / "provider_report.json",
        {
            "status": "ok",
            "conversion_lane": "provider_baked_repaired_same_mesh",
            "face_count": 1_170_124,
            "component_count": 5_040,
            "tiny_island_count": 4_483,
            "provider_bake_assessment": {
                "status": "needs_review",
                "ready_for_auto": False,
                "reasons": [
                    "target face count 1,170,124 is above the auto-use threshold 400,000",
                    "provider baked base-color texture looks like a normal map, not printable color art",
                ],
            },
            "provider_bake_texture_diagnostics": {
                "texture_role": "suspected_normal_map",
            },
        },
    )
    raw = _write_json(
        tmp_path / "raw_report.json",
        {
            "status": "ok",
            "source_format": "glb",
            "face_count": 1200,
            "preview_path": str(tmp_path / "region_preview.png"),
        },
    )

    report = choose_conversion_lane([raw, provider])

    assert report["status"] == "needs_review"
    assert report["selected_lane"] is None
    assert report["ready_for_operator_approval"] is False
    provider_candidate = next(item for item in report["rejected_lanes"] if item["lane"] == "provider_baked_repaired_same_mesh")
    assert any("normal map" in reason for reason in provider_candidate["rejection_reasons"])
    raw_candidate = next(item for item in report["rejected_lanes"] if item["lane"] == "same_mesh_conversion_report")
    assert any("no production or lane assessment gate" in reason for reason in raw_candidate["rejection_reasons"])


def test_normalize_lane_candidate_rejects_production_report_failure(tmp_path: Path) -> None:
    candidate = normalize_lane_candidate(
        tmp_path / "production_report.json",
        {
            "status": "rejected",
            "ready_for_production": False,
            "message": "Best same-mesh candidate did not meet the production quality threshold.",
            "selected_candidate": {"label": "baseline_r16"},
        },
    )

    assert candidate["lane"] == "same_mesh_production"
    assert candidate["ready_for_auto"] is False
    assert any("did not meet" in reason for reason in candidate["rejection_reasons"])
