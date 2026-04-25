from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark import _write_texture_source_preview
from .color_adjustments import posterize
from .model_io import LoadedTexturedMesh, load_textured_model
from .pipeline import convert_loaded_mesh_to_color_assets
from .validation import write_source_export_comparison


@dataclass(frozen=True)
class ProductionCandidate:
    label: str
    n_regions: int
    posterize_levels: int | None = None


DEFAULT_PRODUCTION_CANDIDATES: tuple[ProductionCandidate, ...] = (
    ProductionCandidate("baseline_r16", n_regions=16, posterize_levels=None),
    ProductionCandidate("posterize4_r16", n_regions=16, posterize_levels=4),
    ProductionCandidate("posterize4_r8", n_regions=8, posterize_levels=4),
    ProductionCandidate("baseline_r5", n_regions=5, posterize_levels=None),
)


def _posterize_texture(texture_rgb: np.ndarray, levels: int) -> np.ndarray:
    rgb = np.asarray(texture_rgb, dtype=np.float32) / 255.0
    adjusted = posterize(rgb, int(levels))
    return np.clip(np.rint(adjusted * 255.0), 0, 255).astype(np.uint8)


def _with_texture(loaded: LoadedTexturedMesh, texture_rgb: np.ndarray) -> LoadedTexturedMesh:
    return LoadedTexturedMesh(
        mesh=loaded.mesh,
        positions=np.asarray(loaded.positions, dtype=np.float32),
        faces=np.asarray(loaded.faces, dtype=np.int64),
        texcoords=np.asarray(loaded.texcoords, dtype=np.float32),
        texture_rgb=np.asarray(texture_rgb, dtype=np.uint8),
        source_path=loaded.source_path,
        texture_path=loaded.texture_path,
        source_format=loaded.source_format,
        normal_texture_rgb=None if loaded.normal_texture_rgb is None else np.asarray(loaded.normal_texture_rgb, dtype=np.uint8),
        orm_texture_rgb=None if loaded.orm_texture_rgb is None else np.asarray(loaded.orm_texture_rgb, dtype=np.uint8),
        base_color_factor=None if loaded.base_color_factor is None else np.asarray(loaded.base_color_factor, dtype=np.float32),
        metallic_factor=float(loaded.metallic_factor),
        roughness_factor=float(loaded.roughness_factor),
        normal_scale=float(loaded.normal_scale),
    )


def _replace_prefixes(value: Any, old_prefix: str, new_prefix: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_prefixes(item, old_prefix, new_prefix) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_prefixes(item, old_prefix, new_prefix) for item in value]
    if isinstance(value, str) and value.startswith(old_prefix):
        return new_prefix + value[len(old_prefix) :]
    return value


def _promote_candidate(candidate_dir: Path, selected_dir: Path) -> None:
    if selected_dir.exists():
        shutil.rmtree(selected_dir)
    shutil.copytree(candidate_dir, selected_dir)
    report_path = selected_dir / "conversion_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rewritten = _replace_prefixes(report, str(candidate_dir), str(selected_dir))
        report_path.write_text(json.dumps(rewritten, indent=2), encoding="utf-8")


def run_production_conversion(
    source_path: str | Path,
    *,
    texture_path: str | Path | None = None,
    out_dir: str | Path,
    object_name: str | None = None,
    quality_threshold: float = 0.02,
    fail_closed: bool = True,
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_dir = Path(out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_textured_model(source_path, texture_path=texture_path)
    if loaded.source_format not in {"obj", "objzip", "glb"}:
        raise ValueError(f"Unsupported source format for production conversion: {loaded.source_format}")

    candidate_specs = [
        ProductionCandidate(**spec) for spec in candidates
    ] if candidates else list(DEFAULT_PRODUCTION_CANDIDATES)

    candidate_rows: list[dict[str, Any]] = []
    candidates_dir = output_dir / "_candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    for candidate in candidate_specs:
        candidate_dir = candidates_dir / candidate.label
        candidate_dir.mkdir(parents=True, exist_ok=True)

        candidate_loaded = loaded
        transformed_texture_path = None
        if candidate.posterize_levels is not None:
            transformed_texture = _posterize_texture(loaded.texture_rgb, candidate.posterize_levels)
            transformed_texture_path = candidate_dir / "transformed_texture.png"
            from PIL import Image

            Image.fromarray(transformed_texture, mode="RGB").save(transformed_texture_path)
            candidate_loaded = _with_texture(loaded, transformed_texture)

        source_preview_path = candidate_dir / "source_preview.png"
        _write_texture_source_preview(
            source_preview_path,
            positions=candidate_loaded.positions,
            faces=candidate_loaded.faces,
            texcoords=candidate_loaded.texcoords,
            texture_rgb=candidate_loaded.texture_rgb,
        )

        report = convert_loaded_mesh_to_color_assets(
            candidate_loaded,
            out_dir=candidate_dir,
            n_regions=int(candidate.n_regions),
            strategy="legacy_fast_face_labels",
            object_name=object_name,
        )
        comparison = write_source_export_comparison(
            source_preview_path=source_preview_path,
            export_preview_path=Path(report["preview_path"]),
            comparison_path=candidate_dir / "source_export_comparison.png",
            source_mode="same_mesh_production",
            simplify_applied=False,
            color_transfer_applied=False,
        )

        row = {
            "label": candidate.label,
            "strategy": "legacy_fast_face_labels",
            "n_regions": int(candidate.n_regions),
            "posterize_levels": candidate.posterize_levels,
            "source_preview_path": str(source_preview_path),
            "transformed_texture_path": str(transformed_texture_path) if transformed_texture_path else None,
            "report_path": str(candidate_dir / "conversion_report.json"),
            "export_preview_path": str(report["preview_path"]),
            "comparison_path": str(candidate_dir / "source_export_comparison.png"),
            "assessment": comparison["assessment"],
            "assessment_label": comparison["assessment_label"],
            "mean_pixel_drift": float(comparison["mean_pixel_drift"]),
        }
        candidate_rows.append(row)

    candidate_rows.sort(key=lambda item: (float(item["mean_pixel_drift"]), int(item["n_regions"])))
    best_candidate = candidate_rows[0]
    selected_dir = output_dir / "selected"
    best_candidate_dir = candidates_dir / str(best_candidate["label"])
    _promote_candidate(best_candidate_dir, selected_dir)

    ready_for_production = float(best_candidate["mean_pixel_drift"]) <= float(quality_threshold)
    status = "ready" if ready_for_production else "rejected"
    production_report = {
        "status": status,
        "ready_for_production": bool(ready_for_production),
        "quality_threshold": float(quality_threshold),
        "fail_closed": bool(fail_closed),
        "scope": {
            "same_mesh_only": True,
            "supported_source_formats": ["obj", "objzip", "glb"],
            "unsupported": [
                "automatic repaired-geometry transfer",
                "cross-topology color transfer",
            ],
        },
        "selected_candidate": {
            **best_candidate,
            "selected_dir": str(selected_dir),
        },
        "candidates": candidate_rows,
    }
    report_path = output_dir / "production_report.json"
    report_path.write_text(json.dumps(production_report, indent=2), encoding="utf-8")

    if fail_closed and not ready_for_production:
        return {
            **production_report,
            "message": "Best same-mesh candidate did not meet the production quality threshold.",
            "production_report_path": str(report_path),
        }

    return {
        **production_report,
        "message": "Production conversion completed.",
        "production_report_path": str(report_path),
    }
