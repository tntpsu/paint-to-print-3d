from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .export_obj_vertex_colors import write_obj_with_per_vertex_colors
from .provider_oracle import (
    ProviderOracleVariant,
    _apply_alignment_summary,
    _compute_candidate_hits,
    _load_target_vertex_color_obj,
    _normalize_points,
    _prepare_source_for_variant,
    _sample_texture_mode,
    _shade_features,
    _shade_target_scalars,
    _vertex_color_metrics,
)
from .model_io import load_textured_model
from .surface_transfer import barycentric_weights


def _resolve_alignment_summary(
    target_obj_path: str | Path,
    alignment_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    if alignment_summary is not None:
        return alignment_summary
    alignment_path = Path(target_obj_path).with_suffix(".alignment.json")
    if alignment_path.exists():
        return json.loads(alignment_path.read_text(encoding="utf-8"))
    raise ValueError("alignment_summary is required unless a sibling .alignment.json file exists.")


def _build_pair_shading_inputs(
    *,
    source_path: str | Path,
    target_obj_path: str | Path,
    alignment_summary: dict[str, Any] | None = None,
    sample_indexes: np.ndarray | None = None,
    sample_size: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    loaded = load_textured_model(source_path)
    target_positions, target_faces, target_vertex_colors = _load_target_vertex_color_obj(target_obj_path)
    alignment_summary = _resolve_alignment_summary(target_obj_path, alignment_summary)

    aligned = _apply_alignment_summary(loaded.positions, alignment_summary)
    variant = ProviderOracleVariant(
        "sample",
        method="nearest_surface_uv",
        sampling_mode="bilinear",
        uv_flip_y=True,
        candidate_count=8,
        pad_pixels=4,
    )
    prepared = _prepare_source_for_variant(loaded, aligned, variant)

    if sample_indexes is None:
        rng = np.random.default_rng(int(seed))
        sample_indexes = rng.choice(len(target_positions), size=min(int(sample_size), len(target_positions)), replace=False)
    else:
        sample_indexes = np.asarray(sample_indexes, dtype=np.int64)
    target_points = _normalize_points(target_positions)[sample_indexes].astype(np.float32)
    target_mesh = trimesh.Trimesh(vertices=target_positions, faces=target_faces, process=False)
    target_normals = np.asarray(target_mesh.vertex_normals, dtype=np.float32)[sample_indexes]
    expected_colors = np.asarray(target_vertex_colors[sample_indexes], dtype=np.float32)

    triangles, _, indexes, closest_points, distances = _compute_candidate_hits(
        prepared,
        target_points,
        candidate_count=variant.candidate_count,
    )
    triangle_uvs = np.asarray(prepared.texcoords, dtype=np.float32)[np.asarray(prepared.faces, dtype=np.int64)]
    source_face_normals = np.asarray(trimesh.triangles.normals(triangles)[0], dtype=np.float32)

    sampled_uv = np.zeros((len(target_points), 2), dtype=np.float32)
    projection_features = np.zeros((len(target_points), 13), dtype=np.float32)
    for row_index in range(len(target_points)):
        candidate_faces = indexes[row_index]
        best_row = int(np.argmin(distances[row_index]))
        face_idx = int(candidate_faces[best_row])
        hit_point = closest_points[row_index, best_row]
        weights = barycentric_weights(hit_point, triangles[face_idx])
        sampled_uv[row_index] = np.sum(triangle_uvs[face_idx] * weights[:, None], axis=0, dtype=np.float32)
        row_distances = np.asarray(distances[row_index], dtype=np.float32)
        if len(row_distances) > 1:
            sorted_distances = np.sort(row_distances)
            second_distance = float(sorted_distances[1])
        else:
            second_distance = float(row_distances[0]) if len(row_distances) else 0.0
        best_delta = np.asarray(target_points[row_index] - hit_point, dtype=np.float32)
        target_normal = np.asarray(target_normals[row_index], dtype=np.float32)
        source_normals = np.asarray(source_face_normals[candidate_faces], dtype=np.float32)
        target_norm = float(np.linalg.norm(target_normal))
        if target_norm > 1e-8 and len(source_normals):
            alignments = np.abs(source_normals @ (target_normal / target_norm))
        else:
            alignments = np.zeros((len(candidate_faces),), dtype=np.float32)
        u = float(sampled_uv[row_index, 0])
        v = float(sampled_uv[row_index, 1])
        projection_features[row_index] = np.array(
            [
                u,
                v,
                u * v,
                u * u,
                v * v,
                float(row_distances[best_row]) if len(row_distances) else 0.0,
                float(row_distances.mean()) if len(row_distances) else 0.0,
                float(row_distances.std()) if len(row_distances) else 0.0,
                second_distance,
                float(alignments.max()) if len(alignments) else 0.0,
                float(alignments.mean()) if len(alignments) else 0.0,
                float(alignments.std()) if len(alignments) else 0.0,
                float(np.linalg.norm(best_delta)),
            ],
            dtype=np.float32,
        )

    base_colors = _sample_texture_mode(prepared.texture_rgb, sampled_uv, sampling_mode="bilinear").astype(np.float32)
    features = np.concatenate(
        [
            _shade_features(target_points, target_normals, base_colors),
            projection_features,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    return {
        "features": features.astype(np.float32),
        "base_colors": base_colors.astype(np.float32),
        "expected_colors": expected_colors.astype(np.float32),
        "sample_indexes": np.asarray(sample_indexes, dtype=np.int64),
        "alignment_summary": alignment_summary,
        "target_positions": np.asarray(target_positions[sample_indexes], dtype=np.float32),
        "target_faces": np.asarray(target_faces, dtype=np.int64),
        "full_target_positions": np.asarray(target_positions, dtype=np.float32),
        "full_target_faces": np.asarray(target_faces, dtype=np.int64),
        "full_target_vertex_colors": np.asarray(target_vertex_colors, dtype=np.uint8),
    }


def sample_provider_pair_shading_data(
    *,
    source_path: str | Path,
    target_obj_path: str | Path,
    sample_size: int = 5000,
    seed: int = 42,
    alignment_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sampled = _build_pair_shading_inputs(
        source_path=source_path,
        target_obj_path=target_obj_path,
        sample_size=sample_size,
        seed=seed,
        alignment_summary=alignment_summary,
    )
    shade_targets = _shade_target_scalars(sampled["base_colors"], sampled["expected_colors"])
    sampled["shade_targets"] = shade_targets.astype(np.float32)
    return sampled


def _target_array_for_kind(sampled: dict[str, Any], target_kind: str) -> np.ndarray:
    if target_kind == "scalar":
        targets = sampled["shade_targets"]
    elif target_kind == "direct_rgb":
        targets = sampled["expected_colors"]
    elif target_kind == "residual_rgb":
        targets = sampled["expected_colors"] - sampled["base_colors"]
    elif target_kind == "channel_scale":
        targets = np.clip(
            np.divide(sampled["expected_colors"], np.maximum(sampled["base_colors"], 1.0)),
            0.0,
            4.0,
        ).astype(np.float32)
    else:
        raise ValueError(f"Unsupported target kind: {target_kind}")
    return np.asarray(targets, dtype=np.float32)


def _feature_signature(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        return np.zeros((0,), dtype=np.float32)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    return np.concatenate([means, stds, mins, maxs], axis=0).astype(np.float32, copy=False)


def _build_regressor(model_kind: str, seed: int) -> Any:
    if model_kind == "ridge":
        return Ridge(alpha=1.0)
    if model_kind == "rf":
        return RandomForestRegressor(
            n_estimators=120,
            max_depth=18,
            random_state=int(seed),
            n_jobs=-1,
        )
    if model_kind in {"et", "et_router", "et_residual_router"}:
        return ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=int(seed),
            n_jobs=-1,
        )
    if model_kind == "hgb":
        base = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=300,
            max_depth=8,
            learning_rate=0.08,
            random_state=int(seed),
        )
        return MultiOutputRegressor(base)
    if model_kind == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        solver="adam",
                        learning_rate_init=1e-3,
                        max_iter=400,
                        early_stopping=False,
                        random_state=int(seed),
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported shading model kind: {model_kind}")


def train_shading_model(
    *,
    pair_specs: list[dict[str, Any]],
    out_model_path: str | Path,
    model_kind: str = "rf",
    target_kind: str = "scalar",
    sample_size: int = 5000,
    seed: int = 42,
) -> dict[str, Any]:
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    pair_reports: list[dict[str, Any]] = []
    pair_training_payloads: list[dict[str, Any]] = []

    for spec in pair_specs:
        alignment_summary = spec.get("alignment_summary")
        alignment_json = spec.get("alignment_json")
        if alignment_summary is None and alignment_json:
            alignment_summary = json.loads(Path(str(alignment_json)).expanduser().resolve().read_text(encoding="utf-8"))
        sampled = sample_provider_pair_shading_data(
            source_path=spec["source_path"],
            target_obj_path=spec["target_obj_path"],
            sample_size=int(spec.get("sample_size", sample_size)),
            seed=int(spec.get("seed", seed)),
            alignment_summary=alignment_summary,
        )
        feature_rows.append(sampled["features"])
        targets = _target_array_for_kind(sampled, target_kind)
        target_rows.append(np.asarray(targets, dtype=np.float32))
        resolved_source = str(Path(spec["source_path"]).expanduser().resolve())
        resolved_target = str(Path(spec["target_obj_path"]).expanduser().resolve())
        pair_report = {
            "source_path": resolved_source,
            "target_obj_path": resolved_target,
            "sample_count": int(len(sampled["shade_targets"])),
        }
        pair_reports.append(pair_report)
        pair_training_payloads.append(
            {
                "report": pair_report,
                "features": np.asarray(sampled["features"], dtype=np.float32),
                "targets": np.asarray(targets, dtype=np.float32),
                "signature": _feature_signature(sampled["features"]),
            }
        )

    features = np.concatenate(feature_rows, axis=0).astype(np.float32, copy=False) if feature_rows else np.zeros((0, 0), dtype=np.float32)
    targets = np.concatenate(target_rows, axis=0).astype(np.float32, copy=False) if target_rows else np.zeros((0,), dtype=np.float32)
    if len(targets) == 0:
        raise ValueError("No training samples were collected for the shading model.")

    model = _build_regressor(model_kind, seed)
    model.fit(features, targets)

    model_path = Path(out_model_path).expanduser().resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_kind": model_kind,
        "target_kind": target_kind,
        "model": model,
    }
    if model_kind in {"et_router", "et_residual_router"}:
        pair_models: list[dict[str, Any]] = []
        for pair_index, pair_payload in enumerate(pair_training_payloads):
            pair_model = _build_regressor("et", int(seed) + pair_index + 1)
            if model_kind == "et_residual_router":
                base_pred = np.asarray(model.predict(pair_payload["features"]), dtype=np.float32)
                pair_targets = np.asarray(pair_payload["targets"], dtype=np.float32) - base_pred
            else:
                pair_targets = np.asarray(pair_payload["targets"], dtype=np.float32)
            pair_model.fit(pair_payload["features"], pair_targets)
            pair_models.append(
                {
                    **pair_payload["report"],
                    "signature": pair_payload["signature"],
                    "model": pair_model,
                }
            )
        payload["pair_models"] = pair_models
        payload["router_top_k"] = min(2, len(pair_models))
        payload["shared_blend"] = 0.35 if model_kind == "et_router" else 1.0
        payload["residual_blend"] = 0.8 if model_kind == "et_residual_router" else 1.0
    with model_path.open("wb") as handle:
        pickle.dump(payload, handle)

    report = {
        "model_path": str(model_path),
        "model_kind": model_kind,
        "target_kind": target_kind,
        "training_sample_count": int(len(targets)),
        "feature_dim": int(features.shape[1]),
        "pair_count": int(len(pair_reports)),
        "pairs": pair_reports,
    }
    if model_kind in {"et_router", "et_residual_router"}:
        report["router_top_k"] = int(payload["router_top_k"])
        report["shared_blend"] = float(payload["shared_blend"])
        if model_kind == "et_residual_router":
            report["residual_blend"] = float(payload["residual_blend"])
    report_path = model_path.with_suffix(model_path.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def bundle_shading_models(
    *,
    model_paths: list[str | Path],
    out_model_path: str | Path,
    weights: list[float] | None = None,
) -> dict[str, Any]:
    resolved_paths = [Path(path).expanduser().resolve() for path in model_paths]
    if len(resolved_paths) < 2:
        raise ValueError("At least two model paths are required to build an ensemble bundle.")
    bundles = [load_shading_model(path) for path in resolved_paths]
    if weights is None:
        weight_array = np.full((len(bundles),), 1.0 / len(bundles), dtype=np.float32)
    else:
        weight_array = np.asarray(weights, dtype=np.float32)
        if len(weight_array) != len(bundles):
            raise ValueError("weights must match the number of model paths")
        total = float(weight_array.sum())
        if total <= 1e-8:
            raise ValueError("weights must sum to a positive value")
        weight_array = weight_array / total

    payload = {
        "model_kind": "ensemble_mean",
        "submodels": bundles,
        "weights": weight_array.astype(np.float32).tolist(),
        "model_paths": [str(path) for path in resolved_paths],
    }
    model_path = Path(out_model_path).expanduser().resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump(payload, handle)

    report = {
        "model_path": str(model_path),
        "model_kind": "ensemble_mean",
        "submodel_count": int(len(bundles)),
        "weights": payload["weights"],
        "model_paths": payload["model_paths"],
    }
    report_path = model_path.with_suffix(model_path.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_shading_model(model_path: str | Path) -> dict[str, Any]:
    resolved = Path(model_path).expanduser().resolve()
    with resolved.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "model" not in payload:
        if not (isinstance(payload, dict) and str(payload.get("model_kind") or "") == "ensemble_mean" and payload.get("submodels")):
            raise ValueError(f"{resolved} does not contain a valid shading model bundle.")
    return payload


def _predict_float_colors_from_bundle(
    bundle: dict[str, Any],
    *,
    features: np.ndarray,
    base_colors: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    target_kind = str(bundle.get("target_kind") or "scalar")
    model_kind = str(bundle.get("model_kind") or "rf")
    if model_kind == "ensemble_mean":
        submodels = list(bundle.get("submodels") or [])
        if not submodels:
            raise ValueError("ensemble_mean bundle is missing submodels")
        weights = np.asarray(bundle.get("weights") or [], dtype=np.float32)
        if len(weights) != len(submodels):
            weights = np.full((len(submodels),), 1.0 / len(submodels), dtype=np.float32)
        else:
            weights = weights / np.maximum(weights.sum(), 1e-8)
        blended = np.zeros_like(base_colors, dtype=np.float32)
        component_reports: list[dict[str, Any]] = []
        for weight, submodel in zip(weights.tolist(), submodels):
            component_float, component_report = _predict_float_colors_from_bundle(
                submodel,
                features=features,
                base_colors=base_colors,
            )
            blended += component_float * float(weight)
            component_reports.append(
                {
                    "model_kind": str(submodel.get("model_kind") or ""),
                    "target_kind": str(submodel.get("target_kind") or ""),
                    "weight": float(weight),
                    "router": component_report,
                }
            )
        return np.clip(blended, 0.0, 255.0), {"ensemble": component_reports}

    raw_pred = np.asarray(bundle["model"].predict(features), dtype=np.float32)
    router_report: dict[str, Any] | None = None
    if model_kind in {"et_router", "et_residual_router"}:
        pair_models = list(bundle.get("pair_models") or [])
        query_signature = _feature_signature(features)
        candidate_distances: list[tuple[float, int]] = []
        for idx, pair_payload in enumerate(pair_models):
            signature = np.asarray(pair_payload.get("signature"), dtype=np.float32)
            if len(signature) != len(query_signature):
                continue
            distance = float(np.linalg.norm(signature - query_signature))
            candidate_distances.append((distance, idx))
        candidate_distances.sort(key=lambda item: item[0])
        top_k = max(1, min(int(bundle.get("router_top_k") or 2), len(candidate_distances)))
        chosen = candidate_distances[:top_k]
        if chosen:
            inv = np.array([1.0 / max(item[0], 1e-6) for item in chosen], dtype=np.float32)
            expert_weights = inv / np.maximum(inv.sum(), 1e-6)
            expert_pred = np.zeros_like(raw_pred, dtype=np.float32)
            expert_details: list[dict[str, Any]] = []
            for weight, (distance, idx) in zip(expert_weights.tolist(), chosen):
                pair_payload = pair_models[idx]
                pair_pred = np.asarray(pair_payload["model"].predict(features), dtype=np.float32)
                expert_pred += pair_pred * float(weight)
                expert_details.append(
                    {
                        "source_path": str(pair_payload.get("source_path") or ""),
                        "target_obj_path": str(pair_payload.get("target_obj_path") or ""),
                        "distance": float(distance),
                        "weight": float(weight),
                    }
                )
            if model_kind == "et_residual_router":
                residual_blend = float(bundle.get("residual_blend") or 0.8)
                raw_pred = raw_pred + (residual_blend * expert_pred)
                router_report = {
                    "shared_blend": float(bundle.get("shared_blend") or 1.0),
                    "residual_blend": residual_blend,
                    "experts": expert_details,
                }
            else:
                shared_blend = float(bundle.get("shared_blend") or 0.35)
                raw_pred = (shared_blend * raw_pred) + ((1.0 - shared_blend) * expert_pred)
                router_report = {
                    "shared_blend": shared_blend,
                    "experts": expert_details,
                }
    if target_kind == "scalar":
        scalars = np.clip(raw_pred.reshape((-1,)), 0.0, 2.0).astype(np.float32)
        predicted_float = np.clip(base_colors * scalars[:, None], 0.0, 255.0)
    elif target_kind == "direct_rgb":
        predicted_float = np.clip(raw_pred, 0.0, 255.0)
    elif target_kind == "residual_rgb":
        predicted_float = np.clip(base_colors + raw_pred, 0.0, 255.0)
    elif target_kind == "channel_scale":
        predicted_float = np.clip(base_colors * np.clip(raw_pred, 0.0, 4.0), 0.0, 255.0)
    else:
        raise ValueError(f"Unsupported model target kind: {target_kind}")
    return predicted_float, router_report


def convert_with_shading_model(
    *,
    source_path: str | Path,
    target_obj_path: str | Path,
    model_path: str | Path,
    out_obj_path: str | Path,
    alignment_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bundle = load_shading_model(model_path)
    full_target_positions, full_target_faces, full_target_vertex_colors = _load_target_vertex_color_obj(target_obj_path)
    sampled = _build_pair_shading_inputs(
        source_path=source_path,
        target_obj_path=target_obj_path,
        alignment_summary=alignment_summary,
        sample_indexes=np.arange(len(full_target_positions), dtype=np.int64),
    )
    features = np.asarray(sampled["features"], dtype=np.float32)
    base_colors = np.asarray(sampled["base_colors"], dtype=np.float32)
    model_kind = str(bundle.get("model_kind") or "rf")
    target_kind = str(bundle.get("target_kind") or ("direct_rgb" if model_kind == "ensemble_mean" else "scalar"))
    predicted_float, router_report = _predict_float_colors_from_bundle(
        bundle,
        features=features,
        base_colors=base_colors,
    )
    predicted = np.clip(np.rint(predicted_float), 0, 255).astype(np.uint8)
    output_path = write_obj_with_per_vertex_colors(
        out_obj_path,
        full_target_positions,
        full_target_faces,
        predicted.astype(np.float32) / 255.0,
        object_name="ShadedVertexColorMesh",
    )
    metrics = _vertex_color_metrics(predicted, full_target_vertex_colors)
    report = {
        "model_path": str(Path(model_path).expanduser().resolve()),
        "out_obj_path": str(output_path),
        "model_kind": model_kind,
        "target_kind": target_kind,
        "vertex_count": int(len(predicted)),
        "scalar_min": 0.0,
        "scalar_max": 0.0,
        "scalar_mean": 0.0,
        **metrics,
    }
    if router_report is not None:
        report["router"] = router_report
    report_path = Path(out_obj_path).expanduser().resolve().with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
