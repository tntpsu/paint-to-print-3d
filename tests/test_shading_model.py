from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from color3dconverter.shading_model import bundle_shading_models, convert_with_shading_model, train_shading_model


def test_train_shading_model_writes_model_and_report(tmp_path: Path) -> None:
    texture_path = tmp_path / "texture.png"
    texture = Image.new("RGB", (2, 2), (0, 0, 0))
    texture.putpixel((0, 0), (255, 0, 0))
    texture.putpixel((1, 0), (0, 255, 0))
    texture.putpixel((0, 1), (0, 0, 255))
    texture.putpixel((1, 1), (255, 255, 0))
    texture.save(texture_path)

    source_mtl = tmp_path / "source.mtl"
    source_mtl.write_text("newmtl Material\nmap_Kd texture.png\n", encoding="utf-8")
    source_obj = tmp_path / "source.obj"
    source_obj.write_text(
        "\n".join(
            [
                "mtllib source.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 1 1 0",
                "vt 0 1",
                "vt 1 1",
                "vt 0 0",
                "vt 1 0",
                "usemtl Material",
                "f 1/1 2/2 3/3",
                "f 2/2 4/4 3/3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    target_obj = tmp_path / "target.obj"
    target_obj.write_text(
        "\n".join(
            [
                "v 0 0 0 0.7 0.0 0.0",
                "v 1 0 0 0.0 0.7 0.0",
                "v 0 1 0 0.0 0.0 0.7",
                "v 1 1 0 0.7 0.7 0.0",
                "f 1 2 3",
                "f 2 4 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    alignment_path = tmp_path / "alignment.json"
    alignment_path.write_text(
        json.dumps(
            {
                "best_icp_cost": 0.0,
                "best_perm": [0, 1, 2],
                "best_sign": [1.0, 1.0, 1.0],
                "rotation_matrix": np.eye(3).tolist(),
                "translation": [0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )

    model_path = tmp_path / "shading_model.pkl"
    report = train_shading_model(
        pair_specs=[
            {
                "source_path": str(source_obj),
                "target_obj_path": str(target_obj),
                "alignment_json": str(alignment_path),
                "sample_size": 4,
                "seed": 7,
            }
        ],
        out_model_path=model_path,
        model_kind="ridge",
        target_kind="scalar",
        sample_size=4,
        seed=7,
    )

    assert model_path.exists()
    assert (tmp_path / "shading_model.pkl.json").exists()
    assert report["model_kind"] == "ridge"
    assert report["target_kind"] == "scalar"
    assert report["training_sample_count"] == 4
    assert report["pair_count"] == 1


def test_convert_with_shading_model_writes_vertex_color_obj(tmp_path: Path) -> None:
    texture_path = tmp_path / "texture.png"
    texture = Image.new("RGB", (2, 2), (0, 0, 0))
    texture.putpixel((0, 0), (255, 0, 0))
    texture.putpixel((1, 0), (0, 255, 0))
    texture.putpixel((0, 1), (0, 0, 255))
    texture.putpixel((1, 1), (255, 255, 0))
    texture.save(texture_path)

    source_mtl = tmp_path / "source_convert.mtl"
    source_mtl.write_text("newmtl Material\nmap_Kd texture.png\n", encoding="utf-8")
    source_obj = tmp_path / "source_convert.obj"
    source_obj.write_text(
        "\n".join(
            [
                "mtllib source_convert.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 1 1 0",
                "vt 0 1",
                "vt 1 1",
                "vt 0 0",
                "vt 1 0",
                "usemtl Material",
                "f 1/1 2/2 3/3",
                "f 2/2 4/4 3/3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    target_obj = tmp_path / "target_convert.obj"
    target_obj.write_text(
        "\n".join(
            [
                "v 0 0 0 0.7 0.0 0.0",
                "v 1 0 0 0.0 0.7 0.0",
                "v 0 1 0 0.0 0.0 0.7",
                "v 1 1 0 0.7 0.7 0.0",
                "f 1 2 3",
                "f 2 4 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    alignment_path = tmp_path / "alignment_convert.json"
    alignment_path.write_text(
        json.dumps(
            {
                "best_icp_cost": 0.0,
                "best_perm": [0, 1, 2],
                "best_sign": [1.0, 1.0, 1.0],
                "rotation_matrix": np.eye(3).tolist(),
                "translation": [0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )

    model_path = tmp_path / "convert_model.pkl"
    train_shading_model(
        pair_specs=[
            {
                "source_path": str(source_obj),
                "target_obj_path": str(target_obj),
                "alignment_json": str(alignment_path),
                "sample_size": 4,
                "seed": 9,
            }
        ],
        out_model_path=model_path,
        model_kind="ridge",
        target_kind="direct_rgb",
        sample_size=4,
        seed=9,
    )

    out_obj = tmp_path / "predicted.obj"
    report = convert_with_shading_model(
        source_path=source_obj,
        target_obj_path=target_obj,
        model_path=model_path,
        out_obj_path=out_obj,
        alignment_summary=json.loads(alignment_path.read_text(encoding="utf-8")),
    )

    assert out_obj.exists()
    assert (tmp_path / "predicted.json").exists()
    assert report["vertex_count"] == 4
    assert report["target_kind"] == "direct_rgb"
    assert report["mean_abs_total"] >= 0.0
    assert report["scalar_max"] >= report["scalar_min"]


def test_bundle_shading_models_builds_ensemble_bundle(tmp_path: Path) -> None:
    texture_path = tmp_path / "texture_bundle.png"
    texture = Image.new("RGB", (2, 2), (0, 0, 0))
    texture.putpixel((0, 0), (255, 0, 0))
    texture.putpixel((1, 0), (0, 255, 0))
    texture.putpixel((0, 1), (0, 0, 255))
    texture.putpixel((1, 1), (255, 255, 0))
    texture.save(texture_path)

    source_mtl = tmp_path / "source_bundle.mtl"
    source_mtl.write_text("newmtl Material\nmap_Kd texture_bundle.png\n", encoding="utf-8")
    source_obj = tmp_path / "source_bundle.obj"
    source_obj.write_text(
        "\n".join(
            [
                "mtllib source_bundle.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 1 1 0",
                "vt 0 1",
                "vt 1 1",
                "vt 0 0",
                "vt 1 0",
                "usemtl Material",
                "f 1/1 2/2 3/3",
                "f 2/2 4/4 3/3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    target_obj = tmp_path / "target_bundle.obj"
    target_obj.write_text(
        "\n".join(
            [
                "v 0 0 0 0.7 0.0 0.0",
                "v 1 0 0 0.0 0.7 0.0",
                "v 0 1 0 0.0 0.0 0.7",
                "v 1 1 0 0.7 0.7 0.0",
                "f 1 2 3",
                "f 2 4 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    alignment = {
        "best_icp_cost": 0.0,
        "best_perm": [0, 1, 2],
        "best_sign": [1.0, 1.0, 1.0],
        "rotation_matrix": np.eye(3).tolist(),
        "translation": [0.0, 0.0, 0.0],
    }
    alignment_path = tmp_path / "alignment_bundle.json"
    alignment_path.write_text(json.dumps(alignment), encoding="utf-8")

    model_a = tmp_path / "bundle_a.pkl"
    model_b = tmp_path / "bundle_b.pkl"
    for path, seed in ((model_a, 3), (model_b, 5)):
        train_shading_model(
            pair_specs=[
                {
                    "source_path": str(source_obj),
                    "target_obj_path": str(target_obj),
                    "alignment_json": str(alignment_path),
                    "sample_size": 4,
                    "seed": seed,
                }
            ],
            out_model_path=path,
            model_kind="et",
            target_kind="direct_rgb",
            sample_size=4,
            seed=seed,
        )

    ensemble_path = tmp_path / "ensemble.pkl"
    report = bundle_shading_models(
        model_paths=[model_a, model_b],
        out_model_path=ensemble_path,
        weights=[0.25, 0.75],
    )
    assert ensemble_path.exists()
    assert report["model_kind"] == "ensemble_mean"
    assert report["submodel_count"] == 2

    out_obj = tmp_path / "ensemble_predicted.obj"
    convert_report = convert_with_shading_model(
        source_path=source_obj,
        target_obj_path=target_obj,
        model_path=ensemble_path,
        out_obj_path=out_obj,
        alignment_summary=alignment,
    )
    assert out_obj.exists()
    assert convert_report["model_kind"] == "ensemble_mean"
    assert convert_report["mean_abs_total"] >= 0.0


def test_train_shading_model_supports_extra_trees(tmp_path: Path) -> None:
    texture_path = tmp_path / "texture_et.png"
    texture = Image.new("RGB", (2, 2), (0, 0, 0))
    texture.putpixel((0, 0), (255, 0, 0))
    texture.putpixel((1, 0), (0, 255, 0))
    texture.putpixel((0, 1), (0, 0, 255))
    texture.putpixel((1, 1), (255, 255, 0))
    texture.save(texture_path)

    source_mtl = tmp_path / "source_et.mtl"
    source_mtl.write_text("newmtl Material\nmap_Kd texture_et.png\n", encoding="utf-8")
    source_obj = tmp_path / "source_et.obj"
    source_obj.write_text(
        "\n".join(
            [
                "mtllib source_et.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 1 1 0",
                "vt 0 1",
                "vt 1 1",
                "vt 0 0",
                "vt 1 0",
                "usemtl Material",
                "f 1/1 2/2 3/3",
                "f 2/2 4/4 3/3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    target_obj = tmp_path / "target_et.obj"
    target_obj.write_text(
        "\n".join(
            [
                "v 0 0 0 0.7 0.0 0.0",
                "v 1 0 0 0.0 0.7 0.0",
                "v 0 1 0 0.0 0.0 0.7",
                "v 1 1 0 0.7 0.7 0.0",
                "f 1 2 3",
                "f 2 4 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    alignment_path = tmp_path / "alignment_et.json"
    alignment_path.write_text(
        json.dumps(
            {
                "best_icp_cost": 0.0,
                "best_perm": [0, 1, 2],
                "best_sign": [1.0, 1.0, 1.0],
                "rotation_matrix": np.eye(3).tolist(),
                "translation": [0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )

    model_path = tmp_path / "et_model.pkl"
    report = train_shading_model(
        pair_specs=[
            {
                "source_path": str(source_obj),
                "target_obj_path": str(target_obj),
                "alignment_json": str(alignment_path),
                "sample_size": 4,
                "seed": 5,
            }
        ],
        out_model_path=model_path,
        model_kind="et",
        target_kind="direct_rgb",
        sample_size=4,
        seed=5,
    )

    assert model_path.exists()
    assert report["model_kind"] == "et"
    assert report["target_kind"] == "direct_rgb"
    assert report["training_sample_count"] == 4


def test_train_and_convert_router_shading_model(tmp_path: Path) -> None:
    texture_a_path = tmp_path / "texture_router_a.png"
    texture_a = Image.new("RGB", (2, 2), (0, 0, 0))
    texture_a.putpixel((0, 0), (255, 32, 32))
    texture_a.putpixel((1, 0), (32, 255, 32))
    texture_a.putpixel((0, 1), (32, 32, 255))
    texture_a.putpixel((1, 1), (255, 255, 32))
    texture_a.save(texture_a_path)

    texture_b_path = tmp_path / "texture_router_b.png"
    texture_b = Image.new("RGB", (2, 2), (0, 0, 0))
    texture_b.putpixel((0, 0), (255, 128, 0))
    texture_b.putpixel((1, 0), (0, 255, 255))
    texture_b.putpixel((0, 1), (255, 0, 255))
    texture_b.putpixel((1, 1), (255, 255, 255))
    texture_b.save(texture_b_path)

    source_a_mtl = tmp_path / "source_router_a.mtl"
    source_a_mtl.write_text("newmtl Material\nmap_Kd texture_router_a.png\n", encoding="utf-8")
    source_a_obj = tmp_path / "source_router_a.obj"
    source_a_obj.write_text(
        "\n".join(
            [
                "mtllib source_router_a.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 1 1 0",
                "vt 0 1",
                "vt 1 1",
                "vt 0 0",
                "vt 1 0",
                "usemtl Material",
                "f 1/1 2/2 3/3",
                "f 2/2 4/4 3/3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    source_b_mtl = tmp_path / "source_router_b.mtl"
    source_b_mtl.write_text("newmtl Material\nmap_Kd texture_router_b.png\n", encoding="utf-8")
    source_b_obj = tmp_path / "source_router_b.obj"
    source_b_obj.write_text(
        "\n".join(
            [
                "mtllib source_router_b.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 1 1 0",
                "vt 0 1",
                "vt 1 1",
                "vt 0 0",
                "vt 1 0",
                "usemtl Material",
                "f 1/1 2/2 3/3",
                "f 2/2 4/4 3/3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    target_a_obj = tmp_path / "target_router_a.obj"
    target_a_obj.write_text(
        "\n".join(
            [
                "v 0 0 0 0.9 0.1 0.1",
                "v 1 0 0 0.1 0.9 0.1",
                "v 0 1 0 0.1 0.1 0.9",
                "v 1 1 0 0.9 0.9 0.2",
                "f 1 2 3",
                "f 2 4 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    target_b_obj = tmp_path / "target_router_b.obj"
    target_b_obj.write_text(
        "\n".join(
            [
                "v 0 0 0 1.0 0.6 0.1",
                "v 1 0 0 0.1 1.0 1.0",
                "v 0 1 0 1.0 0.1 1.0",
                "v 1 1 0 1.0 1.0 1.0",
                "f 1 2 3",
                "f 2 4 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    alignment_path = tmp_path / "alignment_router.json"
    alignment_path.write_text(
        json.dumps(
            {
                "best_icp_cost": 0.0,
                "best_perm": [0, 1, 2],
                "best_sign": [1.0, 1.0, 1.0],
                "rotation_matrix": np.eye(3).tolist(),
                "translation": [0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )

    model_path = tmp_path / "router_model.pkl"
    report = train_shading_model(
        pair_specs=[
            {
                "source_path": str(source_a_obj),
                "target_obj_path": str(target_a_obj),
                "alignment_json": str(alignment_path),
                "sample_size": 4,
                "seed": 5,
            },
            {
                "source_path": str(source_b_obj),
                "target_obj_path": str(target_b_obj),
                "alignment_json": str(alignment_path),
                "sample_size": 4,
                "seed": 6,
            },
        ],
        out_model_path=model_path,
        model_kind="et_router",
        target_kind="direct_rgb",
        sample_size=4,
        seed=5,
    )

    out_obj = tmp_path / "router_predicted.obj"
    convert_report = convert_with_shading_model(
        source_path=source_b_obj,
        target_obj_path=target_b_obj,
        model_path=model_path,
        out_obj_path=out_obj,
        alignment_summary=json.loads(alignment_path.read_text(encoding="utf-8")),
    )

    assert report["model_kind"] == "et_router"
    assert report["pair_count"] == 2
    assert out_obj.exists()
    assert convert_report["model_kind"] == "et_router"
    assert "router" in convert_report
    assert len(convert_report["router"]["experts"]) >= 1


def test_train_and_convert_residual_router_shading_model(tmp_path: Path) -> None:
    texture_a_path = tmp_path / "texture_res_router_a.png"
    texture_a = Image.new("RGB", (2, 2), (0, 0, 0))
    texture_a.putpixel((0, 0), (255, 32, 32))
    texture_a.putpixel((1, 0), (32, 255, 32))
    texture_a.putpixel((0, 1), (32, 32, 255))
    texture_a.putpixel((1, 1), (255, 255, 32))
    texture_a.save(texture_a_path)

    texture_b_path = tmp_path / "texture_res_router_b.png"
    texture_b = Image.new("RGB", (2, 2), (0, 0, 0))
    texture_b.putpixel((0, 0), (255, 128, 0))
    texture_b.putpixel((1, 0), (0, 255, 255))
    texture_b.putpixel((0, 1), (255, 0, 255))
    texture_b.putpixel((1, 1), (255, 255, 255))
    texture_b.save(texture_b_path)

    source_a_mtl = tmp_path / "source_res_router_a.mtl"
    source_a_mtl.write_text("newmtl Material\nmap_Kd texture_res_router_a.png\n", encoding="utf-8")
    source_a_obj = tmp_path / "source_res_router_a.obj"
    source_a_obj.write_text(
        "\n".join(
            [
                "mtllib source_res_router_a.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 1 1 0",
                "vt 0 1",
                "vt 1 1",
                "vt 0 0",
                "vt 1 0",
                "usemtl Material",
                "f 1/1 2/2 3/3",
                "f 2/2 4/4 3/3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    source_b_mtl = tmp_path / "source_res_router_b.mtl"
    source_b_mtl.write_text("newmtl Material\nmap_Kd texture_res_router_b.png\n", encoding="utf-8")
    source_b_obj = tmp_path / "source_res_router_b.obj"
    source_b_obj.write_text(
        "\n".join(
            [
                "mtllib source_res_router_b.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 1 1 0",
                "vt 0 1",
                "vt 1 1",
                "vt 0 0",
                "vt 1 0",
                "usemtl Material",
                "f 1/1 2/2 3/3",
                "f 2/2 4/4 3/3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    target_a_obj = tmp_path / "target_res_router_a.obj"
    target_a_obj.write_text(
        "\n".join(
            [
                "v 0 0 0 0.9 0.1 0.1",
                "v 1 0 0 0.1 0.9 0.1",
                "v 0 1 0 0.1 0.1 0.9",
                "v 1 1 0 0.9 0.9 0.2",
                "f 1 2 3",
                "f 2 4 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    target_b_obj = tmp_path / "target_res_router_b.obj"
    target_b_obj.write_text(
        "\n".join(
            [
                "v 0 0 0 1.0 0.6 0.1",
                "v 1 0 0 0.1 1.0 1.0",
                "v 0 1 0 1.0 0.1 1.0",
                "v 1 1 0 1.0 1.0 1.0",
                "f 1 2 3",
                "f 2 4 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    alignment_path = tmp_path / "alignment_res_router.json"
    alignment_path.write_text(
        json.dumps(
            {
                "best_icp_cost": 0.0,
                "best_perm": [0, 1, 2],
                "best_sign": [1.0, 1.0, 1.0],
                "rotation_matrix": np.eye(3).tolist(),
                "translation": [0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )

    model_path = tmp_path / "residual_router_model.pkl"
    report = train_shading_model(
        pair_specs=[
            {
                "source_path": str(source_a_obj),
                "target_obj_path": str(target_a_obj),
                "alignment_json": str(alignment_path),
                "sample_size": 4,
                "seed": 5,
            },
            {
                "source_path": str(source_b_obj),
                "target_obj_path": str(target_b_obj),
                "alignment_json": str(alignment_path),
                "sample_size": 4,
                "seed": 6,
            },
        ],
        out_model_path=model_path,
        model_kind="et_residual_router",
        target_kind="direct_rgb",
        sample_size=4,
        seed=5,
    )

    out_obj = tmp_path / "residual_router_predicted.obj"
    convert_report = convert_with_shading_model(
        source_path=source_b_obj,
        target_obj_path=target_b_obj,
        model_path=model_path,
        out_obj_path=out_obj,
        alignment_summary=json.loads(alignment_path.read_text(encoding="utf-8")),
    )

    assert report["model_kind"] == "et_residual_router"
    assert "residual_blend" in report
    assert out_obj.exists()
    assert convert_report["model_kind"] == "et_residual_router"
    assert "router" in convert_report
    assert "residual_blend" in convert_report["router"]


def test_train_shading_model_supports_hgb_and_mlp(tmp_path: Path) -> None:
    texture_path = tmp_path / "texture_alt_models.png"
    texture = Image.new("RGB", (2, 2), (0, 0, 0))
    texture.putpixel((0, 0), (255, 0, 0))
    texture.putpixel((1, 0), (0, 255, 0))
    texture.putpixel((0, 1), (0, 0, 255))
    texture.putpixel((1, 1), (255, 255, 0))
    texture.save(texture_path)

    source_mtl = tmp_path / "source_alt_models.mtl"
    source_mtl.write_text("newmtl Material\nmap_Kd texture_alt_models.png\n", encoding="utf-8")
    source_obj = tmp_path / "source_alt_models.obj"
    source_obj.write_text(
        "\n".join(
            [
                "mtllib source_alt_models.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 1 1 0",
                "vt 0 1",
                "vt 1 1",
                "vt 0 0",
                "vt 1 0",
                "usemtl Material",
                "f 1/1 2/2 3/3",
                "f 2/2 4/4 3/3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    target_obj = tmp_path / "target_alt_models.obj"
    target_obj.write_text(
        "\n".join(
            [
                "v 0 0 0 0.7 0.0 0.0",
                "v 1 0 0 0.0 0.7 0.0",
                "v 0 1 0 0.0 0.0 0.7",
                "v 1 1 0 0.7 0.7 0.0",
                "f 1 2 3",
                "f 2 4 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    alignment_path = tmp_path / "alignment_alt_models.json"
    alignment_path.write_text(
        json.dumps(
            {
                "best_icp_cost": 0.0,
                "best_perm": [0, 1, 2],
                "best_sign": [1.0, 1.0, 1.0],
                "rotation_matrix": np.eye(3).tolist(),
                "translation": [0.0, 0.0, 0.0],
            }
        ),
        encoding="utf-8",
    )

    for model_kind in ("hgb", "mlp"):
        model_path = tmp_path / f"{model_kind}_model.pkl"
        report = train_shading_model(
            pair_specs=[
                {
                    "source_path": str(source_obj),
                    "target_obj_path": str(target_obj),
                    "alignment_json": str(alignment_path),
                    "sample_size": 4,
                    "seed": 5,
                }
            ],
            out_model_path=model_path,
            model_kind=model_kind,
            target_kind="direct_rgb",
            sample_size=4,
            seed=5,
        )
        assert model_path.exists()
        assert report["model_kind"] == model_kind
