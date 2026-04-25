from __future__ import annotations

import numpy as np

from color3dconverter.color_adjustments import (
    apply_brightness_contrast,
    apply_hue_saturation,
    apply_layer_blend,
    apply_levels,
    hsv_to_rgb,
    posterize,
    remap,
    rgb_to_hsv,
)


def test_rgb_hsv_roundtrip() -> None:
    rgb = np.array([[0.9, 0.4, 0.2], [0.2, 0.7, 0.9]], dtype=np.float32)
    hsv = rgb_to_hsv(rgb)
    roundtrip = hsv_to_rgb(hsv)
    assert np.allclose(roundtrip, rgb, atol=1e-5)


def test_apply_levels_stretches_range() -> None:
    rgb = np.array([[0.1, 0.3, 0.5], [0.7, 0.8, 0.9]], dtype=np.float32)
    adjusted = apply_levels(rgb, 0.1, 0.9, gamma=1.0, out_black=0.0, out_white=1.0)
    assert np.isclose(adjusted.min(), 0.0)
    assert np.isclose(adjusted.max(), 1.0)


def test_apply_layer_blend_overlay_changes_values() -> None:
    base = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    blend = np.array([[1.0, 0.0, 0.5]], dtype=np.float32)
    result = apply_layer_blend(base, blend, "OVERLAY", 1.0)
    assert np.allclose(result, np.array([[1.0, 0.0, 0.5]], dtype=np.float32), atol=1e-5)


def test_hue_saturation_and_posterize_keep_values_bounded() -> None:
    rgb = np.array([[0.7, 0.4, 0.2], [0.2, 0.4, 0.7]], dtype=np.float32)
    adjusted = apply_hue_saturation(rgb, hue_shift=0.5, saturation=1.2, value=1.05)
    boosted = apply_brightness_contrast(adjusted, brightness=0.03, contrast=0.2)
    stepped = posterize(np.clip(boosted, 0.0, 1.0), levels=4)
    assert np.all(stepped >= 0.0)
    assert np.all(stepped <= 1.0)


def test_remap_clamps_to_output_range() -> None:
    values = np.array([-1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    remapped = remap(values, 0.0, 1.0, 0.0, 1.0)
    assert np.all(remapped >= 0.0)
    assert np.all(remapped <= 1.0)
