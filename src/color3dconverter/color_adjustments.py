from __future__ import annotations

import numpy as np


def apply_levels(
    rgb: np.ndarray,
    in_black: float,
    in_white: float,
    gamma: float,
    out_black: float,
    out_white: float,
) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    in_range = max(float(in_white) - float(in_black), 1e-12)
    result = (values - float(in_black)) / in_range
    result = np.clip(result, 0.0, 1.0)
    result = np.power(result, 1.0 / max(float(gamma), 1e-6))
    result = result * (float(out_white) - float(out_black)) + float(out_black)
    return result.astype(np.float32, copy=False)


def apply_brightness_contrast(rgb: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    result = (values - 0.5) * (float(contrast) + 1.0) + 0.5 + float(brightness)
    return result.astype(np.float32, copy=False)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    if values.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    r, g, b = values[:, 0], values[:, 1], values[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc
    safe_delta = np.where(delta > 0.0, delta, 1.0)
    s = np.divide(delta, maxc, out=np.zeros_like(delta), where=maxc > 0.0)

    rc = (maxc - r) / safe_delta
    gc = (maxc - g) / safe_delta
    bc = (maxc - b) / safe_delta

    h = np.where(
        r == maxc,
        bc - gc,
        np.where(g == maxc, 2.0 + rc - bc, 4.0 + gc - rc),
    )
    h = (h / 6.0) % 1.0
    h = np.where(delta > 0.0, h, 0.0)
    return np.column_stack([h, s, v]).astype(np.float32)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    values = np.asarray(hsv, dtype=np.float32)
    if values.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    h, s, v = values[:, 0], values[:, 1], values[:, 2]
    i = (h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - np.floor(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.column_stack([r, g, b]).astype(np.float32)


def apply_hue_saturation(rgb: np.ndarray, hue_shift: float, saturation: float, value: float) -> np.ndarray:
    hsv = rgb_to_hsv(np.asarray(rgb, dtype=np.float32))
    hsv[:, 0] = (hsv[:, 0] + float(hue_shift) - 0.5) % 1.0
    hsv[:, 1] = np.clip(hsv[:, 1] * float(saturation), 0.0, 1.0)
    hsv[:, 2] = np.clip(hsv[:, 2] * float(value), 0.0, 1.0)
    return hsv_to_rgb(hsv)


def apply_layer_blend(base: np.ndarray, blend: np.ndarray, mode: str, factor: float) -> np.ndarray:
    base_values = np.asarray(base, dtype=np.float32)
    blend_values = np.asarray(blend, dtype=np.float32)
    weight = float(factor)
    if mode == "MIX":
        result = base_values + (blend_values - base_values) * weight
    elif mode == "MULTIPLY":
        result = base_values * (1.0 - weight) + base_values * blend_values * weight
    elif mode == "ADD":
        result = base_values + blend_values * weight
    elif mode == "SUBTRACT":
        result = base_values - blend_values * weight
    elif mode == "OVERLAY":
        low = 2.0 * base_values * blend_values
        high = 1.0 - 2.0 * (1.0 - base_values) * (1.0 - blend_values)
        overlay = np.where(base_values < 0.5, low, high)
        result = base_values + (overlay - base_values) * weight
    elif mode == "SCREEN":
        screen = 1.0 - (1.0 - base_values) * (1.0 - blend_values)
        result = base_values + (screen - base_values) * weight
    else:
        result = base_values
    return np.asarray(result, dtype=np.float32)


def posterize(rgb: np.ndarray, levels: int) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    steps = max(int(levels), 2)
    return np.round(values * (steps - 1)) / float(steps - 1)


def remap(values: np.ndarray, min0: float, max0: float, min1: float, max1: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    in_range = max(float(max0) - float(min0), 1e-12)
    result = (array - float(min0)) / in_range
    result = np.clip(result, 0.0, 1.0)
    result = result * (float(max1) - float(min1)) + float(min1)
    return np.asarray(result, dtype=np.float32)
