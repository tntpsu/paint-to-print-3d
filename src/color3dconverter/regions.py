from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans


def build_texture_regions(texture_rgb: np.ndarray, n_regions: int) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(texture_rgb, dtype=np.uint8).reshape((-1, 3))
    cluster_count = max(1, min(int(n_regions), len(pixels)))
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    region_labels = labels.reshape(texture_rgb.shape[0], texture_rgb.shape[1]).astype(np.int32)
    palette = np.clip(np.rint(kmeans.cluster_centers_), 0, 255).astype(np.uint8)
    return region_labels, palette


def clean_texture_regions(region_labels: np.ndarray, n_regions: int, *, kernel_size: int = 3) -> np.ndarray:
    cleaned = np.asarray(region_labels, dtype=np.int32).copy()
    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    for label in range(int(n_regions)):
        mask = cleaned == label
        if not np.any(mask):
            continue
        opened = ndimage.binary_opening(mask, structure=structure)
        closed = ndimage.binary_closing(opened, structure=structure)
        cleaned[closed] = label
    return cleaned


def assign_faces_to_texture_regions(
    faces: np.ndarray,
    texcoords: np.ndarray,
    region_labels: np.ndarray,
) -> np.ndarray:
    face_labels: list[int] = []
    max_x = int(region_labels.shape[1] - 1)
    max_y = int(region_labels.shape[0] - 1)
    for face in np.asarray(faces, dtype=np.int64):
        sampled: list[int] = []
        for u, v in np.asarray(texcoords[face], dtype=np.float32):
            x = int(u * max_x)
            y = int(v * max_y)
            x = max(0, min(x, max_x))
            y = max(0, min(y, max_y))
            sampled.append(int(region_labels[y, x]))
        face_labels.append(int(Counter(sampled).most_common(1)[0][0]))
    return np.asarray(face_labels, dtype=np.int32)
