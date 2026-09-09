from __future__ import annotations

import numpy as np


def _to_hwc(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return np.stack([array, array, array], axis=-1)
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        return np.transpose(array[:3], (1, 2, 0))
    return array[..., :3]


def _normalize_to_unit(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def compute_cloud_shadow_ratio(
    array: np.ndarray,
    *,
    cloud_threshold: float = 0.82,
    shadow_threshold: float = 0.08,
) -> float:
    """
    Basic false-alarm suppression heuristic:
    - bright pixels => likely cloud
    - very dark pixels => likely shadow
    Returns ratio of cloud+shadow pixels in [0, 1].
    """
    img = _normalize_to_unit(_to_hwc(array))
    gray = img.mean(axis=-1)
    cloud_mask = gray >= cloud_threshold
    shadow_mask = gray <= shadow_threshold
    combined = np.logical_or(cloud_mask, shadow_mask)
    return float(np.mean(combined))


def is_cloud_or_shadow_contaminated(
    array: np.ndarray,
    *,
    max_ratio: float = 0.35,
    cloud_threshold: float = 0.82,
    shadow_threshold: float = 0.08,
) -> bool:
    ratio = compute_cloud_shadow_ratio(
        array,
        cloud_threshold=cloud_threshold,
        shadow_threshold=shadow_threshold,
    )
    return ratio > max_ratio