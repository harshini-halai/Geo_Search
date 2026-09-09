from __future__ import annotations

import numpy as np
from skimage.exposure import match_histograms


def _to_chw(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array[np.newaxis, ...]
    if array.shape[0] in (1, 3, 4):
        return array
    return np.transpose(array, (2, 0, 1))


def _to_float01(array: np.ndarray) -> np.ndarray:
    arr = array.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def histogram_match_t2_to_t1(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """
    Per-band histogram matching from T2 -> T1 reference.
    Reduces radiometric differences before embedding drift.
    """
    t1_chw = _to_chw(_to_float01(t1))
    t2_chw = _to_chw(_to_float01(t2))
    bands = min(t1_chw.shape[0], t2_chw.shape[0], 3)

    matched_bands = []
    for b in range(bands):
        ref = t1_chw[b]
        src = t2_chw[b]
        matched = match_histograms(src, ref, channel_axis=None)
        matched_bands.append(matched)

    matched = np.stack(matched_bands, axis=0)
    return matched