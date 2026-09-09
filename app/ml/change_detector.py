from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.core.config import settings
from app.ml.embedder import OpenCLIPEmbedder
from app.services.cloud_mask import is_cloud_or_shadow_contaminated
from app.services.normalizer import histogram_match_t2_to_t1


@dataclass(frozen=True)
class ChangeResult:
    t1_tile_id: str
    t2_tile_id: str
    drift: float
    similarity: float
    confidence: float
    suppressed: bool
    reason: Optional[str] = None


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    a = np.asarray(v1, dtype=np.float32)
    b = np.asarray(v2, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def embedding_drift(v1: list[float], v2: list[float]) -> float:
    sim = cosine_similarity(v1, v2)
    return float(max(0.0, 1.0 - sim))


def detect_change_between_tiles(
    *,
    embedder: OpenCLIPEmbedder,
    t1_tile_id: str,
    t2_tile_id: str,
    t1_array: np.ndarray,
    t2_array: np.ndarray,
    t1_vector: Optional[list[float]] = None,
    t2_vector: Optional[list[float]] = None,
) -> ChangeResult:
    if is_cloud_or_shadow_contaminated(
        t1_array,
        max_ratio=settings.CLOUD_SHADOW_MAX_RATIO,
        cloud_threshold=settings.CLOUD_BRIGHTNESS_THRESHOLD,
        shadow_threshold=settings.SHADOW_BRIGHTNESS_THRESHOLD,
    ):
        return ChangeResult(
            t1_tile_id=t1_tile_id,
            t2_tile_id=t2_tile_id,
            drift=0.0,
            similarity=1.0,
            confidence=0.0,
            suppressed=True,
            reason="T1 cloud/shadow contamination",
        )

    if is_cloud_or_shadow_contaminated(
        t2_array,
        max_ratio=settings.CLOUD_SHADOW_MAX_RATIO,
        cloud_threshold=settings.CLOUD_BRIGHTNESS_THRESHOLD,
        shadow_threshold=settings.SHADOW_BRIGHTNESS_THRESHOLD,
    ):
        return ChangeResult(
            t1_tile_id=t1_tile_id,
            t2_tile_id=t2_tile_id,
            drift=0.0,
            similarity=1.0,
            confidence=0.0,
            suppressed=True,
            reason="T2 cloud/shadow contamination",
        )

    t2_norm = histogram_match_t2_to_t1(t1_array, t2_array)

    vec1 = t1_vector if t1_vector is not None else embedder.embed_image_array(t1_array)
    vec2 = t2_vector if t2_vector is not None else embedder.embed_image_array(t2_norm)

    sim = cosine_similarity(vec1, vec2)
    drift = embedding_drift(vec1, vec2)
    confidence = float(min(1.0, drift / max(settings.CHANGE_DRIFT_THRESHOLD, 1e-6)))

    return ChangeResult(
        t1_tile_id=t1_tile_id,
        t2_tile_id=t2_tile_id,
        drift=drift,
        similarity=sim,
        confidence=confidence,
        suppressed=False,
        reason=None,
    )