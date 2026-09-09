from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np
import rasterio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from qdrant_client.http import models as qmodels

from app.core.config import settings
from app.core.database import ReviewQueueRepository
from app.ml.change_detector import detect_change_between_tiles
from app.ml.embedder import get_embedder
from app.services.qdrant_store import get_qdrant_store
from app.services.tiler import iter_tiles_from_geotiff

router = APIRouter(prefix="/change", tags=["change"])


class ChangeDetectRequest(BaseModel):
    image_path_t1: str
    image_path_t2: str
    date_t1: str
    date_t2: str
    sensor: str = "unknown"
    top_k: int = Field(default=settings.CHANGE_DEFAULT_TOP_K, ge=1, le=500)
    drift_threshold: float = Field(default=settings.CHANGE_DRIFT_THRESHOLD, ge=0.0, le=1.0)
    enqueue_for_review: bool = True


class ChangeCandidate(BaseModel):
    t1_tile_id: str
    t2_tile_id: str
    drift: float
    similarity: float
    confidence: float
    suppressed: bool
    reason: Optional[str] = None
    bbox: dict[str, Any]


class ChangeDetectResponse(BaseModel):
    date_t1: str
    date_t2: str
    candidates: list[ChangeCandidate]
    review_items_created: int


def _read_tile_from_geotiff(image_path: str, row: int, col: int, tile_size: int) -> np.ndarray:
    with rasterio.open(image_path) as src:
        from rasterio.windows import Window

        window = Window(col, row, tile_size, tile_size)
        return src.read(window=window)


def _match_t2_record(t1_payload: dict, t2_records: list) -> Optional[dict]:
    t1_row = t1_payload.get("row")
    t1_col = t1_payload.get("col")
    t1_path = t1_payload.get("image_path")

    for rec in t2_records:
        payload = rec.payload or {}
        if payload.get("row") == t1_row and payload.get("col") == t1_col:
            return {"record": rec, "payload": payload}
    return None


@router.post("/detect", response_model=ChangeDetectResponse)
async def detect_changes(payload: ChangeDetectRequest) -> ChangeDetectResponse:
    embedder = get_embedder()
    store = get_qdrant_store()

    t2_records = await store.run_sync(
        store.scroll_by_payload,
        must=[
            qmodels.FieldCondition(key="date", match=qmodels.MatchValue(value=payload.date_t2)),
            qmodels.FieldCondition(key="sensor", match=qmodels.MatchValue(value=payload.sensor)),
        ],
        limit=5000,
    )

    if not t2_records:
        t2_records = await store.run_sync(
            store.scroll_by_payload,
            must=[qmodels.FieldCondition(key="date", match=qmodels.MatchValue(value=payload.date_t2))],
            limit=5000,
        )

    t1_tiles = list(
        iter_tiles_from_geotiff(
            payload.image_path_t1,
            date=payload.date_t1,
            sensor=payload.sensor,
        )
    )

    candidates: list[ChangeCandidate] = []
    review_rows: list[dict] = []

    for t1_tile in t1_tiles:
        t1_qdrant = await store.run_sync(store.get_by_tile_id, t1_tile.tile_id)
        t1_vector = None
        t1_payload = {
            "row": t1_tile.row,
            "col": t1_tile.col,
            "image_path": t1_tile.image_path,
            "bbox": t1_tile.bbox,
        }

        if t1_qdrant and t1_qdrant.vector is not None:
            t1_vector = list(t1_qdrant.vector)  # type: ignore[arg-type]
            t1_payload = t1_qdrant.payload or t1_payload

        match = _match_t2_record(t1_payload, t2_records)
        if not match:
            continue

        t2_rec = match["record"]
        t2_payload = match["payload"]
        t2_vector = list(t2_rec.vector) if t2_rec.vector is not None else None  # type: ignore[arg-type]

        t2_tile_id = str(t2_payload.get("tile_id", ""))
        t1_array = t1_tile.array
        t2_array = _read_tile_from_geotiff(
            str(t2_payload.get("image_path", payload.image_path_t2)),
            int(t2_payload.get("row", t1_tile.row)),
            int(t2_payload.get("col", t1_tile.col)),
            settings.TILE_SIZE,
        )

        result = detect_change_between_tiles(
            embedder=embedder,
            t1_tile_id=t1_tile.tile_id,
            t2_tile_id=t2_tile_id,
            t1_array=t1_array,
            t2_array=t2_array,
            t1_vector=t1_vector,
            t2_vector=t2_vector,
        )

        if result.suppressed:
            continue
        if result.drift < payload.drift_threshold:
            continue

        bbox = t1_payload.get("bbox", t1_tile.bbox)
        candidates.append(
            ChangeCandidate(
                t1_tile_id=result.t1_tile_id,
                t2_tile_id=result.t2_tile_id,
                drift=result.drift,
                similarity=result.similarity,
                confidence=result.confidence,
                suppressed=result.suppressed,
                reason=result.reason,
                bbox=bbox,
            )
        )

        if payload.enqueue_for_review:
            review_rows.append(
                {
                    "tile_id": f"{result.t1_tile_id}__{result.t2_tile_id}",
                    "t1_tile_id": result.t1_tile_id,
                    "t2_tile_id": result.t2_tile_id,
                    "status": "PENDING",
                    "confidence": result.confidence,
                    "drift_score": result.drift,
                    "remarks": "Auto-enqueued from change detection",
                    "bbox_json": json.dumps(bbox),
                    "date_t1": payload.date_t1,
                    "date_t2": payload.date_t2,
                }
            )

    candidates.sort(key=lambda c: c.drift, reverse=True)
    candidates = candidates[: payload.top_k]

    review_created = 0
    if payload.enqueue_for_review and review_rows:
        filtered_ids = {f"{c.t1_tile_id}__{c.t2_tile_id}" for c in candidates}
        filtered_rows = [r for r in review_rows if r["tile_id"] in filtered_ids]
        review_created = ReviewQueueRepository.bulk_create(filtered_rows)

    return ChangeDetectResponse(
        date_t1=payload.date_t1,
        date_t2=payload.date_t2,
        candidates=candidates,
        review_items_created=review_created,
    )