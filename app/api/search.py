from __future__ import annotations

from typing import Any, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel, Field, field_validator
from qdrant_client.http import models as qmodels
from app.services.qdrant_store import QdrantStore, get_qdrant_store
from app.core.config import settings
from app.ml.embedder import get_embedder

router = APIRouter(prefix="/search", tags=["search"])


class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    top_k: int = Field(default=settings.SEARCH_DEFAULT_TOP_K, ge=1, le=100)
    date: Optional[str] = None
    sensor: Optional[str] = None


class SearchHit(BaseModel):
    score: float
    tile_id: str
    bbox: dict[str, Any]
    date: str
    sensor: str
    image_path: str


class SearchResponse(BaseModel):
    mode: Literal["text", "image"]
    query: Optional[str] = None
    top_k: int
    results: list[SearchHit]


def _build_filter(date: Optional[str], sensor: Optional[str]) -> Optional[qmodels.Filter]:
    must: list[qmodels.FieldCondition] = []
    if date:
        must.append(qmodels.FieldCondition(key="date", match=qmodels.MatchValue(value=date)))
    if sensor:
        must.append(qmodels.FieldCondition(key="sensor", match=qmodels.MatchValue(value=sensor)))
    if not must:
        return None
    return qmodels.Filter(must=must)


def _to_hits(scored_points) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for sp in scored_points:
        payload = sp.payload or {}
        hits.append(
            SearchHit(
                score=float(sp.score),
                tile_id=str(payload.get("tile_id", "")),
                bbox=payload.get("bbox", {}),
                date=str(payload.get("date", "")),
                sensor=str(payload.get("sensor", "")),
                image_path=str(payload.get("image_path", "")),
            )
        )
    return hits


# 1. GET Handler (Frontend query: /api/v1/search/text?query=water&limit=3)
@router.get("/text", response_model=SearchResponse)
async def search_by_text_get(
    query: str = Query(..., min_length=1, max_length=512),
    limit: int = Query(default=settings.SEARCH_DEFAULT_TOP_K, ge=1, le=100),
    date: Optional[str] = Query(default=None),
    sensor: Optional[str] = Query(default=None),
    store: QdrantStore = Depends(get_qdrant_store)
) -> SearchResponse:
    embedder = get_embedder()
    vector = await embedder.embed_text_async(query.strip())
    query_filter = _build_filter(date, sensor)

    # Note: top_k use kiya hai limit ki jagah
    scored = await store.run_sync(
        store.search,
        vector,
        top_k=limit,
        query_filter=query_filter
    )
    return SearchResponse(
        mode="text",
        query=query,
        top_k=limit,
        results=_to_hits(scored)
    )


# 2. POST Handler (Existing API spec compatibility)
@router.post("/text", response_model=SearchResponse)
async def search_by_text(
    payload: TextSearchRequest,
    store: QdrantStore = Depends(get_qdrant_store)
) -> SearchResponse:
    embedder = get_embedder()
    vector = await embedder.embed_text_async(payload.query.strip())
    query_filter = _build_filter(payload.date, payload.sensor)

    scored = await store.run_sync(
        store.search,
        vector,
        top_k=payload.top_k,
        query_filter=query_filter
    )
    return SearchResponse(
        mode="text",
        query=payload.query,
        top_k=payload.top_k,
        results=_to_hits(scored)
    )