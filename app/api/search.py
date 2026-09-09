from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator
from qdrant_client.http import models as qmodels

from app.core.config import settings
from app.ml.embedder import get_embedder
from app.services.qdrant_store import get_qdrant_store

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


@router.post("/text", response_model=SearchResponse)
async def search_by_text(payload: TextSearchRequest) -> SearchResponse:
    embedder = get_embedder()
    store = get_qdrant_store()

    vector = await embedder.embed_text_async(payload.query.strip())
    query_filter = _build_filter(payload.date, payload.sensor)

    scored = await store.run_sync(
        store.search,
        vector,
        top_k=payload.top_k,
        query_filter=query_filter,
    )

    return SearchResponse(
        mode="text",
        query=payload.query,
        top_k=payload.top_k,
        results=_to_hits(scored),
    )


class ImagePathSearchRequest(BaseModel):
    image_path: str = Field(..., min_length=1)
    top_k: int = Field(default=settings.SEARCH_DEFAULT_TOP_K, ge=1, le=100)
    date: Optional[str] = None
    sensor: Optional[str] = None

    @field_validator("image_path")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("image_path cannot be empty")
        return v


@router.post("/image-path", response_model=SearchResponse)
async def search_by_image_path(payload: ImagePathSearchRequest) -> SearchResponse:
    embedder = get_embedder()
    store = get_qdrant_store()

    try:
        vector = await embedder.embed_image_path.__wrapped__  # type: ignore[attr-defined]
    except AttributeError:
        vector = None

    if vector is None:
        from pathlib import Path

        path = Path(payload.image_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {payload.image_path}")
        loop = __import__("asyncio").get_running_loop()
        vector = await loop.run_in_executor(None, embedder.embed_image_path, str(path))

    query_filter = _build_filter(payload.date, payload.sensor)
    scored = await store.run_sync(
        store.search,
        vector,
        top_k=payload.top_k,
        query_filter=query_filter,
    )

    return SearchResponse(
        mode="image",
        query=None,
        top_k=payload.top_k,
        results=_to_hits(scored),
    )


@router.post("/image-upload", response_model=SearchResponse)
async def search_by_image_upload(
    file: UploadFile = File(...),
    top_k: int = Form(default=settings.SEARCH_DEFAULT_TOP_K),
    date: Optional[str] = Form(default=None),
    sensor: Optional[str] = Form(default=None),
) -> SearchResponse:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")

    embedder = get_embedder()
    store = get_qdrant_store()
    data = await file.read()
    vector = await embedder.embed_image_bytes_async(data)

    query_filter = _build_filter(date, sensor)
    scored = await store.run_sync(
        store.search,
        vector,
        top_k=top_k,
        query_filter=query_filter,
    )

    return SearchResponse(
        mode="image",
        query=None,
        top_k=top_k,
        results=_to_hits(scored),
    )