from __future__ import annotations

import json
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import ReviewQueueItem, ReviewQueueRepository, get_session

router = APIRouter(prefix="/analyst", tags=["analyst"])

ReviewStatus = Literal["PENDING", "CONFIRMED", "REJECTED"]


class ReviewItem(BaseModel):
    id: int
    tile_id: str
    t1_tile_id: Optional[str] = None
    t2_tile_id: Optional[str] = None
    status: ReviewStatus
    confidence: float
    drift_score: Optional[float] = None
    remarks: Optional[str] = None
    bbox: Optional[dict] = None
    date_t1: Optional[str] = None
    date_t2: Optional[str] = None
    created_at: str
    updated_at: str


class ReviewListResponse(BaseModel):
    total_returned: int
    items: list[ReviewItem]


class ReviewCreateRequest(BaseModel):
    tile_id: str = Field(..., min_length=1)
    status: ReviewStatus = "PENDING"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    remarks: Optional[str] = None
    t1_tile_id: Optional[str] = None
    t2_tile_id: Optional[str] = None
    drift_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    bbox: Optional[dict] = None
    date_t1: Optional[str] = None
    date_t2: Optional[str] = None


class ReviewUpdateRequest(BaseModel):
    status: Optional[ReviewStatus] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    remarks: Optional[str] = None


def _orm_to_schema(item: ReviewQueueItem) -> ReviewItem:
    bbox = None
    if item.bbox_json:
        try:
            bbox = json.loads(item.bbox_json)
        except json.JSONDecodeError:
            bbox = None

    return ReviewItem(
        id=item.id,
        tile_id=item.tile_id,
        t1_tile_id=item.t1_tile_id,
        t2_tile_id=item.t2_tile_id,
        status=item.status,  # type: ignore[arg-type]
        confidence=item.confidence,
        drift_score=item.drift_score,
        remarks=item.remarks,
        bbox=bbox,
        date_t1=item.date_t1,
        date_t2=item.date_t2,
        created_at=item.created_at.isoformat(),
        updated_at=item.updated_at.isoformat(),
    )


@router.get("/queue", response_model=ReviewListResponse)
async def list_review_queue(
    status: Optional[ReviewStatus] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> ReviewListResponse:
    rows = await ReviewQueueRepository.list_items(
        session,
        status=status,
        limit=limit,
        offset=offset,
    )
    items = [_orm_to_schema(row) for row in rows]
    return ReviewListResponse(total_returned=len(items), items=items)


@router.get("/queue/{item_id}", response_model=ReviewItem)
async def get_review_item(
    item_id: int,
    session: AsyncSession = Depends(get_session),
) -> ReviewItem:
    row = await ReviewQueueRepository.get_item(session, item_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Review item {item_id} not found")
    return _orm_to_schema(row)


@router.post("/queue", response_model=ReviewItem)
async def create_review_item(
    payload: ReviewCreateRequest,
    session: AsyncSession = Depends(get_session),
) -> ReviewItem:
    item = await ReviewQueueRepository.create_item(
        session,
        tile_id=payload.tile_id,
        status=payload.status,
        confidence=payload.confidence,
        remarks=payload.remarks,
        t1_tile_id=payload.t1_tile_id,
        t2_tile_id=payload.t2_tile_id,
        drift_score=payload.drift_score,
        bbox_json=json.dumps(payload.bbox) if payload.bbox else None,
        date_t1=payload.date_t1,
        date_t2=payload.date_t2,
    )
    return _orm_to_schema(item)


@router.patch("/queue/{item_id}", response_model=ReviewItem)
async def update_review_item(
    item_id: int,
    payload: ReviewUpdateRequest,
    session: AsyncSession = Depends(get_session),
) -> ReviewItem:
    if payload.status is None and payload.confidence is None and payload.remarks is None:
        raise HTTPException(status_code=400, detail="At least one field must be provided")

    item = await ReviewQueueRepository.update_item(
        session,
        item_id,
        status=payload.status,
        confidence=payload.confidence,
        remarks=payload.remarks,
    )
    if item is None:
        raise HTTPException(status_code=404, detail=f"Review item {item_id} not found")
    return _orm_to_schema(item)