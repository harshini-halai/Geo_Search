from __future__ import annotations

import json
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from app.core.database import ReviewQueueRepository

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

    @field_validator("status", "confidence", "remarks")
    @classmethod
    def at_least_one_field(cls, v, info):
        return v


def _row_to_item(row) -> ReviewItem:
    bbox = None
    if row["bbox_json"]:
        try:
            bbox = json.loads(row["bbox_json"])
        except json.JSONDecodeError:
            bbox = None
    return ReviewItem(
        id=row["id"],
        tile_id=row["tile_id"],
        t1_tile_id=row["t1_tile_id"],
        t2_tile_id=row["t2_tile_id"],
        status=row["status"],
        confidence=row["confidence"],
        drift_score=row["drift_score"],
        remarks=row["remarks"],
        bbox=bbox,
        date_t1=row["date_t1"],
        date_t2=row["date_t2"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


@router.get("/queue", response_model=ReviewListResponse)
async def list_review_queue(
    status: Optional[ReviewStatus] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> ReviewListResponse:
    rows = ReviewQueueRepository.list_items(status=status, limit=limit, offset=offset)
    items = [_row_to_item(r) for r in rows]
    return ReviewListResponse(total_returned=len(items), items=items)


@router.get("/queue/{item_id}", response_model=ReviewItem)
async def get_review_item(item_id: int) -> ReviewItem:
    row = ReviewQueueRepository.get_item(item_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Review item {item_id} not found")
    return _row_to_item(row)


@router.post("/queue", response_model=ReviewItem)
async def create_review_item(payload: ReviewCreateRequest) -> ReviewItem:
    item_id = ReviewQueueRepository.create_item(
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
    row = ReviewQueueRepository.get_item(item_id)
    if row is None:
        raise HTTPException(status_code=500, detail="Failed to create review item")
    return _row_to_item(row)


@router.patch("/queue/{item_id}", response_model=ReviewItem)
async def update_review_item(item_id: int, payload: ReviewUpdateRequest) -> ReviewItem:
    if payload.status is None and payload.confidence is None and payload.remarks is None:
        raise HTTPException(status_code=400, detail="At least one field must be provided")

    updated = ReviewQueueRepository.update_item(
        item_id,
        status=payload.status,
        confidence=payload.confidence,
        remarks=payload.remarks,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Review item {item_id} not found")

    row = ReviewQueueRepository.get_item(item_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Review item {item_id} not found")
    return _row_to_item(row)