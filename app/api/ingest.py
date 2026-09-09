from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator
from qdrant_client.http import models as qmodels

from app.core.config import settings
from app.ml.embedder import get_embedder
from app.services.qdrant_store import get_qdrant_store, make_point_id, payload_from_tile_metadata
from app.services.tiler import iter_tiles_from_geotiff

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestRequest(BaseModel):
    image_path: str = Field(..., description="Absolute or relative path to GeoTIFF")
    date: str = Field(..., description="Acquisition date, e.g. 2024-06-01")
    sensor: str = Field(..., description="Sensor name, e.g. Sentinel-2")
    tile_size: Optional[int] = Field(default=None, ge=64, le=1024)
    stride: Optional[int] = Field(default=None, ge=32, le=1024)

    @field_validator("image_path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        path = Path(v)
        if not path.exists():
            raise ValueError(f"GeoTIFF not found: {v}")
        if path.suffix.lower() not in {".tif", ".tiff"}:
            raise ValueError("image_path must be a .tif/.tiff file")
        return str(path.resolve())


class IngestResponse(BaseModel):
    image_path: str
    date: str
    sensor: str
    tiles_processed: int
    tiles_indexed: int
    tiles_skipped: int


@router.post("/geotiff", response_model=IngestResponse)
async def ingest_geotiff(payload: IngestRequest) -> IngestResponse:
    store = get_qdrant_store()
    embedder = get_embedder()

    tiles_processed = 0
    tiles_indexed = 0
    tiles_skipped = 0
    batch: list[qmodels.PointStruct] = []

    try:
        for tile in iter_tiles_from_geotiff(
            payload.image_path,
            date=payload.date,
            sensor=payload.sensor,
            tile_size=payload.tile_size,
            stride=payload.stride,
        ):
            tiles_processed += 1
            vector = await embedder.embed_image_array_async(tile.array)
            metadata = {
                "tile_id": tile.tile_id,
                "bbox": tile.bbox,
                "date": tile.date,
                "sensor": tile.sensor,
                "image_path": tile.image_path,
                "crs": tile.crs,
                "row": tile.row,
                "col": tile.col,
            }
            batch.append(
                qmodels.PointStruct(
                    id=make_point_id(),
                    vector=vector,
                    payload=payload_from_tile_metadata(metadata),
                )
            )
            if len(batch) >= 32:
                await store.run_sync(store.upsert_points, batch)
                tiles_indexed += len(batch)
                batch.clear()

        if batch:
            await store.run_sync(store.upsert_points, batch)
            tiles_indexed += len(batch)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    tiles_skipped = max(0, tiles_processed - tiles_indexed)

    return IngestResponse(
        image_path=payload.image_path,
        date=payload.date,
        sensor=payload.sensor,
        tiles_processed=tiles_processed,
        tiles_indexed=tiles_indexed,
        tiles_skipped=tiles_skipped,
    )