import uuid
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

# Exact paths based on your project tree
from app.ml.embedder import OpenCLIPEmbedder, get_embedder
import app.services.tiler as tiler_module
from app.services.qdrant_store import QdrantStore, get_qdrant_store
from app.models.tile_db import record_tile
from app.core.config import settings

router = APIRouter(prefix="/ingest", tags=["Ingest"])


class IngestGeoTIFFRequest(BaseModel):
    file_path: str
    tile_size: int = 256
    overlap: int = 0
    batch_size: int = 16


def run_tile_process(geotiff_path: Path, tile_size: int, overlap: int):
    """Dynamic check to call whichever method is defined inside tiler.py"""
    if hasattr(tiler_module, "tile_geotiff"):
        return tiler_module.tile_geotiff(geotiff_path=geotiff_path, tile_size=tile_size, overlap=overlap)
    elif hasattr(tiler_module, "slice_geotiff"):
        return tiler_module.slice_geotiff(geotiff_path=geotiff_path, tile_size=tile_size, overlap=overlap)
    elif hasattr(tiler_module, "GeoTIFFTiler"):
        tiler = tiler_module.GeoTIFFTiler(tile_size=tile_size, overlap=overlap)
        return tiler.process(geotiff_path) if hasattr(tiler, "process") else tiler.tile(geotiff_path)
    elif hasattr(tiler_module, "Tiler"):
        tiler = tiler_module.Tiler(tile_size=tile_size, overlap=overlap)
        return tiler.slice(geotiff_path)
    raise AttributeError(f"No compatible slicing method in tiler.py. Available: {dir(tiler_module)}")


@router.post("/geotiff")
async def ingest_geotiff(
    request: IngestGeoTIFFRequest,
    embedder: OpenCLIPEmbedder = Depends(get_embedder),
    store: QdrantStore = Depends(get_qdrant_store),
):
    geotiff_path = Path(request.file_path)
    if not geotiff_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {request.file_path}"
        )

    # 1. Slice GeoTIFF
    tiles = run_tile_process(
        geotiff_path=geotiff_path,
        tile_size=request.tile_size,
        overlap=request.overlap,
    )
    if not tiles:
        return {"status": "success", "tiles_ingested": 0, "message": "No tiles generated"}

    # 2. Extract Embeddings
    image_paths = [t.get("tile_path") if isinstance(t, dict) else str(t) for t in tiles]
    embeddings = embedder.embed_images_batch(image_paths, batch_size=request.batch_size)

    collection = (
        getattr(store, "collection_name", None)
        or getattr(store, "collection", None)
        or getattr(settings, "QDRANT_COLLECTION", "satellite_tiles")
    )

    points = []
    for i, t in enumerate(tiles):
        tile_id = str(uuid.uuid4())
        tile_path = t.get("tile_path", str(t)) if isinstance(t, dict) else str(t)
        bounds = t.get("bounds") if isinstance(t, dict) else None

        payload = {
            "image_path": tile_path,
            "bounds": bounds,
            "source_geotiff": str(geotiff_path),
            "label": "unassigned",
        }
        points.append({
            "id": tile_id,
            "vector": embeddings[i],
            "payload": payload,
        })

    # 3. Upsert to Qdrant
    if hasattr(store, "upsert_points"):
        store.upsert_points(points, collection_name=collection)
    else:
        from qdrant_client.models import PointStruct
        client = getattr(store, "_client", None) or getattr(store, "client", None)
        point_structs = [
            PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
            for p in points
        ]
        client.upsert(collection_name=collection, points=point_structs)

    # 4. Auto-register in SQLite Audit Ledger
    for p in points:
        rel_path = str(p["payload"]["image_path"]).replace("\\", "/")
        await record_tile(
            tile_id=p["id"],
            path=rel_path,
            tag=p["payload"].get("label", "unassigned"),
            cluster=0,
            score=0.0,
        )

    return {
        "status": "success",
        "tiles_ingested": len(points),
        "source": str(geotiff_path),
    }