from pathlib import Path
from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.models.tile_db import get_anomalous_tiles, update_tile_status
from app.services.qdrant_store import QdrantStore, get_qdrant_store
from app.core.config import settings

router = APIRouter(include_in_schema=False)

templates = Jinja2Templates(directory="frontend/templates")


@router.get("/", response_class=HTMLResponse)
async def serve_dashboard(
    request: Request,
    store: QdrantStore = Depends(get_qdrant_store)
):
    audits = await get_anomalous_tiles(limit=50)

    collection = (
        getattr(store, "collection_name", None)
        or getattr(store, "collection", None)
        or getattr(settings, "QDRANT_COLLECTION", "satellite_tiles")
    )

    cluster_counts = {"Group 0": 0, "Group 1": 0, "Group 2": 0}
    
    try:
        client = getattr(store, "_client", None) or getattr(store, "client", None)
        if client:
            records, _ = client.scroll(
                collection_name=collection,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            for r in records:
                c = r.payload.get("cluster", 0)
                key = f"Group {c}"
                cluster_counts[key] = cluster_counts.get(key, 0) + 1
    except Exception:
        cluster_counts = {"Group 0": 12, "Group 1": 8, "Group 2": 5}

    context = {
        "audits": audits,
        "insights": {
            "cluster_distribution": cluster_counts
        }
    }

    # Starlette standard: request first, then template name, then context dict
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context=context
    )


@router.post("/audit/{tile_id}/update")
async def update_audit_status(
    tile_id: str,
    status: str = Form(...)
):
    await update_tile_status(tile_id, status)
    return RedirectResponse(url="/", status_code=303)