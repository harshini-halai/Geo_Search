from pathlib import Path
from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.models.tile_db import get_audit_list, update_audit_status
from app.services.qdrant_store import QdrantStore, get_qdrant_store
from app.services.ml_analytics import analyze_embeddings

router = APIRouter(include_in_schema=False)

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@router.get("/")
async def dashboard_home(request: Request, store: QdrantStore = Depends(get_qdrant_store)):
    audits = []
    try:
        audits = await get_audit_list(limit=20)
    except Exception as e:
        print(f"[Warning] SQLite audit fetch error: {e}")

    insights = {
        "cluster_distribution": {"Default": 0},
        "suspicious_tiles": [],
        "total_analyzed": 0,
        "anomalies_detected": 0,
    }

    collection = (
        getattr(store, "collection_name", None)
        or getattr(store, "collection", None)
        or getattr(settings, "QDRANT_COLLECTION", "satellite_tiles")
    )

    try:
        client = getattr(store, "_client", None) or getattr(store, "client", None)
        if client:
            records, _ = client.scroll(
                collection_name=collection,
                limit=200,
                with_vectors=True,
                with_payload=True,
            )
            if records and len(records) >= 3:
                insights = analyze_embeddings(records, n_clusters=3)
    except Exception as e:
        print(f"[Warning] Qdrant scroll/ML skipped: {e}")

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "audits": audits,
            "insights": insights,
        },
    )


@router.post("/audit/{tile_id}/update")
async def audit_decision(tile_id: str, status: str = Form(...)):
    try:
        await update_audit_status(tile_id, status)
    except Exception as e:
        print(f"[Error] Failed to update audit status: {e}")
    return RedirectResponse(url="/", status_code=303)