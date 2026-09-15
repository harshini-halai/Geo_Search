from fastapi import APIRouter, Request, Form, Depends
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.models.tile_db import get_audit_list, update_audit_status
from app.services.qdrant_store import QdrantStore, get_qdrant_store
from app.services.ml_analytics import analyze_embeddings

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory="templates")

@router.get("/")
async def dashboard_home(
    request: Request,
    store: QdrantStore = Depends(get_qdrant_store),
):
    # 1. Fetch live audit records from SQLite
    audits = await get_audit_list(limit=20)
    
    # 2. Extract cluster stats directly from Qdrant
    records, _ = store._client.scroll(
        collection_name=store.collection_name, 
        limit=200, 
        with_vectors=True, 
        with_payload=True
    )
    insights = analyze_embeddings(records, n_clusters=3)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "audits": audits,
        "insights": insights
    })

@router.post("/audit/{tile_id}/update")
async def audit_decision(tile_id: str, status: str = Form(...)):
    await update_audit_status(tile_id, status)
    return RedirectResponse(url="/", status_code=303)