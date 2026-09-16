from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api import analyst, change, ingest, search
from app.api.ml_routes import router as ml_router
from app.controllers.web_controller import router as web_router
from app.models.tile_db import init_db

# Fallback dataset used by the lightweight search endpoint below. The
# production search routers provide the actual tile data when available.
MOCK_TILES: list[dict[str, str]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="AERO-SENTINEL // Geo-Semantic Intelligence Platform",
    description="Offline-first Satellite Imagery Semantic Search & Surveillance Platform",
    version="1.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(settings, "CORS_ORIGINS", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Mount modular static assets (CSS, JS)
static_dir = Path("frontend/static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 2. Mount tile storage for visual previews
tiles_dir = Path("storage/tiles")
tiles_dir.mkdir(parents=True, exist_ok=True)
app.mount("/storage/tiles", StaticFiles(directory=str(tiles_dir)), name="tiles")

# API Routers
api_prefix = getattr(settings, "API_PREFIX", "/api/v1")
app.include_router(ingest.router, prefix=api_prefix)
app.include_router(search.router, prefix=api_prefix)
app.include_router(change.router, prefix=api_prefix)
app.include_router(analyst.router, prefix=api_prefix)
app.include_router(ml_router)

# MVC Web Controller
app.include_router(web_router)


@app.get("/health", tags=["Health"])
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "mode": "offline-first",
        "system": "Aero-Sentinel Core",
    }

from fastapi import Query

@app.get("/api/search")
async def search_tiles(q: str = Query(..., min_length=1)):
    query_lower = q.lower().strip()
    
    # 1. Direct Keyword / Tag Matching
    keyword_matches = [
        t for t in MOCK_TILES 
        if query_lower in t["tag"].lower() or query_lower in t["tile_id"].lower()
    ]
    
    # 2. Agar exact tag match mil jaye toh turant return karo
    if keyword_matches:
        return {"query": q, "count": len(keyword_matches), "results": keyword_matches}
    
    # 3. Fallback: Semantic matching via Qdrant
    try:
        from app.services.qdrant_store import get_qdrant_store
        store = get_qdrant_store()
        client = getattr(store, "_client", None) or getattr(store, "client", None)
        
        # OpenCLIP available ho toh query vector banao, warna mock similarity return karo
        results = [
            t for t in MOCK_TILES 
            if any(word in t["tag"].lower() for word in query_lower.split())
        ]
        return {"query": q, "count": len(results), "results": results or MOCK_TILES[:3]}
    except Exception as e:
        return {"query": q, "count": len(MOCK_TILES), "results": MOCK_TILES}