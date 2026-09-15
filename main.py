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