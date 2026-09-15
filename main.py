from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.core.config import settings
from app.api import analyst, change, ingest, search
from app.api.ml_routes import router as ml_router
from app.controllers.web_controller import router as web_router
from app.models.tile_db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize SQLite database tables
    await init_db()
    yield
    # Shutdown logic (if any) can be placed here


app = FastAPI(
    title="Offline-First Geo-Semantic Intelligence System",
    description="Local satellite imagery semantic search, change detection, and ML analytics",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(settings, "CORS_ORIGINS", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static files directory if it exists (for CSS/JS/images)
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# 1. API Routers (Prefix based for JSON data/services)
api_prefix = getattr(settings, "API_PREFIX", "/api/v1")

app.include_router(ingest.router, prefix=api_prefix)
app.include_router(search.router, prefix=api_prefix)
app.include_router(change.router, prefix=api_prefix)
app.include_router(analyst.router, prefix=api_prefix)
app.include_router(ml_router)

# 2. Web MVC Router (Serves the HTML Dashboard View on root "/")
app.include_router(web_router)


@app.get("/health", tags=["Health"])
async def health() -> dict[str, str]:
    return {
        "status": "ok",
        "mode": "offline-first",
        "system": "Geo-Semantic Engine",
    }