from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import analyst, change, ingest, search
from app.core.config import settings
from app.core.database import close_db, init_db
from app.ml.embedder import get_embedder
from app.services.qdrant_store import get_qdrant_store

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Geo-Semantic Backend (offline mode)")
    await init_db()
    get_qdrant_store()
    get_embedder()
    logger.info("PostgreSQL, Qdrant, and OpenCLIP initialized")
    yield
    await close_db()
    logger.info("Shutting down Geo-Semantic Backend")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Offline semantic satellite imagery search and multi-temporal change detection",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix=settings.API_PREFIX)
app.include_router(search.router, prefix=settings.API_PREFIX)
app.include_router(change.router, prefix=settings.API_PREFIX)
app.include_router(analyst.router, prefix=settings.API_PREFIX)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": "offline", "database": "postgresql"}