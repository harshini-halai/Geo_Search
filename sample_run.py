import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "frontend" / "static"
TEMPLATES_DIR = BASE_DIR / "frontend" / "templates"
STORAGE_DIR = BASE_DIR / "storage"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if STORAGE_DIR.exists():
    app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

MOCK_TILES = [
    {"tile_id": "tile_alpha_01", "image_path": "/storage/tiles/tile_alpha_01.png", "tag": "Dense Forest Canopy", "cluster": 0, "score": 0.88, "status": "pending"},
    {"tile_id": "tile_bravo_02", "image_path": "/storage/tiles/tile_bravo_02.png", "tag": "River Delta & Estuary", "cluster": 1, "score": 0.42, "status": "verified"},
    {"tile_id": "tile_charlie_03", "image_path": "/storage/tiles/tile_charlie_03.png", "tag": "Industrial Logistics Zone", "cluster": 2, "score": 0.94, "status": "pending"},
    {"tile_id": "tile_delta_04", "image_path": "/storage/tiles/tile_delta_04.png", "tag": "Rapid Deforestation Scar", "cluster": 0, "score": 0.96, "status": "pending"},
    {"tile_id": "tile_echo_05", "image_path": "/storage/tiles/tile_echo_05.png", "tag": "High-Density Residential", "cluster": 2, "score": 0.31, "status": "false_positive"},
    {"tile_id": "tile_foxtrot_06", "image_path": "/storage/tiles/tile_foxtrot_06.png", "tag": "Offshore Harbor Vessel Lane", "cluster": 1, "score": 0.79, "status": "pending"}
]

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "tiles": MOCK_TILES,
            "total_vectors": 6,
            "pending_count": 4,
            "verified_count": 1,
            "rejected_count": 1
        }
    )

@app.post("/api/triage/{tile_id}")
async def update_status(tile_id: str, payload: dict):
    new_status = payload.get("status", "pending")
    for t in MOCK_TILES:
        if t["tile_id"] == tile_id:
            t["status"] = new_status
            break
    return {"status": "success", "tile_id": tile_id, "new_status": new_status}

if __name__ == "__main__":
    uvicorn.run("simple_run:app", host="127.0.0.1", port=8000, reload=False)