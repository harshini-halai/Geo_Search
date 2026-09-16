import asyncio
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from app.models.tile_db import init_db, record_tile
from app.services.qdrant_store import get_qdrant_store
from app.core.config import settings
from qdrant_client.http import models

TILES_DIR = Path("storage/tiles")
TILES_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_TILES = [
    {
        "id": "tile_alpha_01",
        "tag": "Dense Forest Canopy",
        "cluster": 0,
        "score": 0.88,
        "status": "pending",
        "color": (34, 139, 34),
        "text": "FOREST"
    },
    {
        "id": "tile_bravo_02",
        "tag": "River Delta & Estuary",
        "cluster": 1,
        "score": 0.42,
        "status": "verified",
        "color": (30, 144, 255),
        "text": "WATER"
    },
    {
        "id": "tile_charlie_03",
        "tag": "Industrial / Logistics Zone",
        "cluster": 2,
        "score": 0.94,
        "status": "pending",
        "color": (169, 169, 169),
        "text": "URBAN"
    },
    {
        "id": "tile_delta_04",
        "tag": "Rapid Deforestation Scar",
        "cluster": 0,
        "score": 0.96,
        "status": "pending",
        "color": (139, 69, 19),
        "text": "SCAR"
    },
    {
        "id": "tile_echo_05",
        "tag": "High-Density Residential",
        "cluster": 2,
        "score": 0.31,
        "status": "false_positive",
        "color": (218, 165, 32),
        "text": "HOUSING"
    },
    {
        "id": "tile_foxtrot_06",
        "tag": "Offshore Harbor Vessel Lane",
        "cluster": 1,
        "score": 0.79,
        "status": "pending",
        "color": (70, 130, 180),
        "text": "PORT"
    }
]

def generate_tile_image(path: Path, bg_color: tuple, label: str):
    img = Image.new("RGB", (256, 256), color=bg_color)
    draw = ImageDraw.Draw(img)
    for x in range(0, 256, 32):
        draw.line([(x, 0), (x, 256)], fill=(255, 255, 255, 40), width=1)
    for y in range(0, 256, 32):
        draw.line([(0, y), (256, y)], fill=(255, 255, 255, 40), width=1)
    draw.rectangle([10, 10, 120, 35], fill=(0, 0, 0))
    draw.text((15, 15), label, fill=(0, 255, 150))
    img.save(path)

async def run_seed():
    print("[*] Initializing SQLite Audit DB...")
    await init_db()

    print("[*] Connecting to Qdrant...")
    store = get_qdrant_store()
    collection = getattr(settings, "QDRANT_COLLECTION", "satellite_tiles")

    client = getattr(store, "_client", None) or getattr(store, "client", None)
    if client:
        collections = [c.name for c in client.get_collections().collections]
        if collection not in collections:
            client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE)
            )

    print("[*] Populating SQLite & Qdrant with synthetic tiles...")
    points = []
    
    for idx, item in enumerate(SAMPLE_TILES):
        img_filename = f"{item['id']}.png"
        img_disk_path = TILES_DIR / img_filename
        
        generate_tile_image(img_disk_path, item["color"], item["text"])
        web_tile_path = f"/storage/tiles/{img_filename}"

        await record_tile(
            tile_id=item["id"],
            path=web_tile_path,
            tag=item["tag"],
            cluster=item["cluster"],
            score=item["score"],
            status=item["status"]
        )

        rng = np.random.default_rng(seed=idx + 42)
        vec = rng.standard_normal(512).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        points.append(
            models.PointStruct(
                id=idx + 1,
                vector=vec.tolist(),
                payload={
                    "tile_id": item["id"],
                    "image_path": web_tile_path,
                    "tag": item["tag"],
                    "cluster": item["cluster"],
                    "score": item["score"],
                }
            )
        )

    if client and points:
        client.upsert(collection_name=collection, points=points)
        print(f"[+] Upserted {len(points)} vectors into Qdrant '{collection}'.")

    print("[✔] Sample Data Added Successfully!")

if __name__ == "__main__":
    asyncio.run(run_seed())

    