from pathlib import Path
import aiosqlite

DB_PATH = Path("data/geo_system.db")

async def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tile_audit (
                id TEXT PRIMARY KEY,
                image_path TEXT,
                primary_tag TEXT,
                cluster INTEGER,
                anomaly_score REAL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()

async def record_tile(tile_id: str, path: str, tag: str, cluster: int, score: float):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO tile_audit (id, image_path, primary_tag, cluster, anomaly_score)
            VALUES (?, ?, ?, ?, ?)
        """, (tile_id, path, tag, cluster, score))
        await db.commit()

async def get_audit_list(limit: int = 50):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM tile_audit ORDER BY anomaly_score DESC LIMIT ?", 
            (limit,)
        )
        return await cursor.fetchall()

async def update_audit_status(tile_id: str, status: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE tile_audit SET status = ? WHERE id = ?", 
            (status, tile_id)
        )
        await db.commit()