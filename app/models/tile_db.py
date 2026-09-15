from pathlib import Path
import aiosqlite

DB_PATH = Path("data/geo_system.db")


async def init_db():
    """Initializes the SQLite audit database and ensures required directories exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS tile_audit (
                id TEXT PRIMARY KEY,
                image_path TEXT NOT NULL,
                primary_tag TEXT DEFAULT 'unassigned',
                cluster INTEGER DEFAULT 0,
                anomaly_score REAL DEFAULT 0.0,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await db.commit()


async def record_tile(
    tile_id: str,
    path: str,
    tag: str = "unassigned",
    cluster: int = 0,
    score: float = 0.0,
    status: str = "pending"
):
    """Inserts or updates a tile record in the SQLite ledger."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            INSERT INTO tile_audit (id, image_path, primary_tag, cluster, anomaly_score, status)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                image_path = excluded.image_path,
                primary_tag = excluded.primary_tag,
                cluster = excluded.cluster,
                anomaly_score = excluded.anomaly_score
            """,
            (tile_id, path, tag, cluster, score, status),
        )
        await db.commit()


async def get_anomalous_tiles(limit: int = 50) -> list[dict]:
    """Fetches tiles for the human-in-the-loop review queue."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT id, image_path, primary_tag, cluster, anomaly_score, status, created_at
            FROM tile_audit
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def update_tile_status(tile_id: str, status: str):
    """Updates the triage decision (e.g., 'verified' or 'false_positive')."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """
            UPDATE tile_audit
            SET status = ?
            WHERE id = ?
            """,
            (status, tile_id),
        )
        await db.commit()