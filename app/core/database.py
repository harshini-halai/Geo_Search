from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator, Iterable, Optional

from app.core.config import settings

CREATE_REVIEW_QUEUE_TABLE = """
CREATE TABLE IF NOT EXISTS review_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tile_id TEXT NOT NULL,
    t1_tile_id TEXT,
    t2_tile_id TEXT,
    status TEXT NOT NULL CHECK(status IN ('PENDING', 'CONFIRMED', 'REJECTED')),
    confidence REAL NOT NULL DEFAULT 0.0,
    drift_score REAL,
    remarks TEXT,
    bbox_json TEXT,
    date_t1 TEXT,
    date_t2 TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_review_queue_status ON review_queue(status);",
    "CREATE INDEX IF NOT EXISTS idx_review_queue_tile_id ON review_queue(tile_id);",
]


def init_db() -> None:
    settings.ensure_dirs()
    with sqlite3.connect(settings.SQLITE_PATH) as conn:
        conn.execute(CREATE_REVIEW_QUEUE_TABLE)
        for stmt in CREATE_INDEXES:
            conn.execute(stmt)
        conn.commit()


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(settings.SQLITE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class ReviewQueueRepository:
    @staticmethod
    def create_item(
        *,
        tile_id: str,
        status: str,
        confidence: float,
        remarks: Optional[str] = None,
        t1_tile_id: Optional[str] = None,
        t2_tile_id: Optional[str] = None,
        drift_score: Optional[float] = None,
        bbox_json: Optional[str] = None,
        date_t1: Optional[str] = None,
        date_t2: Optional[str] = None,
    ) -> int:
        now = utc_now_iso()
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO review_queue (
                    tile_id, t1_tile_id, t2_tile_id, status, confidence, drift_score,
                    remarks, bbox_json, date_t1, date_t2, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tile_id,
                    t1_tile_id,
                    t2_tile_id,
                    status,
                    confidence,
                    drift_score,
                    remarks,
                    bbox_json,
                    date_t1,
                    date_t2,
                    now,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    @staticmethod
    def list_items(status: Optional[str] = None, limit: int = 100, offset: int = 0) -> list[sqlite3.Row]:
        query = "SELECT * FROM review_queue"
        params: list[object] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with get_db_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return list(rows)

    @staticmethod
    def get_item(item_id: int) -> Optional[sqlite3.Row]:
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM review_queue WHERE id = ?", (item_id,)).fetchone()
            return row

    @staticmethod
    def update_item(
        item_id: int,
        *,
        status: Optional[str] = None,
        confidence: Optional[float] = None,
        remarks: Optional[str] = None,
    ) -> bool:
        fields: list[str] = []
        params: list[object] = []
        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if confidence is not None:
            fields.append("confidence = ?")
            params.append(confidence)
        if remarks is not None:
            fields.append("remarks = ?")
            params.append(remarks)
        if not fields:
            return False
        fields.append("updated_at = ?")
        params.append(utc_now_iso())
        params.append(item_id)
        with get_db_connection() as conn:
            cursor = conn.execute(
                f"UPDATE review_queue SET {', '.join(fields)} WHERE id = ?",
                params,
            )
            return cursor.rowcount > 0

    @staticmethod
    def bulk_create(items: Iterable[dict]) -> int:
        now = utc_now_iso()
        rows = []
        for item in items:
            rows.append(
                (
                    item["tile_id"],
                    item.get("t1_tile_id"),
                    item.get("t2_tile_id"),
                    item["status"],
                    item.get("confidence", 0.0),
                    item.get("drift_score"),
                    item.get("remarks"),
                    item.get("bbox_json"),
                    item.get("date_t1"),
                    item.get("date_t2"),
                    now,
                    now,
                )
            )
        if not rows:
            return 0
        with get_db_connection() as conn:
            conn.executemany(
                """
                INSERT INTO review_queue (
                    tile_id, t1_tile_id, t2_tile_id, status, confidence, drift_score,
                    remarks, bbox_json, date_t1, date_t2, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            return len(rows)