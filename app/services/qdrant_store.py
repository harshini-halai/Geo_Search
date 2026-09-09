from __future__ import annotations

import asyncio
import threading
import uuid
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import settings

_store: Optional["QdrantStore"] = None
_store_lock = threading.Lock()


class QdrantStore:
    def __init__(self) -> None:
        settings.ensure_dirs()
        self._client = QdrantClient(path=str(settings.QDRANT_PATH))
        self._io_lock = threading.RLock()
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        with self._io_lock:
            collections = self._client.get_collections().collections
            names = {c.name for c in collections}
            if settings.QDRANT_COLLECTION not in names:
                self._client.create_collection(
                    collection_name=settings.QDRANT_COLLECTION,
                    vectors_config=qmodels.VectorParams(
                        size=settings.EMBEDDING_DIM,
                        distance=qmodels.Distance.COSINE,
                    ),
                )

    def upsert_points(self, points: list[qmodels.PointStruct]) -> None:
        if not points:
            return
        with self._io_lock:
            self._client.upsert(
                collection_name=settings.QDRANT_COLLECTION,
                points=points,
                wait=True,
            )

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        query_filter: Optional[qmodels.Filter] = None,
    ) -> list[qmodels.ScoredPoint]:
        with self._io_lock:
            return self._client.search(
                collection_name=settings.QDRANT_COLLECTION,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )

    def scroll_by_payload(
        self,
        *,
        must: list[qmodels.FieldCondition],
        limit: int = 1000,
    ) -> list[qmodels.Record]:
        query_filter = qmodels.Filter(must=must)
        with self._io_lock:
            records, _ = self._client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=True,
            )
            return list(records)

    def get_by_tile_id(self, tile_id: str) -> Optional[qmodels.Record]:
        records = self.scroll_by_payload(
            must=[qmodels.FieldCondition(key="tile_id", match=qmodels.MatchValue(value=tile_id))],
            limit=1,
        )
        return records[0] if records else None

    async def run_sync(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


def get_qdrant_store() -> QdrantStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = QdrantStore()
    return _store


def make_point_id() -> str:
    return str(uuid.uuid4())


def payload_from_tile_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "tile_id": metadata["tile_id"],
        "bbox": metadata["bbox"],
        "date": metadata["date"],
        "sensor": metadata["sensor"],
        "image_path": metadata["image_path"],
        "crs": metadata.get("crs"),
        "row": metadata.get("row"),
        "col": metadata.get("col"),
    }