from fastapi import APIRouter, Depends, Query
from app.services.qdrant_store import QdrantStore
from app.services.ml_analytics import analyze_embeddings

router = APIRouter(prefix="/api/v1/ml", tags=["ML Analytics"])

def get_qdrant_store():
    from app.services.qdrant_store import get_store
    return get_store()

@router.get("/insights")
async def get_geo_insights(
    n_clusters: int = Query(default=3, ge=2, le=10),
    store: QdrantStore = Depends(get_qdrant_store)
):
    """
    Performs Unsupervised K-Means Clustering and Isolation Forest
    Anomaly Detection directly on Qdrant vectors.
    """
    # Qdrant client se saare vectors scroll/fetch karo
    records, _ = store._client.scroll(
        collection_name=store.collection_name,
        limit=500,
        with_vectors=True,
        with_payload=True
    )

    insights = analyze_embeddings(records, n_clusters=n_clusters)
    return insights