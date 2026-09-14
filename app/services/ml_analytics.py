import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

def analyze_embeddings(points: list, n_clusters: int = 3) -> dict:
    """
    Takes points from Qdrant, extracts vectors, runs K-Means clustering
    and Isolation Forest anomaly detection.
    """
    if not points or len(points) < n_clusters:
        return {
            "total_tiles": len(points),
            "clusters": [],
            "anomalies": [],
            "message": "Not enough tiles to perform ML analysis (minimum required: 3)"
        }

    # Extract vectors and metadata
    vectors = np.array([pt.vector for pt in points], dtype=np.float32)
    payloads = [pt.payload for pt in points]
    point_ids = [pt.id for pt in points]

    # 1. K-Means Clustering (Land Cover Grouping)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
    cluster_labels = kmeans.fit_predict(vectors)

    # 2. Isolation Forest (Anomaly / Suspicious Patch Detection)
    # Contamination defines the percentage of expected outliers
    iso_forest = IsolationForest(contamination=0.15, random_state=42)
    anomaly_preds = iso_forest.fit_predict(vectors)  # -1 = anomaly, 1 = normal
    anomaly_scores = iso_forest.score_samples(vectors)

    # Prepare Structured Output
    cluster_groups = {f"cluster_{i}": [] for i in range(n_clusters)}
    anomalies = []

    for idx, (p_id, payload, c_lbl, is_anom, score) in enumerate(
        zip(point_ids, payloads, cluster_labels, anomaly_preds, anomaly_scores)
    ):
        tile_info = {
            "id": p_id,
            "image_path": payload.get("image_path", ""),
            "date": payload.get("date", ""),
            "bbox": payload.get("bbox", {}),
            "cluster": int(c_lbl),
            "anomaly_score": round(float(-score), 4) # Higher means more anomalous
        }

        cluster_groups[f"cluster_{c_lbl}"].append(tile_info)

        if is_anom == -1:
            anomalies.append(tile_info)

    # Sort anomalies from most anomalous to least
    anomalies.sort(key=lambda x: x["anomaly_score"], reverse=True)

    return {
        "total_analyzed": len(points),
        "cluster_distribution": {k: len(v) for k, v in cluster_groups.items()},
        "clusters": cluster_groups,
        "anomalies_detected": len(anomalies),
        "suspicious_tiles": anomalies
    }