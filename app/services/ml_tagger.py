import torch
from app.ml.embedder import get_embedder

LAND_COVER_LABELS = [
    "dense green forest",
    "agricultural farmland",
    "water body river lake",
    "dense urban residential buildings",
    "industrial factories and warehouses",
    "barren land and desert",
    "road transport network"
]

def predict_land_cover(tile_vector: list, top_k: int = 2) -> list[dict]:
    """
    Zero-shot semantic classification comparing tile embedding 
    against pre-computed land cover category embeddings.
    """
    embedder = get_embedder()
    
    # Text prompts encode karo
    text_features = embedder.embed_texts(LAND_COVER_LABELS)  # shape: (N, D)
    
    tile_feat = torch.tensor(tile_vector).unsqueeze(0)  # shape: (1, D)
    
    # Cosine Similarity
    similarity = torch.cosine_similarity(tile_feat, torch.tensor(text_features))
    
    values, indices = torch.topk(similarity, k=top_k)
    
    tags = []
    for val, idx in zip(values.tolist(), indices.tolist()):
        tags.append({
            "label": LAND_COVER_LABELS[idx],
            "confidence": round(float(val), 4)
        })
    return tags