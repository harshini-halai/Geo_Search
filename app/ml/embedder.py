from __future__ import annotations

import asyncio
import threading
from io import BytesIO
from pathlib import Path
from typing import Optional, Union

import numpy as np
import open_clip
import torch
from PIL import Image

from app.core.config import settings
from app.services.cloud_mask import _normalize_to_unit, _to_hwc

_embedder: Optional["OpenCLIPEmbedder"] = None
_embedder_lock = threading.Lock()


class OpenCLIPEmbedder:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.OPENCLIP_MODEL,
            pretrained=settings.OPENCLIP_PRETRAINED,
        )
        self.tokenizer = open_clip.get_tokenizer(settings.OPENCLIP_MODEL)
        self.model.to(self.device)
        self.model.eval()
        self._infer_lock = threading.RLock()

    def _array_to_pil(self, array: np.ndarray) -> Image.Image:
        img = _normalize_to_unit(_to_hwc(array))
        rgb = (img * 255.0).astype(np.uint8)
        return Image.fromarray(rgb)

    def _l2_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor / tensor.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    def embed_image_array(self, array: np.ndarray) -> list[float]:
        pil = self._array_to_pil(array)
        return self.embed_image_pil(pil)

    def embed_image_pil(self, image: Image.Image) -> list[float]:
        with self._infer_lock:
            with torch.no_grad():
                tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                features = self.model.encode_image(tensor)
                features = self._l2_normalize(features)
                return features.squeeze(0).cpu().tolist()

    def embed_image_path(self, path: Union[str, Path]) -> list[float]:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
        return self.embed_image_pil(rgb)

    def embed_image_bytes(self, data: bytes) -> list[float]:
        with Image.open(BytesIO(data)) as img:
            rgb = img.convert("RGB")
        return self.embed_image_pil(rgb)

    def embed_text(self, text: str) -> list[float]:
        with self._infer_lock:
            with torch.no_grad():
                tokens = self.tokenizer([text]).to(self.device)
                features = self.model.encode_text(tokens)
                features = self._l2_normalize(features)
                return features.squeeze(0).cpu().tolist()

    async def embed_text_async(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_text, text)

    async def embed_image_array_async(self, array: np.ndarray) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_image_array, array)

    async def embed_image_bytes_async(self, data: bytes) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_image_bytes, data)


def get_embedder() -> OpenCLIPEmbedder:
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = OpenCLIPEmbedder()
    return _embedder