from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "Geo-Semantic Backend"
    APP_VERSION: str = "0.1.0"
    API_PREFIX: str = "/api/v1"

    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data")
    QDRANT_PATH: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "qdrant"
    )

    DATABASE_URL: str = Field(
        default_factory=lambda: f"sqlite+aiosqlite:///{(Path(__file__).resolve().parents[2] / 'data' / 'geo_semantic.db').as_posix()}"
    )
    DB_ECHO: bool = False
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10

    OPENCLIP_MODEL: str = "ViT-B-32"
    OPENCLIP_PRETRAINED: str = "openai"
    EMBEDDING_DIM: int = 512

    TILE_SIZE: int = 256
    TILE_STRIDE: int = 256
    MIN_VALID_PIXEL_RATIO: float = 0.85
    BLACK_PIXEL_THRESHOLD: float = 0.02

    QDRANT_COLLECTION: str = "satellite_tiles"
    SEARCH_DEFAULT_TOP_K: int = 10
    CHANGE_DEFAULT_TOP_K: int = 50
    CHANGE_DRIFT_THRESHOLD: float = 0.25

    CLOUD_BRIGHTNESS_THRESHOLD: float = 0.82
    SHADOW_BRIGHTNESS_THRESHOLD: float = 0.08
    CLOUD_SHADOW_MAX_RATIO: float = 0.35

    CORS_ORIGINS: List[str] = ["*"]
    LOG_LEVEL: str = "INFO"

    def ensure_dirs(self) -> None:
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.QDRANT_PATH.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()