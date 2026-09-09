from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box, mapping

from app.core.config import settings


@dataclass(frozen=True)
class TileRecord:
    tile_id: str
    array: np.ndarray
    bbox: dict
    row: int
    col: int
    crs: str
    date: str
    sensor: str
    image_path: str


def _is_mostly_black(tile: np.ndarray, threshold: float) -> bool:
    if tile.size == 0:
        return True
    if tile.ndim == 2:
        data = tile.astype(np.float32)
    else:
        data = tile[:3].astype(np.float32) if tile.shape[0] >= 3 else tile.astype(np.float32)
    if data.max() > 1.0:
        data = data / 255.0
    dark_ratio = float(np.mean(data <= threshold))
    return dark_ratio > (1.0 - settings.MIN_VALID_PIXEL_RATIO)


def _is_mostly_nodata(tile: np.ndarray, nodata: Optional[float]) -> bool:
    if nodata is None:
        return False
    if tile.ndim == 2:
        valid = tile != nodata
    else:
        valid = np.any(tile != nodata, axis=0)
    valid_ratio = float(np.mean(valid))
    return valid_ratio < settings.MIN_VALID_PIXEL_RATIO


def _window_bbox(window: Window, transform, crs: str) -> dict:
    left, bottom, right, top = rasterio.windows.bounds(window, transform)
    geom = box(left, bottom, right, top)
    return {
        "type": "Polygon",
        "coordinates": [list(mapping(geom)["coordinates"][0])],
        "crs": crs,
        "bounds": [left, bottom, right, top],
    }


def iter_tiles_from_geotiff(
    image_path: str | Path,
    *,
    date: str,
    sensor: str,
    tile_size: int | None = None,
    stride: int | None = None,
) -> Generator[TileRecord, None, None]:
    image_path = Path(image_path)
    tile_size = tile_size or settings.TILE_SIZE
    stride = stride or settings.TILE_STRIDE

    with rasterio.open(image_path) as src:
        crs = src.crs.to_string() if src.crs else "EPSG:4326"
        nodata = src.nodata
        height, width = src.height, src.width

        for row_off in range(0, height, stride):
            for col_off in range(0, width, stride):
                win_height = min(tile_size, height - row_off)
                win_width = min(tile_size, width - col_off)
                if win_height < tile_size or win_width < tile_size:
                    continue

                window = Window(col_off, row_off, win_width, win_height)
                tile = src.read(window=window)

                if _is_mostly_nodata(tile, nodata):
                    continue
                if _is_mostly_black(tile, settings.BLACK_PIXEL_THRESHOLD):
                    continue

                bbox = _window_bbox(window, src.transform, crs)
                tile_id = f"{image_path.stem}_r{row_off}_c{col_off}_{date}"

                yield TileRecord(
                    tile_id=tile_id,
                    array=tile,
                    bbox=bbox,
                    row=row_off,
                    col=col_off,
                    crs=crs,
                    date=date,
                    sensor=sensor,
                    image_path=str(image_path.resolve()),
                )