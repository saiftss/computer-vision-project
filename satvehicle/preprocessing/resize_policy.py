"""
Resize / tiling policy for small vehicles in high-resolution aerial frames.

Satellite and DOTA-style images are large; cars occupy few pixels. Strategies:

1. **Train at high ``imgsz``** (e.g. 1024–1280) with YOLOv8 so small objects retain detail.
2. **Letterbox** is handled internally by Ultralytics; avoid downscaling below ~640 if mAP collapses.
3. **Sliding-window tiling** (optional): chop each frame into overlapping tiles, re-map boxes, train on tiles.

This module provides helpers for (3) when you need explicit tiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass
class TileSpec:
    x0: int
    y0: int
    width: int
    height: int


def iter_tiles(
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap: int = 0,
) -> Iterator[TileSpec]:
    """Yield axis-aligned tiles that cover the image (row-major)."""
    step = max(1, tile_size - overlap)
    y = 0
    while y < image_height:
        x = 0
        while x < image_width:
            x0, y0 = x, y
            w = min(tile_size, image_width - x0)
            h = min(tile_size, image_height - y0)
            if w > 0 and h > 0:
                yield TileSpec(x0, y0, w, h)
            x += step
        y += step


def yolo_lines_in_tile(
    yolo_lines: list[str],
    tile: TileSpec,
    full_w: int,
    full_h: int,
    min_intersection_ratio: float = 0.35,
) -> list[str]:
    """
    Clip normalized YOLO boxes to a tile and re-normalize to tile coordinates.

    ``yolo_lines`` entries: ``cls cx cy w h`` in full-image normalized space.
    """
    out: list[str] = []
    for line in yolo_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        px_w = bw * full_w
        px_h = bh * full_h
        cx_px = cx * full_w
        cy_px = cy * full_h
        xmin = cx_px - px_w / 2
        ymin = cy_px - px_h / 2
        xmax = cx_px + px_w / 2
        ymax = cy_px + px_h / 2

        tx0, ty0 = tile.x0, tile.y0
        tx1, ty1 = tile.x0 + tile.width, tile.y0 + tile.height
        inter_xmin = max(xmin, tx0)
        inter_ymin = max(ymin, ty0)
        inter_xmax = min(xmax, tx1)
        inter_ymax = min(ymax, ty1)
        iw = max(0.0, inter_xmax - inter_xmin)
        ih = max(0.0, inter_ymax - inter_ymin)
        inter_area = iw * ih
        box_area = max(1.0, (xmax - xmin) * (ymax - ymin))
        if inter_area / box_area < min_intersection_ratio:
            continue

        nxmin = max(xmin, tx0) - tx0
        nymin = max(ymin, ty0) - ty0
        nxmax = min(xmax, tx1) - tx0
        nymax = min(ymax, ty1) - ty0
        tw, th = tile.width, tile.height
        ncx = (nxmin + nxmax) / 2 / tw
        ncy = (nymin + nymax) / 2 / th
        nw = (nxmax - nxmin) / tw
        nh = (nymax - nymin) / th
        if nw < 0.01 or nh < 0.01:
            continue
        out.append(f"{cls_id} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
    return out
