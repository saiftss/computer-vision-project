"""
Convert DOTA aerial annotations to YOLO horizontal bounding-box labels.

DOTA uses oriented quadrilaterals and class names like ``small-vehicle``.
For a *car-focused* project, we keep only ``small-vehicle`` and map it to YOLO class 0.

YOLO line format: ``cls cx cy w h`` (normalized to [0, 1]).

Usage (example):
    python -m satvehicle.preprocessing.dota_to_yolo \\
        --dota-root /data/DOTA \\
        --out-root /data/dota_vehicle_yolo \\
        --splits train,val
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

# DOTA class name used as a proxy for passenger cars in aerial imagery.
CAR_CLASS_DOTA = "small-vehicle"
YOLO_CLASS_ID = 0


def _quad_to_aabb(xs: list[float], ys: list[float]) -> tuple[float, float, float, float]:
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return xmin, ymin, xmax, ymax


def _aabb_to_yolo_line(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: int,
    img_h: int,
    min_rel_side: float = 0.002,
) -> str | None:
    w_px = xmax - xmin
    h_px = ymax - ymin
    if w_px <= 1 or h_px <= 1:
        return None
    if w_px / img_w < min_rel_side or h_px / img_h < min_rel_side:
        return None
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cx_n = cx / img_w
    cy_n = cy / img_h
    w_n = w_px / img_w
    h_n = h_px / img_h
    cx_n = float(np.clip(cx_n, 0.0, 1.0))
    cy_n = float(np.clip(cy_n, 0.0, 1.0))
    w_n = float(np.clip(w_n, 1e-6, 1.0))
    h_n = float(np.clip(h_n, 1e-6, 1.0))
    return f"{YOLO_CLASS_ID} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}"


def parse_dota_label_line(line: str) -> tuple[list[float], list[float], str, str] | None:
    parts = line.strip().split()
    if len(parts) < 9:
        return None
    nums = list(map(float, parts[:8]))
    category = parts[8]
    diff = parts[9] if len(parts) > 9 else "0"
    xs = [nums[0], nums[2], nums[4], nums[6]]
    ys = [nums[1], nums[3], nums[5], nums[7]]
    return xs, ys, category, diff


def image_long_side(img_path: Path) -> int:
    im = cv2.imread(str(img_path))
    if im is None:
        return 0
    h, w = im.shape[:2]
    return max(h, w)


def convert_dota_split_to_yolo(
    dota_split_images: Path,
    dota_label_txt: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    skip_difficult: bool = True,
) -> tuple[int, int]:
    """
    Convert one DOTA split (e.g. train) into YOLO layout.

    Returns (images_copied, labels_written).
    """
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    img_paths = sorted(dota_split_images.glob("*.png")) + sorted(dota_split_images.glob("*.jpg"))
    n_img = 0
    n_lbl = 0
    for img_path in tqdm(img_paths, desc=f"convert {dota_split_images.name}"):
        stem = img_path.stem
        lbl_path = dota_label_txt / f"{stem}.txt"
        if not lbl_path.is_file():
            continue
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]
        lines_out: list[str] = []
        with lbl_path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                parsed = parse_dota_label_line(raw)
                if parsed is None:
                    continue
                xs, ys, category, diff = parsed
                if category != CAR_CLASS_DOTA:
                    continue
                if skip_difficult and diff == "1":
                    continue
                xmin, ymin, xmax, ymax = _quad_to_aabb(xs, ys)
                xmin = max(0.0, min(float(w - 1), xmin))
                xmax = max(0.0, min(float(w - 1), xmax))
                ymin = max(0.0, min(float(h - 1), ymin))
                ymax = max(0.0, min(float(h - 1), ymax))
                if xmax <= xmin or ymax <= ymin:
                    continue
                yl = _aabb_to_yolo_line(xmin, ymin, xmax, ymax, w, h)
                if yl:
                    lines_out.append(yl)
        if not lines_out:
            # Skip images with zero car boxes to avoid confusing empty labels.
            continue
        dest_im = out_images_dir / img_path.name
        shutil.copy2(img_path, dest_im)
        dest_lb = out_labels_dir / f"{stem}.txt"
        dest_lb.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
        n_img += 1
        n_lbl += 1
    return n_img, n_lbl


def default_dota_paths(dota_root: Path, split: str) -> tuple[Path, Path]:
    split = split.strip().lower()
    images = dota_root / split / "images"
    labels = dota_root / split / "labelTxt"
    if not images.is_dir():
        images = dota_root / "train" / "images" if split == "train" else images
    return images, labels


def main(argv: Iterable[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="DOTA → YOLO conversion (small-vehicle as car).")
    p.add_argument("--dota-root", type=Path, required=True, help="DOTA root containing train/, val/, ...")
    p.add_argument("--out-root", type=Path, required=True, help="Output YOLO dataset root")
    p.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Comma-separated splits (folder names under dota-root), e.g. train,val",
    )
    p.add_argument("--keep-difficult", action="store_true", help="Keep difficult=1 DOTA instances")
    args = p.parse_args(list(argv) if argv is not None else None)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for sp in splits:
        img_dir, lbl_dir = default_dota_paths(args.dota_root, sp)
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            raise SystemExit(f"Missing DOTA folders for split '{sp}':\n  {img_dir}\n  {lbl_dir}")
        out_im = args.out_root / "images" / sp
        out_lb = args.out_root / "labels" / sp
        n_im, n_lb = convert_dota_split_to_yolo(
            img_dir,
            lbl_dir,
            out_im,
            out_lb,
            skip_difficult=not args.keep_difficult,
        )
        print(f"[{sp}] images with ≥1 car: {n_im}, label files: {n_lb}")


if __name__ == "__main__":
    main()
