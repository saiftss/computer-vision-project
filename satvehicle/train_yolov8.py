"""
Train YOLOv8 on a YOLO-format dataset (DOTA cars or other aerial data).

Ultralytics runs optimized training and validation loops internally, logs losses and
metrics to ``runs/detect/<name>/results.csv``, and saves ``best.pt`` when enabled.

CLI example:
    python -m satvehicle.train_yolov8 --data config/data_dota_car.yaml --model n --epochs 50 --imgsz 1024
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from satvehicle.metrics_report import metrics_from_ultralytics_results, print_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 for vehicle detection.")
    p.add_argument("--data", type=Path, required=True, help="Ultralytics data YAML")
    p.add_argument(
        "--model",
        type=str,
        default="n",
        help="Model size: n, s, m, l, x or a path like yolov8n.pt",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=1024, help="Train/val square size (prefer ≥1024 for small cars)")
    p.add_argument("--project", type=str, default="runs/detect")
    p.add_argument("--name", type=str, default="vehicle_satellite")
    p.add_argument("--patience", type=int, default=30, help="Early stopping patience (epochs)")
    p.add_argument("--device", type=str, default="", help="cuda device, e.g. 0 or cpu")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_augment", action="store_true", help="Disable default YOLO augmentations")
    return p.parse_args()


def resolve_weights(model_arg: str) -> str:
    m = model_arg.lower().strip()
    if m.endswith(".pt") or m.endswith(".yaml"):
        return model_arg
    if m in {"n", "s", "m", "l", "x"}:
        return f"yolov8{m}.pt"
    return model_arg


def main() -> None:
    args = parse_args()
    weights = resolve_weights(args.model)
    model = YOLO(weights)

    train_kw: dict = dict(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        seed=args.seed,
        plots=True,
        save=True,
        val=True,
        device=args.device if args.device else None,
        # Augmentations (mirror assignment: rotation, flips, scale, photometric)
        degrees=180.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.2,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=0.7,
    )
    if args.no_augment:
        train_kw.update(
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            fliplr=0.0,
            flipud=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
            mosaic=0.0,
        )

    model.train(**train_kw)

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if best.is_file():
        print(f"\nReloading best weights: {best}")
        model = YOLO(str(best))
    metrics = model.val(data=str(args.data))
    print("\n--- Validation summary (best or last) ---")
    print_metrics(metrics_from_ultralytics_results(metrics))


if __name__ == "__main__":
    main()
