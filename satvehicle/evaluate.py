"""Run validation / test evaluation and print standard detection metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from satvehicle.metrics_report import metrics_from_ultralytics_results, print_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a YOLOv8 vehicle detector.")
    p.add_argument("--weights", type=Path, required=True, help="Path to best.pt or last.pt")
    p.add_argument("--data", type=Path, required=True, help="Ultralytics data YAML")
    p.add_argument("--split", type=str, default="val", choices=("val", "test"))
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights))
    split = args.split
    metrics = model.val(
        data=str(args.data),
        split=split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        plots=True,
    )
    print_metrics(metrics_from_ultralytics_results(metrics))


if __name__ == "__main__":
    main()
