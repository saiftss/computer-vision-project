"""Plot training curves from Ultralytics ``results.csv`` and save prediction figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO


def plot_training_curves(results_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_csv)
    epoch = df["epoch"] if "epoch" in df.columns else range(len(df))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax0, ax1, ax2, ax3 = axes.ravel()

    if "train/box_loss" in df.columns:
        ax0.plot(epoch, df["train/box_loss"], label="train box")
    if "train/cls_loss" in df.columns:
        ax0.plot(epoch, df["train/cls_loss"], label="train cls")
    if "train/dfl_loss" in df.columns:
        ax0.plot(epoch, df["train/dfl_loss"], label="train dfl")
    ax0.set_title("Training losses")
    ax0.set_xlabel("epoch")
    ax0.set_ylabel("loss")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    if "metrics/precision(B)" in df.columns:
        ax1.plot(epoch, df["metrics/precision(B)"], label="precision")
    if "metrics/recall(B)" in df.columns:
        ax1.plot(epoch, df["metrics/recall(B)"], label="recall")
    ax1.set_title("Precision / recall (box)")
    ax1.set_xlabel("epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for col, title in [
        ("metrics/mAP50(B)", "mAP@0.5"),
        ("metrics/mAP50-95(B)", "mAP@0.5:0.95"),
    ]:
        if col in df.columns:
            ax2.plot(epoch, df[col], label=title)
    ax2.set_title("mAP metrics")
    ax2.set_xlabel("epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    lr_col = next((c for c in df.columns if c.startswith("lr/pg")), None)
    if lr_col:
        ax3.plot(epoch, df[lr_col], label=lr_col)
    ax3.set_title("Learning rate")
    ax3.set_xlabel("epoch")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def run_predictions(weights: Path, source: Path, out_dir: Path, imgsz: int, conf: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    model.predict(
        source=str(source),
        imgsz=imgsz,
        conf=conf,
        save=True,
        project=str(out_dir),
        name="pred",
        exist_ok=True,
    )
    print(f"Predictions written under {out_dir / 'pred'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize training curves and predictions.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_curves = sub.add_parser("curves", help="Plot results.csv from a training run")
    p_curves.add_argument("--results", type=Path, required=True)
    p_curves.add_argument("--out", type=Path, default=Path("figures"))

    p_pred = sub.add_parser("predict", help="Draw boxes on images or a video folder")
    p_pred.add_argument("--weights", type=Path, required=True)
    p_pred.add_argument("--source", type=Path, required=True, help="Image, folder, or video path")
    p_pred.add_argument("--out", type=Path, default=Path("figures"))
    p_pred.add_argument("--imgsz", type=int, default=1024)
    p_pred.add_argument("--conf", type=float, default=0.25)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "curves":
        plot_training_curves(args.results, args.out)
    elif args.cmd == "predict":
        run_predictions(args.weights, args.source, args.out, args.imgsz, args.conf)


if __name__ == "__main__":
    main()
