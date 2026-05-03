"""Summarize YOLOv8 validation metrics (mAP@0.5, P, R, F1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionMetrics:
    map50: float
    precision: float
    recall: float
    f1: float


def f1_score(precision: float, recall: float, eps: float = 1e-9) -> float:
    denom = precision + recall
    if denom < eps:
        return 0.0
    return float(2.0 * precision * recall / denom)


def metrics_from_ultralytics_results(results) -> DetectionMetrics:
    """
    ``results`` is the return value of ``YOLO().val(...)`` (Ultralytics).

    Uses mean metrics over classes at IoU 0.5 where available.
    """
    box = results.box
    map50 = float(getattr(box, "map50", getattr(box, "ap50", np.nan)))
    p = float(getattr(box, "mp", getattr(box, "p", np.nan)))
    r = float(getattr(box, "mr", getattr(box, "r", np.nan)))
    if np.isnan(map50) and hasattr(box, "maps"):
        map50 = float(np.mean(box.maps)) if box.maps is not None else 0.0
    return DetectionMetrics(map50=map50, precision=p, recall=r, f1=f1_score(p, r))


def print_metrics(m: DetectionMetrics) -> None:
    print(f"mAP@0.5:   {m.map50:.4f}")
    print(f"Precision: {m.precision:.4f}")
    print(f"Recall:    {m.recall:.4f}")
    print(f"F1-score:  {m.f1:.4f}")
