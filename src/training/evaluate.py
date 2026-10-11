"""Evaluate a chosen checkpoint on validation data and record real detection metrics."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def evaluate(weights, data, output, device, image_size, batch_size):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    result = model.val(data=str(data), split="val", device=device, imgsz=image_size,
                       batch=batch_size, workers=0, plots=False, verbose=False,
                       project=str(output), name="evaluation", exist_ok=False)
    metrics = {"precision": float(result.box.mp), "recall": float(result.box.mr),
               "map50": float(result.box.map50), "map50_95": float(result.box.map)}
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in metrics.values()):
        raise ValueError("Evaluation produced invalid detection metrics")
    report = {
        "evaluation_split": "validation", "metrics": metrics,
        "per_class": [{key: value.item() if isinstance(value, np.generic) else value
                       for key, value in row.items()} for row in result.summary()],
        "checkpoint": str(Path(weights).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
        "stable_model": False,
        "note": "Validation metrics are used for selection, not an unbiased final-test estimate.",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
