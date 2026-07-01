#!/usr/bin/env python3
"""
evaluate_model.py
=================
Evaluates the custom-trained YOLOv8 model on the Food-101 validation split.
Reports precision, recall, mAP50, and mAP50-95.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
YAML_PATH   = SCRIPT_DIR / "food101.yaml"
CUSTOM_MODEL  = SCRIPT_DIR / "models" / "best.pt"


def main() -> None:
    if not CUSTOM_MODEL.exists():
        print(f"ERROR: Model file not found at {CUSTOM_MODEL}")
        print("Please train the model first by running:")
        print("  python train_model.py")
        sys.exit(1)

    if not YAML_PATH.exists():
        print(f"ERROR: Dataset description {YAML_PATH} not found.")
        print("Please run prepare_dataset.py first.")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    print("=" * 60)
    print("  YOLOv8 — Food-101 Validation / Evaluation")
    print("=" * 60)
    print(f"  Model  : {CUSTOM_MODEL}")
    print(f"  Data   : {YAML_PATH}")
    print("=" * 60)
    print()

    # Load model
    model = YOLO(str(CUSTOM_MODEL))

    # Run validation
    metrics = model.val(
        data=str(YAML_PATH),
        split="val",
        verbose=True,
    )

    print()
    print("=" * 60)
    print("  Evaluation Metrics Summary")
    print("=" * 60)
    print(f"  mAP@50     : {metrics.results_dict.get('metrics/mAP50(B)', 0.0):.4f}")
    print(f"  mAP@50-95  : {metrics.results_dict.get('metrics/mAP50-95(B)', 0.0):.4f}")
    print(f"  Precision  : {metrics.results_dict.get('metrics/precision(B)', 0.0):.4f}")
    print(f"  Recall     : {metrics.results_dict.get('metrics/recall(B)', 0.0):.4f}")
    print("=" * 60)
    print("\nMetrics plots and confusion matrices saved to the validation run folder.")


if __name__ == "__main__":
    main()
