#!/usr/bin/env python3
"""
train_model.py
==============
Trains YOLOv8n on the Food-101 dataset prepared by prepare_dataset.py.

Usage:
  python train_model.py
  python train_model.py --epochs 100 --batch 16 --imgsz 320 --device cpu

Prerequisites:
  1. Run  python prepare_dataset.py  to generate labels and food101.yaml
  2. Install:  pip install ultralytics torch torchvision

Outputs:
  runs/food101/weights/best.pt     (full training artifacts)
  models/best.pt                   (copy used by main.py)
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
YAML_PATH   = SCRIPT_DIR / "food101.yaml"
RUNS_DIR    = SCRIPT_DIR / "runs"
MODELS_DIR  = SCRIPT_DIR / "models"
FINAL_MODEL = MODELS_DIR / "best.pt"


def main() -> None:
    import torch
    default_device = "0" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(
        description="Train YOLOv8n on Food-101",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs",  type=int,   default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch",   type=int,   default=16,
                        help="Batch size (reduce to 8 if GPU OOM)")
    parser.add_argument("--imgsz",   type=int,   default=224,
                        help="Input image size (pixels, square)")
    parser.add_argument("--weights", type=str,   default="yolov8n.pt",
                        help="Pretrained YOLOv8 weights to start from")
    parser.add_argument("--device",  type=str,   default=default_device,
                        help="Device: 'cpu', '0', etc.")
    parser.add_argument("--workers", type=int,   default=0,
                        help="DataLoader worker threads (0 = safe on Windows)")
    parser.add_argument("--patience",type=int,   default=5,
                        help="Early-stopping patience (epochs without improvement)")
    parser.add_argument("--lr",      type=float, default=1e-3,
                        help="Initial learning rate")
    args = parser.parse_args()

    # ── Pre-checks ────────────────────────────────────────────────────────────
    if not YAML_PATH.exists():
        print(f"ERROR: {YAML_PATH} not found.")
        print("Run prepare_dataset.py first:")
        print("  python prepare_dataset.py")
        sys.exit(1)

    # Path existence verification from food101.yaml
    try:
        import yaml
        with open(YAML_PATH, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)
        
        dataset_path = Path(yaml_data.get("path", ""))
        train_relative = yaml_data.get("train", "")
        val_relative = yaml_data.get("val", "")

        train_abs = (dataset_path / train_relative).resolve()
        val_abs = (dataset_path / val_relative).resolve()

        if not train_abs.exists():
            print(f"ERROR: Training path specified in {YAML_PATH.name} does not exist: {train_abs}")
            print("Please run prepare_dataset.py first to create the directory and split the images.")
            sys.exit(1)

        if not val_abs.exists():
            print(f"ERROR: Validation path specified in {YAML_PATH.name} does not exist: {val_abs}")
            print("Please run prepare_dataset.py first to create the directory and split the images.")
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: Failed to read/verify paths from {YAML_PATH}: {e}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.")
        print("  pip install ultralytics")
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Windows safe default: 0 workers avoids multiprocessing spawn issues
    workers = args.workers
    if platform.system() == "Windows" and workers > 0:
        print(f"[INFO] Windows detected — setting workers=0 (was {workers}). "
              "Pass --workers 0 explicitly to suppress this message.")
        workers = 0

    # ── Print config ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  YOLOv8n — Food-101 Training")
    print("=" * 60)
    print(f"  Weights  : {args.weights}")
    print(f"  Data     : {YAML_PATH}")
    print(f"  Epochs   : {args.epochs}  (early-stop patience={args.patience})")
    print(f"  Batch    : {args.batch}")
    print(f"  Imgsz    : {args.imgsz}")
    print(f"  Device   : {args.device}")
    print(f"  Workers  : {workers}")
    print(f"  LR0      : {args.lr}")
    print("=" * 60)
    print()

    # ── Train ─────────────────────────────────────────────────────────────────
    model = YOLO(args.weights)

    model.train(
        data=str(YAML_PATH),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=workers,
        device=args.device,
        project=str(RUNS_DIR),
        name="food101",
        exist_ok=True,                # allow re-running / resuming
        pretrained=True,
        optimizer="AdamW",
        lr0=args.lr,
        lrf=0.01,                     # final lr = lr0 * lrf
        patience=args.patience,
        cache=False,                  # set True to cache images in RAM for speed
        verbose=True,
        plots=True,                   # save training plots to runs/food101/
    )

    # ── Copy best checkpoint to models/best.pt ────────────────────────────────
    best_src = RUNS_DIR / "food101" / "weights" / "best.pt"

    if not best_src.exists():
        # Fallback: search recursively (Ultralytics may use a numbered sub-dir)
        candidates = sorted(RUNS_DIR.rglob("best.pt"))
        if candidates:
            best_src = candidates[-1]
        else:
            print("\nWARNING: best.pt not found in runs/. "
                  "Check runs/ directory manually.")
            return

    shutil.copy(str(best_src), str(FINAL_MODEL))

    print("\n" + "=" * 60)
    print("Training complete. Model saved to: models/best.pt")
    print("Run python main.py to start the app.")
    print("=" * 60)


if __name__ == "__main__":
    main()
