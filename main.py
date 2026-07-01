#!/usr/bin/env python3
"""
main.py — AI Food Calorie Tracker (entry point)
=================================================

Two modes:
  CLI  →  python main.py <image_path>     detect food in a static image
  GUI  →  python main.py                  live webcam + Tkinter window

Model priority (automatic):
  1. models/best.pt    — Food-101 fine-tuned YOLOv8n
  2. yolov8n.pt        — COCO pretrained fallback (~7 food classes)

All application logic lives in the app/ package:
  app/detector.py    — YOLO inference with temporal smoothing + NMS tuning
  app/estimation.py  — Serving size estimation from bounding box area
  app/nutrition.py   — Unified nutrition lookup (JSON + USDA API)
  app/tracker.py     — Tkinter GUI class
  app/widgets.py     — Reusable UI components (FoodCard, StatusBar, etc.)
  app/cli.py         — CLI mode handler
  app/constants.py   — All configuration constants
"""

from __future__ import annotations

import logging
import sys

# ──────────────────────────────────────────────────────────────────────────────
# Logging configuration
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def _print_usage() -> None:
    print("Usage:")
    print("  python main.py <image_path>   — CLI: analyse an image file")
    print("  python main.py                — GUI: live webcam tracker")


if __name__ == "__main__":
    args = sys.argv[1:]

    if args and args[0] in ("-h", "--help"):
        _print_usage()
        sys.exit(0)

    if args:
        # ── CLI mode ──────────────────────────────────────────────────────────
        from app.cli import run_cli
        run_cli(args[0])
    else:
        # ── GUI mode ──────────────────────────────────────────────────────────
        import tkinter as tk
        from app.tracker import FoodCalorieTracker

        root = tk.Tk()
        app  = FoodCalorieTracker(root)
        root.mainloop()