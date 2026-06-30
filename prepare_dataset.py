#!/usr/bin/env python3
"""
prepare_dataset.py
==================
Prepares the Food-101 dataset for YOLOv8 training.

Steps:
  1. Reads all 101 class folders from  <data-dir>/images/
  2. Auto-generates YOLO labels  (full-image box: cx=0.5 cy=0.5 w=1.0 h=1.0)
  3. Splits images 80 % train / 20 % val  (reproducible seed)
  4. Saves labels →  <data-dir>/labels/<class_name>/<image>.txt
  5. Writes       →  <data-dir>/train.txt  and  val.txt
  6. Writes       →  food101.yaml  (placed next to this script)

Usage:
  python prepare_dataset.py
  python prepare_dataset.py --data-dir /absolute/path/to/food-101

Default data-dir search order (relative to this script):
  1. ../food-101      (dataset is sibling of food-calorie-detector/)
  2. ./food-101       (dataset is inside food-calorie-detector/)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

# ── Optional tqdm for progress bars ───────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):          # type: ignore[misc]
        print(f"  Processing {kwargs.get('desc', '')}…")
        return iterable

# ── Config ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
YAML_OUT    = SCRIPT_DIR / "food101.yaml"
TRAIN_RATIO = 0.80
RANDOM_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_data_dir(override: str | None) -> Path:
    """Return the resolved path to the food-101 root directory."""
    if override:
        p = Path(override).resolve()
        if not p.exists():
            print(f"ERROR: --data-dir path does not exist: {p}")
            sys.exit(1)
        return p

    candidates = [
        SCRIPT_DIR.parent / "food-101",   # ../food-101
        SCRIPT_DIR / "food-101",           # ./food-101
    ]
    for c in candidates:
        if (c / "images").exists():
            print(f"Found Food-101 dataset at: {c}")
            return c.resolve()

    print("ERROR: Food-101 dataset not found. Tried:")
    for c in candidates:
        print(f"  {c}")
    print("\nOptions:")
    print("  1. Place the dataset at  ../food-101/images/<class>/")
    print("  2. Run:  python prepare_dataset.py --data-dir /path/to/food-101")
    sys.exit(1)


def _write_yaml(path: Path, data: dict) -> None:
    """Write a YAML file without requiring PyYAML."""
    try:
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False,
                      default_flow_style=False)
        return
    except ImportError:
        pass

    # Manual fallback
    lines: list[str] = []
    for k, v in data.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        elif isinstance(v, int):
            lines.append(f"{k}: {v}")
        else:
            lines.append(f"{k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_or_symlink(src: Path, dst: Path) -> None:
    """Creates a symbolic link or copies the file if symlinking fails.
    Includes retry logic for OneDrive permission errors on Windows."""
    import shutil
    import time
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    # Try symlink first
    try:
        os.symlink(src.resolve(), dst.resolve())
        return
    except (OSError, PermissionError):
        pass

    # Fall back to copy with retries (OneDrive may lock files briefly)
    for attempt in range(3):
        try:
            shutil.copy2(src, dst)
            return
        except PermissionError:
            if attempt < 2:
                time.sleep(0.5)
            else:
                # Last resort: write bytes directly
                try:
                    dst.write_bytes(src.read_bytes())
                    return
                except Exception as e:
                    print(f"  WARNING: Could not copy {src.name}: {e}")
                    return

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Food-101 for YOLOv8 training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Path to the food-101 root directory (contains images/ subfolder)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=TRAIN_RATIO,
        help="Fraction of images used for training",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed for reproducible splits",
    )
    args = parser.parse_args()

    data_dir   = _find_data_dir(args.data_dir)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"

    # ── 1. Discover classes (Alphabetized order) ──────────────────────────────
    target_classes = [
        'apple_pie', 'chicken_curry', 'chicken_wings', 'dumplings',
        'french_toast', 'fried_calamari', 'fried_rice', 'garlic_bread',
        'hamburger', 'hot_and_sour_soup', 'omelette', 'pancakes',
        'pizza', 'samosa', 'spring_rolls'
    ]

    class_dirs = []
    for cls_name in target_classes:
        cls_dir = images_dir / cls_name
        if cls_dir.is_dir():
            class_dirs.append(cls_dir)
        else:
            print(f"Skipping class '{cls_name}' (directory not found under {images_dir})")

    nc = len(target_classes)
    print(f"\n{'-'*50}")
    print(f"  Target Classes: {nc}")
    print(f"  Existing dirs : {len(class_dirs)}")
    print(f"  Images root   : {images_dir}")
    print(f"  Labels root   : {labels_dir}")
    print(f"  YAML output   : {YAML_OUT}")
    print(f"  Train ratio   : {args.train_ratio:.0%}")
    print(f"  Random seed   : {args.seed}")
    print(f"{'-'*50}\n")

    # ── 2. Generate labels + split paths + copy/symlink files ──────────────────
    skipped = 0
    total_train = 0
    total_val = 0

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        cls_idx = target_classes.index(cls_name)

        # Collect images (case-insensitive extension matching)
        images: list[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            images.extend(cls_dir.glob(ext))
        images = sorted(set(images))

        # Filter out any nested train/val folders that might be inside (should not happen, but safe)
        images = [img for img in images if "train" not in img.parts and "val" not in img.parts]

        if not images:
            print(f"  WARNING: No images in class '{cls_name}' — skipped.")
            skipped += 1
            continue

        # Reproducible per-class shuffle
        rng = random.Random(args.seed + cls_idx)
        rng.shuffle(images)

        split_at   = max(1, int(len(images) * args.train_ratio))
        train_imgs = images[:split_at]
        val_imgs   = images[split_at:]

        # Create splits and generate labels
        label_line = f"{cls_idx} 0.5 0.5 1.0 1.0\n"

        # Copy train split
        for img in train_imgs:
            dst_img = images_dir / "train" / cls_name / img.name
            _copy_or_symlink(img, dst_img)

            dst_lbl = labels_dir / "train" / cls_name / (img.stem + ".txt")
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.write_text(label_line, encoding="utf-8")
            total_train += 1

        # Copy val split
        for img in val_imgs:
            dst_img = images_dir / "val" / cls_name / img.name
            _copy_or_symlink(img, dst_img)

            dst_lbl = labels_dir / "val" / cls_name / (img.stem + ".txt")
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.write_text(label_line, encoding="utf-8")
            total_val += 1

        print(f"Processing {cls_name}... {len(images)} images split (train={len(train_imgs)}, val={len(val_imgs)})")

    # ── 3. Write food101.yaml with absolute path ──────────────────────────────
    yaml_data = {
        "path":  str(data_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    nc,
        "names": target_classes,
    }
    _write_yaml(YAML_OUT, yaml_data)

    # ── 4. Print Summary and YAML Contents ────────────────────────────────────
    print(f"\n{'-'*50}")
    print(f"  *  Classes    : {nc - skipped} ({skipped} skipped)")
    print(f"  *  Train      : {total_train:,} images split into images/train")
    print(f"  *  Val        : {total_val:,} images split into images/val")
    print(f"  *  Labels     : {labels_dir}")
    print(f"  *  YAML       : {YAML_OUT}")
    print(f"{'-'*50}\n")

    print(f"--- Contents of {YAML_OUT.name} ---")
    if YAML_OUT.exists():
        print(YAML_OUT.read_text(encoding="utf-8"))
    print("---------------------------------")
    print("\nDataset ready. Next step:")
    print("  python train_model.py")


if __name__ == "__main__":
    main()
