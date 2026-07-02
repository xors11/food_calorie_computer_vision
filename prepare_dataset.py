#!/usr/bin/env python3
"""
prepare_dataset.py
==================
An automated dataset preparation system for the AI Food Calorie Tracker.

This script scans Food-101 and Indian Food Images datasets, normalizes class names,
merges duplicate classes, splits images into training (80%) and validation (20%) sets,
checks for corruption, generates YOLO configuration, and creates YOLO label files.

Features:
  - Dynamically scans both datasets using pathlib.
  - Normalizes class names (lowercase, replaces spaces/hyphens with underscores, strips suffixes).
  - Merges duplicate class names across datasets.
  - Skips empty folders and corrupted images (using PIL.Image.verify()).
  - Generates unique image filenames to avoid collision.
  - Generates YOLO label files (class_id 0.5 0.5 1.0 1.0) automatically.
  - Exports a unified dataset YAML file.
  - Completely configurable via CLI arguments (argparse).
  - Platform-agnostic (Windows, Linux, Colab).
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Suffixes that are redundant in class directory names and will be removed during normalization
REDUNDANT_SUFFIXES = [
    "_images", "_image", "_photos", "_photo", "_pics", "_pic",
    "_dataset", "_food", "_foods"
]


def normalize_class_name(name: str) -> str:
    """
    Normalizes a folder name to establish a unified class name.
    
    Normalization steps:
      1. Convert to lowercase.
      2. Replace spaces and hyphens with underscores.
      3. Replace all non-alphanumeric characters (except underscores) with underscores.
      4. Collapse consecutive underscores into a single underscore.
      5. Strip redundant suffixes (e.g. '_images', '_dataset').
      6. Strip leading and trailing underscores.
      
    Example:
      "Butter Chicken" -> "butter_chicken"
      "pizza_images"   -> "pizza"
      "Butter-Chicken" -> "butter_chicken"
    """
    # Lowercase
    name = name.lower()
    
    # Replace spaces and hyphens with underscores
    name = name.replace(" ", "_").replace("-", "_")
    
    # Replace any other non-alphanumeric character with underscore
    name = re.sub(r"[^a-z0-9_]", "_", name)
    
    # Collapse consecutive underscores
    name = re.sub(r"_+", "_", name)
    
    # Strip redundant suffixes in a loop until no more changes occur
    changed = True
    while changed:
        changed = False
        for suffix in REDUNDANT_SUFFIXES:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                changed = True
                break
                
    # Remove leading/trailing underscores
    name = name.strip("_")
    return name


def check_image_valid(img_path: Path) -> bool:
    """
    Validates that an image file exists, is non-empty, and is not corrupted.
    Uses PIL's Image.verify() for a fast, memory-efficient check.
    """
    if not img_path.exists():
        return False
    
    # Skip empty files
    if img_path.stat().st_size == 0:
        return False
        
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def scan_dataset_directory(
    dataset_dir: Path,
    forbidden_folders: set[str],
    output_dir: Path | None = None
) -> dict[str, dict[str, any]]:
    """
    Recursively scans a dataset directory for folders containing images.
    
    Parameters:
      dataset_dir: Root directory of the dataset.
      forbidden_folders: Directory names to skip (e.g. 'train', 'val', 'labels').
      output_dir: Optional path to the output directory to ignore during scanning.
      
    Returns:
      A dictionary mapping normalized class name to a dict containing:
        - "images": list of Path objects pointing to image files
        - "raw_folders": set of original raw folder names mapping to this class
    """
    class_data: dict[str, dict[str, any]] = {}
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    
    if not dataset_dir.exists():
        return class_data

    # Iterate over all subdirectories
    for path in dataset_dir.rglob("*"):
        if not path.is_dir():
            continue
            
        # Skip if the path is inside the output directory (to avoid scanning output splits)
        if output_dir is not None:
            try:
                path.relative_to(output_dir)
                continue
            except ValueError:
                pass
                
        # Check if the folder is in any forbidden/split directory
        try:
            rel_parts = path.relative_to(dataset_dir).parts
        except ValueError:
            rel_parts = path.parts
            
        if any(part.lower() in forbidden_folders for part in rel_parts):
            continue
            
        # Find all valid image files directly inside this subdirectory
        image_files = []
        for file in path.iterdir():
            if file.is_file() and file.suffix.lower() in valid_extensions:
                image_files.append(file)
                
        if not image_files:
            continue
            
        # The folder name is the raw class name
        raw_class_name = path.name
        norm_class_name = normalize_class_name(raw_class_name)
        
        # Initialize dictionary keys if not already present
        if norm_class_name not in class_data:
            class_data[norm_class_name] = {
                "images": [],
                "raw_folders": set()
            }
            
        class_data[norm_class_name]["images"].extend(image_files)
        class_data[norm_class_name]["raw_folders"].add(raw_class_name)
        
    return class_data


def main() -> None:
    # ── CLI Arguments ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Automated Food Dataset Builder & YOLO Formatter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--food101-dir", type=str, default="food-101",
        help="Path to the Food-101 directory"
    )
    parser.add_argument(
        "--indian-dir", type=str, default="indian_food_dataset",
        help="Path to the Indian Food Images dataset directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="merged_dataset",
        help="Directory where the unified split dataset will be created"
    )
    parser.add_argument(
        "--yaml-path", type=str, default="merged_dataset.yaml",
        help="Path where the YOLO config YAML will be written"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Fixed random seed for splitting reproducibility"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="Ratio of training split (0.0 to 1.0)"
    )
    parser.add_argument(
        "--no-clean", dest="clean", action="store_false",
        help="Do not clean/delete the output directory if it already exists"
    )
    parser.set_defaults(clean=True)
    args = parser.parse_args()

    # ── Path Resolution ───────────────────────────────────────────────────────
    script_dir = Path(__file__).parent.resolve()
    food101_dir = (script_dir / args.food101_dir).resolve()
    indian_dir = (script_dir / args.indian_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    yaml_path = (script_dir / args.yaml_path).resolve()

    print("=" * 60)
    print("  AI Food Calorie Tracker — Dataset Preparation Pipeline")
    print("=" * 60)
    print(f"  Food-101 Path  : {food101_dir}")
    print(f"  Indian Food Path: {indian_dir}")
    print(f"  Output Directory: {output_dir}")
    print(f"  YAML Config Path: {yaml_path}")
    print(f"  Random Seed     : {args.seed}")
    print(f"  Train/Val Ratio : {int(args.train_ratio * 100)}% / {int((1 - args.train_ratio) * 100)}%")
    print("=" * 60)

    # ── Scan Datasets ─────────────────────────────────────────────────────────
    # Folders we should skip during parsing to avoid duplicate scanning of split data
    forbidden_folders = {"train", "val", "validation", "test", "labels"}

    print("\nScanning Food-101 directory...")
    food101_scan = scan_dataset_directory(food101_dir, forbidden_folders, output_dir=output_dir)
    print(f"  Detected {len(food101_scan)} classes in Food-101.")

    print("\nScanning Indian Food dataset directory...")
    indian_scan = scan_dataset_directory(indian_dir, forbidden_folders, output_dir=output_dir)
    print(f"  Detected {len(indian_scan)} classes in Indian Food Images.")

    # ── Merge and Align Classes ───────────────────────────────────────────────
    all_class_names = set(food101_scan.keys()) | set(indian_scan.keys())
    
    if not all_class_names:
        print("\nERROR: No food class folders or images were found in the scanned directories.")
        print("Please check that your input paths are correct and contain image files.")
        sys.exit(1)

    print(f"\nMerging classes and aligning datasets...")
    unified_dataset: dict[str, dict[str, any]] = {}
    
    merged_classes_names = set(food101_scan.keys()) & set(indian_scan.keys())
    
    # Calculate duplicate classes removed
    total_raw_folders_scanned = 0
    
    for cls in all_class_names:
        unified_dataset[cls] = {
            "images": [],
            "raw_folders": set()
        }
        
        # Gather Food-101 images
        if cls in food101_scan:
            unified_dataset[cls]["images"].extend(
                [(img, "food101") for img in food101_scan[cls]["images"]]
            )
            unified_dataset[cls]["raw_folders"].update(food101_scan[cls]["raw_folders"])
            
        # Gather Indian Food images
        if cls in indian_scan:
            unified_dataset[cls]["images"].extend(
                [(img, "indianfood") for img in indian_scan[cls]["images"]]
            )
            unified_dataset[cls]["raw_folders"].update(indian_scan[cls]["raw_folders"])
            
        total_raw_folders_scanned += len(unified_dataset[cls]["raw_folders"])

    duplicate_classes_removed = total_raw_folders_scanned - len(all_class_names)

    # Sort classes alphabetically to ensure deterministic class ID mapping
    sorted_classes = sorted(list(all_class_names))
    class_to_id = {cls: idx for idx, cls in enumerate(sorted_classes)}

    # ── Initialize Output Directories ─────────────────────────────────────────
    if output_dir.exists() and args.clean:
        print(f"\nCleaning existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    images_train_dir = output_dir / "images" / "train"
    images_val_dir = output_dir / "images" / "val"
    labels_train_dir = output_dir / "labels" / "train"
    labels_val_dir = output_dir / "labels" / "val"

    images_train_dir.mkdir(parents=True, exist_ok=True)
    images_val_dir.mkdir(parents=True, exist_ok=True)
    labels_train_dir.mkdir(parents=True, exist_ok=True)
    labels_val_dir.mkdir(parents=True, exist_ok=True)

    # ── Split, Validate, Copy and Label Generation ─────────────────────────────
    print("\nProcessing images, validating data, and writing YOLO splits...")
    
    total_train_copied = 0
    total_val_copied = 0
    total_corrupted = 0
    empty_classes_skipped = 0

    # Calculate total image entries to process for the global progress bar
    total_raw_images = sum(len(unified_dataset[cls]["images"]) for cls in sorted_classes)
    
    # We will use a nested progress bar or a single global progress bar
    # Using tqdm for single progress bar over classes
    for cls in tqdm(sorted_classes, desc="Classes processed", unit="class"):
        class_id = class_to_id[cls]
        raw_images_with_source = unified_dataset[cls]["images"]
        
        # 1. Filter out corrupted images
        valid_images_with_source = []
        for img_path, src_tag in raw_images_with_source:
            if check_image_valid(img_path):
                valid_images_with_source.append((img_path, src_tag))
            else:
                total_corrupted += 1
                
        # 2. Skip empty class folders
        if not valid_images_with_source:
            empty_classes_skipped += 1
            print(f"\n[WARNING] Skipping class '{cls}': No valid/uncorrupted images found.")
            continue
            
        # 3. Sort files to ensure deterministic shuffling across different runs/OS
        valid_images_with_source.sort(key=lambda x: x[0].name)
        
        # 4. Shuffle using the fixed seed
        random.seed(args.seed)
        random.shuffle(valid_images_with_source)
        
        # 5. Split train/val
        n_images = len(valid_images_with_source)
        split_idx = int(round(n_images * args.train_ratio))
        
        # Ensure at least 1 image is in training
        if split_idx == 0 and n_images > 0:
            split_idx = 1
        # Ensure at least 1 image is in validation if there are at least 2 images
        elif split_idx == n_images and n_images > 1:
            split_idx = n_images - 1
            
        train_split = valid_images_with_source[:split_idx]
        val_split = valid_images_with_source[split_idx:]
        
        # Helper function to copy split and write labels
        def process_split(
            split_data: list[tuple[Path, str]],
            dest_img_dir: Path,
            dest_lbl_dir: Path
        ) -> int:
            copied_count = 0
            for src_path, src_tag in split_data:
                # Construct unique file name to prevent namespace collisions
                # Format: {src_tag}_{norm_class_name}_{original_filename}
                unique_filename = f"{src_tag}_{cls}_{src_path.name}"
                dest_img_path = dest_img_dir / unique_filename
                
                # Copy image
                try:
                    shutil.copy2(src_path, dest_img_path)
                    
                    # Generate YOLO Label file (.txt)
                    # Coordinates are 0.5 0.5 1.0 1.0 to fit the entire image
                    dest_label_path = dest_lbl_dir / f"{dest_img_path.stem}.txt"
                    with open(dest_label_path, "w", encoding="utf-8") as lf:
                        lf.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                        
                    copied_count += 1
                except Exception as e:
                    print(f"\n[ERROR] Failed to process {src_path}: {e}")
            return copied_count

        total_train_copied += process_split(train_split, images_train_dir, labels_train_dir)
        total_val_copied += process_split(val_split, images_val_dir, labels_val_dir)

    # ── Write YOLO YAML Configuration ─────────────────────────────────────────
    # Compute dataset path in YAML relative to the directory where YAML resides
    try:
        yaml_parent = yaml_path.parent.resolve()
        relative_path = output_dir.resolve().relative_to(yaml_parent)
        path_yaml_str = str(relative_path.as_posix())
    except ValueError:
        # Cross-drive or absolute path required
        path_yaml_str = str(output_dir.resolve().as_posix())

    # Write the YAML file directly matching requirements
    print(f"\nWriting dataset config to: {yaml_path}")
    try:
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(f"path: {path_yaml_str}\n")
            f.write("train: images/train\n")
            f.write("val: images/val\n\n")
            f.write(f"nc: {len(sorted_classes)}\n\n")
            f.write("names:\n")
            for cls in sorted_classes:
                f.write(f"  - {cls}\n")
    except Exception as e:
        print(f"ERROR: Failed to write {yaml_path}: {e}")
        sys.exit(1)

    # ── Print Detailed Summary Report ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DATASET PREPARATION PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Food-101 classes detected : {len(food101_scan)}")
    print(f"  Indian Food classes detected: {len(indian_scan)}")
    print(f"  Merged classes            : {len(merged_classes_names)}")
    print(f"  Duplicate classes removed : {duplicate_classes_removed}")
    print(f"  Empty classes skipped     : {empty_classes_skipped}")
    print(f"  Total classes             : {len(sorted_classes)}")
    print(f"  Training images copied    : {total_train_copied}")
    print(f"  Validation images copied  : {total_val_copied}")
    print(f"  Corrupted images skipped  : {total_corrupted}")
    print("=" * 60)
    print("  Setup complete! You can now run model training:")
    print(f"  python train_model.py --yaml {yaml_path.name}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
