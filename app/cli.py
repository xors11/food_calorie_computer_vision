"""
app/cli.py — CLI mode handler.

Analyse a single image file: detect food, print a nutrition table,
save an annotated copy.
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.constants import FOOD_CALORIES_PER_100G
from app.detector import FoodDetector
from app.estimation import estimate_serving
from app.nutrition import get_nutrition


def fmt(label: str) -> str:
    """'french_fries' → 'French Fries'"""
    return label.replace("_", " ").title()


def run_cli(image_path: str) -> None:
    """
    Analyse a single image file.
    Prints a nutrition table and saves an annotated copy of the image.
    """
    import cv2

    img_p = Path(image_path)
    if not img_p.exists():
        print(f"ERROR: File not found — {image_path}")
        sys.exit(1)

    frame = cv2.imread(str(img_p))
    if frame is None:
        print(f"ERROR: Could not read image — {image_path}")
        sys.exit(1)

    print(f"\nAnalysing: {img_p.name}  ({frame.shape[1]}×{frame.shape[0]})")

    detector = FoodDetector()
    if detector.model_error:
        print(f"ERROR: {detector.model_error}")
        sys.exit(1)

    result = detector.detect(frame, conf=0.15)

    # Deduplicate: keep highest-confidence detection per label
    best: dict[str, tuple[float, float]] = {}  # label → (conf, area_ratio)
    for det in result.detections:
        if det.label not in best or det.confidence > best[det.label][0]:
            best[det.label] = (det.confidence, det.bbox_area_ratio)

    # Build serving estimates
    estimates = []
    for label, (conf, area_ratio) in best.items():
        nutrition_info = get_nutrition(label)
        est = estimate_serving(
            label=label,
            display=fmt(label),
            confidence=conf,
            bbox_area_ratio=area_ratio,
            nutrition_info=nutrition_info,
        )
        estimates.append(est)

    # Sort by confidence descending
    estimates.sort(key=lambda e: -e.confidence)

    # Print nutrition table
    print()
    print("─" * 72)
    print(
        f"  {'Food Item':<22} {'Serving':<18} "
        f"{'kcal':>6}  {'P':>5}  {'F':>5}  {'C':>5}  {'Conf':>6}"
    )
    print("─" * 72)

    total_cal = 0.0
    total_p = 0.0
    total_f = 0.0
    total_c = 0.0

    if estimates:
        for est in estimates:
            print(
                f"  {est.display:<22} {est.serving_desc:<18} "
                f"{est.calories:>6.0f}  {est.protein:>5.1f}  "
                f"{est.fat:>5.1f}  {est.carbs:>5.1f}  {est.confidence:>5.0%}"
            )
            total_cal += est.calories
            total_p += est.protein
            total_f += est.fat
            total_c += est.carbs
    else:
        print("  No food detected in this image.")

    print("─" * 72)
    print(
        f"  {'MEAL TOTAL':<22} {'':18} "
        f"{total_cal:>6.0f}  {total_p:>5.1f}  {total_f:>5.1f}  {total_c:>5.1f}"
    )
    print("─" * 72)

    # Save annotated image
    for det in result.detections:
        cv2.rectangle(
            frame, (det.x1, det.y1), (det.x2, det.y2),
            (120, 220, 100), 2,
        )
        tag = f"{det.display}  {det.confidence:.0%}"
        cv2.putText(
            frame, tag, (det.x1 + 4, max(20, det.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1,
            cv2.LINE_AA,
        )

    out_path = img_p.parent / f"{img_p.stem}_detected{img_p.suffix}"
    cv2.imwrite(str(out_path), frame)
    print(f"\n  Annotated image → {out_path}")
