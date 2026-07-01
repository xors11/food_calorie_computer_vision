"""
app/estimation.py — Serving size and calorie estimation from bounding boxes.

Uses the bounding-box area relative to the frame as a proxy for portion size.
This is the simplest practical approach that doesn't require depth sensors,
reference objects, or additional ML models.

Approach:
  1. Compute bbox_area_ratio = (box_w × box_h) / (frame_w × frame_h)
  2. Map to a serving multiplier via size buckets
  3. Scale the food's standard serving_g and calories accordingly

Accuracy: ±30-50% — suitable for rough estimation, not clinical use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.constants import SERVING_BUCKETS

logger = logging.getLogger("FoodTracker.Estimation")


@dataclass
class ServingEstimate:
    """Estimated serving size and nutritional values for one detection."""
    label: str
    display: str
    confidence: float

    # Serving estimation
    serving_multiplier: float   # e.g. 1.0 = standard, 1.5 = large
    serving_g: float            # estimated grams
    serving_desc: str           # e.g. "~1 serving (150g)"

    # Calories — total for the estimated serving
    calories: float

    # Macros — scaled to the estimated serving
    protein: float
    fat: float
    carbs: float
    fiber: float
    sugar: float
    sodium: float

    # Source for display
    source: str                 # "local", "USDA", "default"


def estimate_serving_multiplier(bbox_area_ratio: float) -> float:
    """
    Map a bounding-box area ratio to a serving multiplier.

    bbox_area_ratio = (box_w × box_h) / (frame_w × frame_h)

    Returns a multiplier (0.5, 1.0, 1.5, or 2.0) based on how much
    of the frame the food item occupies.
    """
    for max_ratio, multiplier in SERVING_BUCKETS:
        if bbox_area_ratio <= max_ratio:
            return multiplier
    return SERVING_BUCKETS[-1][1]  # largest bucket


def estimate_serving(
    label: str,
    display: str,
    confidence: float,
    bbox_area_ratio: float,
    nutrition_info: dict,
) -> ServingEstimate:
    """
    Create a full serving estimate for a detected food item.

    Args:
        label:           YOLO class name, e.g. "french_fries"
        display:         Formatted name, e.g. "French Fries"
        confidence:      Detection confidence 0.0–1.0
        bbox_area_ratio: Fraction of frame occupied by the bounding box
        nutrition_info:  Dict from nutrition lookup (per 100g values)

    Returns:
        ServingEstimate with all computed values.
    """
    multiplier = estimate_serving_multiplier(bbox_area_ratio)

    # Standard serving size from nutrition data
    base_serving_g = nutrition_info.get("serving_g", 100)
    estimated_g = base_serving_g * multiplier
    serving_desc_base = nutrition_info.get("serving_desc", "1 serving")

    # Build human-readable serving description
    if multiplier == 1.0:
        serving_desc = f"~{serving_desc_base} ({estimated_g:.0f}g)"
    elif multiplier < 1.0:
        serving_desc = f"~½ serving ({estimated_g:.0f}g)"
    else:
        serving_desc = f"~{multiplier:.1f}× serving ({estimated_g:.0f}g)"

    # Scale all nutrients from per-100g to estimated serving
    factor = estimated_g / 100.0
    calories_per_100g = nutrition_info.get("calories", 200)

    return ServingEstimate(
        label=label,
        display=display,
        confidence=confidence,
        serving_multiplier=multiplier,
        serving_g=estimated_g,
        serving_desc=serving_desc,
        calories=round(calories_per_100g * factor, 1),
        protein=round(nutrition_info.get("protein", 0.0) * factor, 1),
        fat=round(nutrition_info.get("fat", 0.0) * factor, 1),
        carbs=round(nutrition_info.get("carbs", 0.0) * factor, 1),
        fiber=round(nutrition_info.get("fiber", 0.0) * factor, 1),
        sugar=round(nutrition_info.get("sugar", 0.0) * factor, 1),
        sodium=round(nutrition_info.get("sodium", 0.0) * factor, 1),
        source=nutrition_info.get("source", "default"),
    )
