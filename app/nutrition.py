"""
app/nutrition.py — Unified nutrition lookup with full macros (H4).

Provides a single entry point for all nutrition data queries.
Priority: nutrition_db.py (JSON + USDA API) → built-in constants fallback.

Returns full macro breakdown: calories, protein, fat, carbs, fiber, sugar, sodium.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.constants import FOOD_CALORIES_PER_100G

logger = logging.getLogger("FoodTracker.Nutrition")

# ──────────────────────────────────────────────────────────────────────────────
# Try to import the full nutrition_db module
# ──────────────────────────────────────────────────────────────────────────────
try:
    import nutrition_db as _ndb
except ImportError:
    _ndb = None
    logger.info("nutrition_db module not available — using built-in calorie table.")

# ──────────────────────────────────────────────────────────────────────────────
# Default nutrition template (returned when no data source has the food)
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULT_NUTRITION: dict = {
    "calories": 200,
    "protein":  5.0,
    "fat":      8.0,
    "carbs":    25.0,
    "fiber":    2.0,
    "sugar":    5.0,
    "sodium":   300,
    "serving_g": 100,
    "serving_desc": "1 serving (est.)",
    "source": "default",
}


def get_nutrition(label: str) -> dict:
    """
    Return a nutrition dict for a food label.

    Keys: calories, protein, fat, carbs, fiber, sugar, sodium (per 100g),
          serving_g, serving_desc, source.

    Never raises.
    """
    # 1. Try nutrition_db (JSON + USDA API)
    if _ndb is not None:
        try:
            info = _ndb.get_nutrition(label)
            if info and info.get("calories", 0) > 0:
                # Ensure all macro keys exist (backfill missing ones)
                result = dict(info)
                result.setdefault("fiber", 0.0)
                result.setdefault("sugar", 0.0)
                result.setdefault("sodium", 0)
                return result
        except Exception:
            pass

    # 2. Built-in calories fallback (no macros)
    cal = FOOD_CALORIES_PER_100G.get(label, 0)
    if cal > 0:
        fallback = dict(_DEFAULT_NUTRITION)
        fallback["calories"] = cal
        fallback["source"] = "built-in"
        return fallback

    # 3. Hard default
    return dict(_DEFAULT_NUTRITION)


def get_calories(label: str) -> int:
    """Return kcal per 100g for a food label. Convenience shortcut."""
    return int(get_nutrition(label).get("calories", 200))


def get_macro_summary(label: str) -> str:
    """
    One-line macro string for UI display.
    e.g. 'P 11g · F 10g · C 33g'
    """
    info = get_nutrition(label)
    factor = info.get("serving_g", 100) / 100.0
    p = round(info.get("protein", 0) * factor, 1)
    f = round(info.get("fat", 0) * factor, 1)
    c = round(info.get("carbs", 0) * factor, 1)
    return f"P {p}g · F {f}g · C {c}g"
