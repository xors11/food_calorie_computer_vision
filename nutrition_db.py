"""
nutrition_db.py — Nutrition data lookup module.

Priority chain for every query:
  1. In-memory runtime cache (instant)
  2. Local JSON file  (nutrition_data.json, < 1 ms)
  3. USDA FoodData Central REST API (network, ~200–500 ms, cached after first hit)

All calorie figures returned by get_serving_calories() are per the food's
typical serving size (serving_g grams), NOT per 100 g.

Usage:
    import nutrition_db
    info = nutrition_db.get_nutrition("french_fries")
    # -> {"calories": 312, "protein": 3.4, "fat": 15.0, "carbs": 41.0,
    #     "serving_g": 150, "serving_desc": "1 serving", "source": "local"}
    kcal = nutrition_db.get_serving_calories("french_fries")  # -> 468.0
"""

from __future__ import annotations

import json
import os
import threading
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
_LOCAL_DB_PATH = os.path.join(_DIR, "nutrition_data.json")
_API_CACHE_PATH = os.path.join(_DIR, "nutrition_cache.json")

# ---------------------------------------------------------------------------
# USDA FoodData Central
# ---------------------------------------------------------------------------
USDA_API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
# Set env var USDA_API_KEY for higher rate limits (free at fdc.nal.usda.gov).
# DEMO_KEY allows ≈1000 req/hour and works without registration.
USDA_API_KEY = os.getenv("USDA_API_KEY", "DEMO_KEY")

# ---------------------------------------------------------------------------
# Default returned when all sources fail
# ---------------------------------------------------------------------------
DEFAULT_NUTRITION: dict = {
    "calories": 200,
    "protein": 5.0,
    "fat": 8.0,
    "carbs": 25.0,
    "serving_g": 100,
    "serving_desc": "1 serving (est.)",
    "source": "default",
}

# ---------------------------------------------------------------------------
# Module-level state (lazy loaded)
# ---------------------------------------------------------------------------
_local_db: dict = {}
_api_cache: dict = {}          # persisted to _API_CACHE_PATH
_runtime_cache: dict = {}      # in-memory for this session
_lock = threading.Lock()
_db_loaded = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_local_db() -> None:
    global _local_db, _db_loaded
    if _db_loaded:
        return
    try:
        with open(_LOCAL_DB_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Strip meta / comment keys
        _local_db = {k: v for k, v in raw.items()
                     if not k.startswith("_") and isinstance(v, dict)}
    except Exception as e:
        print(f"[NutritionDB] Could not load local DB: {e}")
        _local_db = {}
    _db_loaded = True


def _load_api_cache() -> None:
    global _api_cache
    if os.path.exists(_API_CACHE_PATH):
        try:
            with open(_API_CACHE_PATH, "r", encoding="utf-8") as f:
                _api_cache = json.load(f)
        except Exception:
            _api_cache = {}


def _save_api_cache() -> None:
    try:
        with open(_API_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_api_cache, f, indent=2)
    except Exception as e:
        print(f"[NutritionDB] Could not save API cache: {e}")


def normalize(label: str) -> str:
    """
    Normalise a food label to the local-DB key format.
    'French Fries' / 'french fries' / 'french-fries' → 'french_fries'
    """
    return label.lower().strip().replace(" ", "_").replace("-", "_")


def format_label(label: str) -> str:
    """'french_fries' → 'French Fries' (title-case, spaces)."""
    return label.replace("_", " ").title()


def _query_usda(query: str) -> Optional[dict]:
    """
    Hit USDA FoodData Central API. Returns per-100g values.
    Returns None on any error (network, timeout, parse).
    """
    try:
        import requests
        params = {
            "query": query,
            "api_key": USDA_API_KEY,
            "pageSize": 5,
            "dataType": ["SR Legacy", "Foundation", "Survey (FNDDS)"],
        }
        resp = requests.get(USDA_API_URL, params=params, timeout=6)
        resp.raise_for_status()
        foods = resp.json().get("foods", [])
        if not foods:
            return None

        food = foods[0]
        nutrients: dict[str, float] = {}
        for n in food.get("foodNutrients", []):
            name = n.get("nutrientName", "")
            value = n.get("value", 0)
            nutrients[name] = float(value)

        calories = nutrients.get(
            "Energy",
            nutrients.get("Energy (Atwater General Factors)", 0.0),
        )
        protein = nutrients.get("Protein", 0.0)
        fat = nutrients.get("Total lipid (fat)", 0.0)
        carbs = nutrients.get("Carbohydrate, by difference", 0.0)

        return {
            "calories": round(calories, 1),
            "protein":  round(protein, 1),
            "fat":      round(fat, 1),
            "carbs":    round(carbs, 1),
            "serving_g": 100,
            "serving_desc": "per 100 g (USDA)",
            "source": "USDA",
        }
    except Exception as e:
        print(f"[NutritionDB] USDA query failed for '{query}': {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_nutrition(food_label: str) -> dict:
    """
    Return a nutrition dict for *food_label*.

    Keys: calories, protein, fat, carbs (all per 100 g),
          serving_g, serving_desc, source.

    Never raises; returns DEFAULT_NUTRITION as last resort.
    """
    with _lock:
        _load_local_db()

    key = normalize(food_label)

    # 1. Runtime cache
    if key in _runtime_cache:
        return _runtime_cache[key]

    # 2. Local JSON DB (exact match)
    if key in _local_db:
        result = dict(_local_db[key])
        result.setdefault("source", "local")
        _runtime_cache[key] = result
        return result

    # 3. Partial / fuzzy match inside local DB
    for db_key, db_val in _local_db.items():
        if key in db_key or db_key in key:
            result = dict(db_val)
            result.setdefault("source", "local (partial)")
            _runtime_cache[key] = result
            return result

    # 4. Persistent API cache (from a previous session)
    _load_api_cache()
    if key in _api_cache:
        _runtime_cache[key] = _api_cache[key]
        return _api_cache[key]

    # 5. Live USDA query
    query = food_label.replace("_", " ")
    result = _query_usda(query)
    if result:
        with _lock:
            _api_cache[key] = result
            _save_api_cache()
        _runtime_cache[key] = result
        return result

    # 6. Hard fallback
    print(f"[NutritionDB] No data found for '{food_label}', using defaults.")
    fallback = dict(DEFAULT_NUTRITION)
    _runtime_cache[key] = fallback
    return fallback


def get_serving_calories(food_label: str) -> float:
    """
    Return calories for the food's *typical serving size*.

    Formula: (calories_per_100g / 100) × serving_g
    """
    info = get_nutrition(food_label)
    return round(info["calories"] * info["serving_g"] / 100.0, 1)


def get_macro_summary(food_label: str) -> str:
    """
    One-line macro string for UI display.
    e.g. 'P 11g  F 10g  C 33g'
    """
    info = get_nutrition(food_label)
    serving_factor = info["serving_g"] / 100.0
    p = round(info["protein"] * serving_factor, 1)
    f = round(info["fat"]     * serving_factor, 1)
    c = round(info["carbs"]   * serving_factor, 1)
    return f"P {p}g  F {f}g  C {c}g"


# ---------------------------------------------------------------------------
# Background USDA prefetch (optional — call to warm up common items)
# ---------------------------------------------------------------------------

def prefetch_async(labels: list[str]) -> None:
    """Fire-and-forget background prefetch for a list of food labels."""
    def _fetch():
        for lbl in labels:
            get_nutrition(lbl)
    threading.Thread(target=_fetch, daemon=True).start()
