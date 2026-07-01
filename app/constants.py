"""
app/constants.py — Centralized configuration and constants.

All colours, paths, thresholds, and tuning knobs in one place.
Import from here — never hardcode values elsewhere.
"""

from __future__ import annotations

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# File paths
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent.parent          # food-calorie-detector/
CUSTOM_MODEL  = SCRIPT_DIR / "models" / "best.pt"
COCO_FALLBACK = "yolov8n.pt"

# ──────────────────────────────────────────────────────────────────────────────
# Detection thresholds (H1: tuned for 15-class food model)
# ──────────────────────────────────────────────────────────────────────────────
# Confidence: 0.45 ↑ from 0.35 — with only 15 classes the model has fewer
# confusion categories, so it can afford a higher bar.  Reduces false positives
# without hurting recall on clear food images.
CONF_THRESHOLD     = 0.45
# IoU for Non-Maximum Suppression — merge boxes overlapping >50 %
IOU_THRESHOLD      = 0.50
# Maximum detections per frame (prevents runaway box count)
MAX_DETECTIONS     = 20
# YOLO inference image size — should match training imgsz for best results
INFERENCE_IMGSZ    = 416

# ──────────────────────────────────────────────────────────────────────────────
# Temporal smoothing for webcam (H1: reduces flicker)
# ──────────────────────────────────────────────────────────────────────────────
# A food label must appear in ≥ MIN_STABLE_COUNT of the last SMOOTHING_WINDOW
# frames to be accepted into the sidebar.  Eliminates single-frame false alarms.
SMOOTHING_WINDOW   = 5       # frames
MIN_STABLE_COUNT   = 3       # required appearances in the window

# ──────────────────────────────────────────────────────────────────────────────
# Detection loop timing
# ──────────────────────────────────────────────────────────────────────────────
# 33 ms ≈ 30 FPS ceiling.  YOLO inference itself takes 50-200 ms so the real
# frame rate is bounded by the model.
LOOP_INTERVAL_MS   = 33

# ──────────────────────────────────────────────────────────────────────────────
# COCO fallback food classes (when models/best.pt is absent)
# ──────────────────────────────────────────────────────────────────────────────
COCO_FOOD_CLASSES: set[str] = {
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
}

# ──────────────────────────────────────────────────────────────────────────────
# Serving-size estimation buckets (H2)
# ──────────────────────────────────────────────────────────────────────────────
# bbox_area_ratio = (box_w × box_h) / (frame_w × frame_h)
SERVING_BUCKETS: list[tuple[float, float]] = [
    # (max_ratio, multiplier)
    (0.05,  0.5),    # tiny detection   → half a serving
    (0.15,  1.0),    # medium detection → standard serving
    (0.30,  1.5),    # large detection  → 1.5× serving
    (1.00,  2.0),    # very large       → double serving
]

# ──────────────────────────────────────────────────────────────────────────────
# UI colour palette
# ──────────────────────────────────────────────────────────────────────────────
BG_DARK    = "#0d0d1a"
BG_CARD    = "#1a1a2e"
BG_SIDEBAR = "#12122a"
ACCENT     = "#7c4dff"
ACCENT_L   = "#b39ddb"
SUCCESS    = "#00e5a0"
WARNING_C  = "#ff9800"
DANGER     = "#ff5252"
TEXT_PRI   = "#f0f0ff"
TEXT_SEC   = "#8888aa"
BORDER     = "#2a2a4a"

# Confidence badge colours
CONF_HIGH    = SUCCESS       # ≥ 70 %
CONF_MED     = WARNING_C     # 45 – 70 %
CONF_LOW     = DANGER        # < 45 %

# OpenCV bounding-box colour (BGR green)
BOX_BGR    = (120, 220, 100)

# ──────────────────────────────────────────────────────────────────────────────
# Calorie & nutrition
# ──────────────────────────────────────────────────────────────────────────────
DAILY_GOAL = 2000           # kcal daily target

# Built-in calorie reference — kcal per 100 g — all 101 Food-101 classes
# Kept here as a fast fallback when nutrition_db / nutrition_data.json are missing.
FOOD_CALORIES_PER_100G: dict[str, int] = {
    "apple_pie": 237,           "baby_back_ribs": 295,      "baklava": 428,
    "beef_carpaccio": 157,      "beef_tartare": 193,         "beet_salad": 45,
    "beignets": 392,            "bibimbap": 131,             "bread_pudding": 225,
    "breakfast_burrito": 197,   "bruschetta": 185,           "caesar_salad": 145,
    "cannoli": 368,             "caprese_salad": 94,         "carrot_cake": 340,
    "ceviche": 63,              "cheese_plate": 350,         "cheesecake": 321,
    "chicken_curry": 150,       "chicken_quesadilla": 215,   "chicken_wings": 290,
    "chocolate_cake": 371,      "chocolate_mousse": 244,     "churros": 364,
    "clam_chowder": 78,         "club_sandwich": 228,        "crab_cakes": 187,
    "creme_brulee": 224,        "croque_madame": 267,        "cup_cakes": 305,
    "deviled_eggs": 174,        "donuts": 452,               "dumplings": 232,
    "edamame": 122,             "eggs_benedict": 182,        "escargots": 90,
    "falafel": 333,             "filet_mignon": 267,         "fish_and_chips": 280,
    "foie_gras": 462,           "french_fries": 312,         "french_onion_soup": 55,
    "french_toast": 229,        "fried_calamari": 175,       "fried_rice": 163,
    "frozen_yogurt": 127,       "garlic_bread": 287,         "gnocchi": 131,
    "greek_salad": 74,          "grilled_cheese_sandwich": 350, "grilled_salmon": 208,
    "guacamole": 155,           "gyoza": 213,                "hamburger": 295,
    "hot_and_sour_soup": 40,    "hot_dog": 290,              "huevos_rancheros": 150,
    "hummus": 177,              "ice_cream": 207,            "lasagna": 135,
    "lobster_bisque": 82,       "lobster_roll_sandwich": 261,"macaroni_and_cheese": 160,
    "macarons": 392,            "miso_soup": 40,             "mussels": 86,
    "nachos": 322,              "omelette": 154,             "onion_rings": 411,
    "oysters": 69,              "pad_thai": 164,             "paella": 150,
    "pancakes": 227,            "panna_cotta": 140,          "peking_duck": 337,
    "pho": 58,                  "pizza": 266,                "pork_chop": 231,
    "poutine": 260,             "prime_rib": 307,            "pulled_pork_sandwich": 277,
    "ramen": 61,                "ravioli": 209,              "red_velvet_cake": 363,
    "risotto": 150,             "samosa": 262,               "sashimi": 145,
    "scallops": 88,             "seaweed_salad": 45,         "shrimp_and_grits": 160,
    "spaghetti_bolognese": 137, "spaghetti_carbonara": 195,  "spring_rolls": 200,
    "steak": 271,               "strawberry_shortcake": 238, "sushi": 143,
    "tacos": 218,               "takoyaki": 179,             "tiramisu": 240,
    "tuna_tartare": 130,        "waffles": 291,
    # ── COCO fallback labels ──
    "banana": 89,   "apple": 52,   "orange": 47,  "broccoli": 34,
    "carrot": 41,   "hot dog": 290, "donut": 452,  "cake": 360,
    "sandwich": 250,
}
