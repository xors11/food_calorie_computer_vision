"""
classifier.py — Food-101 fine-grained image classifier.

Model: nateraw/food (ViT-base-patch16-224 fine-tuned on Food-101)
Source: https://huggingface.co/nateraw/food
Classes: 101 Food-101 categories

First call to classify() or load() will download the model weights (~350 MB)
and cache them in ~/.cache/huggingface/hub/  (automatic, standard HuggingFace
caching — happens only once).

Design:
  - Lazy-loaded: model is NOT imported at module import time.
  - Thread-safe: internal lock prevents double-loading.
  - Non-blocking startup: call load() in a background Thread so the Tkinter
    UI is responsive while the model downloads / loads.
  - Graceful degradation: if the model is not yet loaded (or fails), classify()
    returns ("unknown", 0.0) instead of raising.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_pipe = None                          # transformers image-classification pipeline
_load_lock = threading.Lock()
_load_error: Optional[str] = None    # set if loading fails permanently
_loading = False                      # True while download / init is in progress

# Confidence threshold: below this value we consider the prediction unreliable
# and the caller should fall back to the YOLO label.
CONFIDENCE_THRESHOLD: float = 0.25

# ---------------------------------------------------------------------------
# Internal loading
# ---------------------------------------------------------------------------

def load() -> None:
    """
    Load (or download) the Food-101 classifier pipeline.
    Idempotent and thread-safe — safe to call multiple times.
    Intended to be called in a background Thread.
    """
    global _pipe, _load_error, _loading

    if _pipe is not None or _load_error is not None:
        return  # already loaded or permanently failed

    with _load_lock:
        if _pipe is not None or _load_error is not None:
            return  # double-check inside lock

        _loading = True
        try:
            from transformers import pipeline as hf_pipeline  # noqa: PLC0415
            print("[Classifier] Loading Food-101 model (nateraw/food)…")
            print("[Classifier] First run downloads ~350 MB — please wait.")
            _pipe = hf_pipeline(
                "image-classification",
                model="nateraw/food",
                top_k=3,
            )
            print("[Classifier] ✓ Food-101 model ready.")
        except Exception as exc:
            _load_error = str(exc)
            print(f"[Classifier] ✗ Failed to load model: {exc}")
        finally:
            _loading = False


def load_async() -> threading.Thread:
    """
    Start loading the model in a daemon background thread.
    Returns the Thread object (already started).
    """
    t = threading.Thread(target=load, daemon=True, name="classifier-loader")
    t.start()
    return t


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_ready() -> bool:
    """Return True if the model is loaded and ready."""
    return _pipe is not None


def is_loading() -> bool:
    """Return True while the model is being downloaded/initialized."""
    return _loading


def status() -> str:
    """Human-readable status string for UI display."""
    if _pipe is not None:
        return "✓ AI classifier ready (Food-101)"
    if _loading:
        return "⏳ Loading AI classifier…"
    if _load_error:
        return f"⚠ Classifier unavailable: {_load_error[:60]}"
    return "⏳ AI classifier not started"


def classify(image) -> tuple[str, float]:
    """
    Classify a single food crop.

    Args:
        image: One of
            - PIL.Image.Image  (any mode; will be converted to RGB)
            - numpy.ndarray    (H×W×3, BGR uint8 — OpenCV format)

    Returns:
        (label, confidence)
            label:      Food-101 class name with underscores, e.g. 'french_fries'
                        'unknown' if model is not ready or crop is too small.
            confidence: float in [0.0, 1.0]

    Never raises.
    """
    if _pipe is None:
        return "unknown", 0.0

    try:
        from PIL import Image as PILImage  # noqa: PLC0415

        # ---- Convert input to RGB PIL Image ----
        if isinstance(image, np.ndarray):
            import cv2  # noqa: PLC0415
            if image.size == 0 or image.shape[0] < 10 or image.shape[1] < 10:
                return "unknown", 0.0
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)
        elif hasattr(image, "convert"):          # PIL Image
            pil_img = image.convert("RGB")
        else:
            return "unknown", 0.0

        # Reject tiny crops (e.g. partially off-screen boxes)
        if pil_img.width < 20 or pil_img.height < 20:
            return "unknown", 0.0

        results = _pipe(pil_img)               # top_k=3 list
        if not results:
            return "unknown", 0.0

        top = results[0]
        raw_label: str = top["label"]
        confidence: float = float(top["score"])

        # nateraw/food returns labels like "french fries" (with spaces).
        # Normalise to underscore format used by nutrition_db.
        label = raw_label.strip().lower().replace(" ", "_").replace("-", "_")

        return label, confidence

    except Exception as exc:
        print(f"[Classifier] classify() error: {exc}")
        return "unknown", 0.0


def classify_top3(image) -> list[tuple[str, float]]:
    """
    Return the top-3 predictions as a list of (label, confidence) tuples.
    Returns an empty list if the model is not ready or on any error.
    """
    if _pipe is None:
        return []

    try:
        from PIL import Image as PILImage  # noqa: PLC0415

        if isinstance(image, np.ndarray):
            import cv2  # noqa: PLC0415
            if image.size == 0:
                return []
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(rgb)
        else:
            pil_img = image.convert("RGB")

        results = _pipe(pil_img)
        return [
            (r["label"].strip().lower().replace(" ", "_"), float(r["score"]))
            for r in results
        ]
    except Exception as exc:
        print(f"[Classifier] classify_top3() error: {exc}")
        return []
