"""
app/detector.py — YOLO-based food detection engine.

Wraps the Ultralytics YOLO model with:
  • Optimised confidence / IoU thresholds for food detection (H1)
  • Temporal smoothing to eliminate single-frame flicker on webcam (H1)
  • IoU-based duplicate merging for overlapping boxes (H3)
  • FP16 inference when CUDA is available (M2)
  • Graceful error handling for corrupted / missing models (H6)
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from app.constants import (
    COCO_FOOD_CLASSES,
    CONF_THRESHOLD,
    CUSTOM_MODEL,
    COCO_FALLBACK,
    INFERENCE_IMGSZ,
    IOU_THRESHOLD,
    MAX_DETECTIONS,
    MIN_STABLE_COUNT,
    SMOOTHING_WINDOW,
)

logger = logging.getLogger("FoodTracker.Detector")


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single food detection from one frame."""
    label: str
    display: str            # "French Fries"
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    bbox_area_ratio: float  # (box area) / (frame area)

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass
class FrameResult:
    """All detections from a single frame, plus timing info."""
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    fps: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# FoodDetector
# ──────────────────────────────────────────────────────────────────────────────

class FoodDetector:
    """
    YOLO-based food detection with production-quality inference.

    Usage:
        detector = FoodDetector()
        result   = detector.detect(frame)      # returns FrameResult
        stable   = detector.get_stable_items()  # temporally smoothed
    """

    def __init__(self) -> None:
        self.model = None
        self.is_custom: bool = False
        self.device: str = "cpu"
        self.use_half: bool = False
        self._model_error: Optional[str] = None

        # Temporal smoothing state (H1)
        self._history: deque[list[Detection]] = deque(maxlen=SMOOTHING_WINDOW)

        # FPS tracking
        self._last_time: float = time.perf_counter()
        self._fps: float = 0.0

        # Load the model
        self._load_model()

    # ── Model loading with error handling (H6) ────────────────────────────────

    def _load_model(self) -> None:
        """Load YOLO model with graceful error handling."""
        try:
            from ultralytics import YOLO
        except ImportError:
            self._model_error = (
                "ultralytics package not installed.\n"
                "Run: pip install ultralytics"
            )
            logger.error(self._model_error)
            return

        # Detect compute device
        try:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
                self.use_half = True
                gpu_name = torch.cuda.get_device_name(0)
                logger.info("GPU detected: %s (FP16 enabled)", gpu_name)
            else:
                self.device = "cpu"
                self.use_half = False
                logger.info("No GPU — using CPU inference")
        except Exception:
            self.device = "cpu"
            self.use_half = False

        # Load model file
        model_path = CUSTOM_MODEL if CUSTOM_MODEL.exists() else COCO_FALLBACK
        self.is_custom = CUSTOM_MODEL.exists()

        try:
            self.model = YOLO(str(model_path))
            if self.is_custom:
                nc = len(self.model.names)
                logger.info("Loaded Food-101 model: %s (%d classes)", model_path, nc)
            else:
                logger.warning(
                    "models/best.pt not found — using COCO fallback (%s). "
                    "Run 'python train_model.py' to train a food model.",
                    COCO_FALLBACK,
                )
        except Exception as exc:
            self._model_error = f"Failed to load model '{model_path}': {exc}"
            logger.error(self._model_error)
            self.model = None

    @property
    def model_error(self) -> Optional[str]:
        """Non-None if the model failed to load."""
        return self._model_error

    @property
    def model_tag(self) -> str:
        """Human-readable model description for UI display."""
        if self._model_error:
            return "⚠ Model Error"
        if self.is_custom and self.model:
            nc = len(self.model.names)
            return f"Food-101 Model ({nc} classes)"
        return "COCO fallback (7 classes)"

    @property
    def device_tag(self) -> str:
        """Human-readable device description for UI."""
        if self.device == "cuda":
            try:
                import torch
                name = torch.cuda.get_device_name(0)
                # Shorten: "NVIDIA GeForce RTX 3060" → "RTX 3060"
                for prefix in ("NVIDIA GeForce ", "NVIDIA ", "AMD "):
                    if name.startswith(prefix):
                        name = name[len(prefix):]
                return f"GPU: {name}"
            except Exception:
                return "GPU"
        return "CPU"

    @property
    def fps(self) -> float:
        return self._fps

    # ── Core detection ────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, conf: Optional[float] = None) -> FrameResult:
        """
        Run YOLO inference on a single frame.

        Returns a FrameResult with all food detections and timing info.
        """
        if self.model is None:
            return FrameResult()

        h, w = frame.shape[:2]
        frame_area = h * w

        # Use passed conf, otherwise fall back to class default
        conf_val = conf if conf is not None else CONF_THRESHOLD

        # Measure inference time
        t0 = time.perf_counter()

        kwargs = {
            "conf": conf_val,
            "iou": IOU_THRESHOLD,
            "imgsz": INFERENCE_IMGSZ,
            "max_det": MAX_DETECTIONS,
            "verbose": False,
        }
        if self.use_half:
            kwargs["half"] = True

        try:
            results = self.model(frame, **kwargs)
        except RuntimeError as exc:
            # Handle CUDA OOM — fallback to CPU (H6)
            if "out of memory" in str(exc).lower() and self.device == "cuda":
                logger.warning("CUDA OOM — falling back to CPU inference")
                self.use_half = False
                self.device = "cpu"
                self.model.to("cpu")
                kwargs["half"] = False
                if "half" in kwargs:
                    del kwargs["half"]
                results = self.model(frame, **kwargs)
            else:
                logger.error("Inference error: %s", exc)
                return FrameResult()

        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000

        # FPS calculation (exponential moving average)
        dt = t1 - self._last_time
        self._last_time = t1
        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps = 0.7 * self._fps + 0.3 * instant_fps  # EMA smoothing

        # Parse detections
        detections: list[Detection] = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label  = self.model.names[cls_id]
                conf   = float(box.conf[0])

                if not self._is_food(label):
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                box_area = (x2 - x1) * (y2 - y1)
                area_ratio = box_area / frame_area if frame_area > 0 else 0.0

                detections.append(Detection(
                    label=label,
                    display=label.replace("_", " ").title(),
                    confidence=conf,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    bbox_area_ratio=area_ratio,
                ))

        # Merge overlapping same-class detections (H3)
        detections = self._merge_overlapping(detections)

        result = FrameResult(
            detections=detections,
            inference_ms=inference_ms,
            fps=self._fps,
        )

        # Update temporal history for smoothing
        self._history.append(detections)

        return result

    def _is_food(self, label: str) -> bool:
        """Check if a detection label is a food item."""
        if self.is_custom:
            return True  # custom model only knows food classes
        return label in COCO_FOOD_CLASSES

    # ── Duplicate merging (H3) ────────────────────────────────────────────────

    @staticmethod
    def _merge_overlapping(detections: list[Detection]) -> list[Detection]:
        """
        Merge overlapping boxes of the same class.

        When two boxes for the same food class overlap with IoU > 0.6,
        keep only the higher-confidence one.  Different spatial regions
        of the same class are kept as separate items (e.g. two pizzas).
        """
        if len(detections) <= 1:
            return detections

        MERGE_IOU = 0.6
        merged: list[Detection] = []
        used = [False] * len(detections)

        for i, det_i in enumerate(detections):
            if used[i]:
                continue
            best = det_i
            used[i] = True

            for j in range(i + 1, len(detections)):
                if used[j]:
                    continue
                det_j = detections[j]
                if det_i.label != det_j.label:
                    continue

                iou = FoodDetector._compute_iou(det_i, det_j)
                if iou > MERGE_IOU:
                    used[j] = True
                    if det_j.confidence > best.confidence:
                        best = det_j

            merged.append(best)

        return merged

    @staticmethod
    def _compute_iou(a: Detection, b: Detection) -> float:
        """Compute intersection-over-union between two detections."""
        inter_x1 = max(a.x1, b.x1)
        inter_y1 = max(a.y1, b.y1)
        inter_x2 = min(a.x2, b.x2)
        inter_y2 = min(a.y2, b.y2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union = area_a + area_b - inter_area

        return inter_area / union if union > 0 else 0.0

    # ── Temporal smoothing (H1) ───────────────────────────────────────────────

    def get_stable_labels(self) -> set[str]:
        """
        Return the set of food labels that have appeared in at least
        MIN_STABLE_COUNT of the last SMOOTHING_WINDOW frames.

        This eliminates single-frame false positives on webcam.
        """
        if len(self._history) < 2:
            # Not enough history yet — accept everything
            if self._history:
                return {d.label for d in self._history[-1]}
            return set()

        counts: dict[str, int] = {}
        for frame_dets in self._history:
            seen = {d.label for d in frame_dets}
            for label in seen:
                counts[label] = counts.get(label, 0) + 1

        return {
            label for label, count in counts.items()
            if count >= MIN_STABLE_COUNT
        }

    def clear_history(self) -> None:
        """Reset temporal smoothing history."""
        self._history.clear()

    # ── Model reload (for post-training) ──────────────────────────────────────

    def reload_model(self) -> None:
        """Reload the model from disk (called after training completes)."""
        self._history.clear()
        self._model_error = None
        self._load_model()
