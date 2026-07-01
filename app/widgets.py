"""
app/widgets.py — Reusable Tkinter UI components (H5).

All custom widgets for the food calorie tracker:
  • FoodCard        — sidebar card with confidence badge + macros
  • StatusBar       — bottom bar with FPS, device, item count
  • MealTotals      — total calories + macros section
  • ProgressRing    — animated daily goal progress
"""

from __future__ import annotations

import tkinter as tk
from typing import Optional

from app.constants import (
    BG_CARD, BG_SIDEBAR, BORDER, TEXT_PRI, TEXT_SEC,
    SUCCESS, WARNING_C, DANGER, ACCENT, ACCENT_L,
    CONF_HIGH, CONF_MED, CONF_LOW,
)


def conf_color(confidence: float) -> str:
    """Return a colour string based on detection confidence."""
    if confidence >= 0.70:
        return CONF_HIGH
    elif confidence >= 0.45:
        return CONF_MED
    return CONF_LOW


def conf_label(confidence: float) -> str:
    """Return a short confidence label."""
    if confidence >= 0.70:
        return "HIGH"
    elif confidence >= 0.45:
        return "MED"
    return "LOW"


class FoodCard(tk.Frame):
    """
    A sidebar card displaying one detected food item.

    Layout:
      ┌──────────────────────────────────┐
      │ 🍽 Spring Rolls         87% HIGH │  Row 1: name + confidence badge
      │ ~1 serving (150g)               │  Row 2: estimated serving
      │ 300 kcal                        │  Row 3: calories for serving
      │ P 9g · F 14g · C 36g           │  Row 4: macros
      └──────────────────────────────────┘
    """

    def __init__(
        self,
        parent: tk.Widget,
        display: str,
        confidence: float,
        serving_desc: str,
        calories: float,
        protein: float,
        fat: float,
        carbs: float,
        **kwargs,
    ) -> None:
        super().__init__(parent, bg=BG_CARD, pady=8, padx=10, **kwargs)

        # Row 1: Food name + confidence badge
        row1 = tk.Frame(self, bg=BG_CARD)
        row1.pack(fill="x")

        tk.Label(
            row1, text=f"🍽  {display}",
            font=("Segoe UI", 11, "bold"),
            bg=BG_CARD, fg=TEXT_PRI,
        ).pack(side="left")

        badge_color = conf_color(confidence)
        badge_text = f"{confidence:.0%} {conf_label(confidence)}"
        tk.Label(
            row1, text=badge_text,
            font=("Segoe UI", 8, "bold"),
            bg=badge_color, fg="#000000",
            padx=6, pady=1,
        ).pack(side="right")

        # Row 2: Serving estimate
        tk.Label(
            self, text=serving_desc,
            font=("Segoe UI", 9),
            bg=BG_CARD, fg=TEXT_SEC,
            anchor="w",
        ).pack(fill="x", pady=(2, 0))

        # Row 3: Calories (large, coloured)
        tk.Label(
            self, text=f"{calories:.0f} kcal",
            font=("Segoe UI", 13, "bold"),
            bg=BG_CARD, fg=SUCCESS,
            anchor="w",
        ).pack(fill="x")

        # Row 4: Macros
        macro_text = f"P {protein:.0f}g · F {fat:.0f}g · C {carbs:.0f}g"
        tk.Label(
            self, text=macro_text,
            font=("Segoe UI", 9),
            bg=BG_CARD, fg=ACCENT_L,
            anchor="w",
        ).pack(fill="x")


class MealTotals(tk.Frame):
    """
    Meal totals section at the bottom of the sidebar.

    Shows total calories + summed macros for all detected items.
    """

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(parent, bg=BG_SIDEBAR, **kwargs)

        # Separator
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=4, pady=(0, 8))

        # Header
        tk.Label(
            self, text="MEAL TOTAL",
            font=("Segoe UI", 9, "bold"),
            bg=BG_SIDEBAR, fg=TEXT_SEC,
        ).pack(anchor="w")

        # Calories (big number)
        self.cal_label = tk.Label(
            self, text="0 kcal",
            font=("Segoe UI", 26, "bold"),
            bg=BG_SIDEBAR, fg=SUCCESS,
        )
        self.cal_label.pack(anchor="w")

        # Macros row
        self.macro_label = tk.Label(
            self, text="P 0g · F 0g · C 0g",
            font=("Segoe UI", 10),
            bg=BG_SIDEBAR, fg=ACCENT_L,
        )
        self.macro_label.pack(anchor="w")

    def update_totals(
        self,
        calories: float,
        protein: float,
        fat: float,
        carbs: float,
        daily_goal: int,
    ) -> None:
        """Update the displayed totals."""
        self.cal_label.config(text=f"{calories:.0f} kcal")

        # Colour based on daily goal progress
        if calories > daily_goal * 0.85:
            fg = DANGER
        elif calories > daily_goal * 0.60:
            fg = WARNING_C
        else:
            fg = SUCCESS
        self.cal_label.config(fg=fg)

        self.macro_label.config(
            text=f"P {protein:.0f}g · F {fat:.0f}g · C {carbs:.0f}g"
        )


class StatusBar(tk.Frame):
    """
    Bottom status bar with FPS, device indicator, status text, and item count.
    """

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(parent, bg="#0a0a18", height=28, **kwargs)

        # Left: status text
        self.status_label = tk.Label(
            self, text="Ready",
            font=("Segoe UI", 9),
            bg="#0a0a18", fg=TEXT_SEC, anchor="w",
        )
        self.status_label.pack(side="left", padx=12)

        # Right side: item count, FPS, device
        self.count_label = tk.Label(
            self, text="0 items",
            font=("Segoe UI", 9),
            bg="#0a0a18", fg=TEXT_SEC,
        )
        self.count_label.pack(side="right", padx=(0, 12))

        self.fps_label = tk.Label(
            self, text="-- FPS",
            font=("Segoe UI", 9),
            bg="#0a0a18", fg=TEXT_SEC,
        )
        self.fps_label.pack(side="right", padx=(0, 12))

        self.device_label = tk.Label(
            self, text="CPU",
            font=("Segoe UI", 9, "bold"),
            bg="#0a0a18", fg=ACCENT_L,
        )
        self.device_label.pack(side="right", padx=(0, 12))

    def set_status(self, text: str, fg: str = TEXT_SEC) -> None:
        self.status_label.config(text=text, fg=fg)

    def set_count(self, n: int) -> None:
        self.count_label.config(text=f"{n} item{'s' if n != 1 else ''}")

    def set_fps(self, fps: float) -> None:
        self.fps_label.config(text=f"{fps:.0f} FPS")

    def set_device(self, tag: str) -> None:
        self.device_label.config(text=tag)
