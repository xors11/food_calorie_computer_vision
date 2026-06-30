#!/usr/bin/env python3
"""
main.py — AI Food Calorie Tracker
===================================

Two modes:
  CLI  →  python main.py <image_path>     detect food in a static image
  GUI  →  python main.py                  live webcam + Tkinter window

Model priority (automatic):
  1. models/best.pt    — Food-101 fine-tuned YOLOv8n (101 classes)
                         Train with:  python train_model.py
  2. yolov8n.pt        — COCO pretrained fallback (~7 food classes)

No HuggingFace / transformers dependency.
Nutrition data: built-in dict (all 101 Food-101 classes, kcal per 100 g)
                + optional nutrition_db.py for extended info (macros, serving size).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Built-in calorie reference — kcal per 100 g — all 101 Food-101 classes
# ──────────────────────────────────────────────────────────────────────────────
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
    # ── COCO fallback labels (active when models/best.pt is not yet trained) ──
    "banana": 89,   "apple": 52,   "orange": 47,  "broccoli": 34,
    "carrot": 41,   "hot dog": 290, "donut": 452,  "cake": 360,
    "sandwich": 250,
}

# ──────────────────────────────────────────────────────────────────────────────
# Model paths & COCO fallback food classes
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = Path(__file__).parent
CUSTOM_MODEL  = SCRIPT_DIR / "models" / "best.pt"
COCO_FALLBACK = "yolov8n.pt"

# COCO class names treated as food when using the pretrained fallback model
COCO_FOOD_CLASSES: set[str] = {
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
}

# ── Colour palette ─────────────────────────────────────────────────────────────
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
BOX_BGR    = (120, 220, 100)    # OpenCV bounding-box colour (BGR green)
DAILY_GOAL = 2000               # kcal daily target


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_calories(label: str) -> int:
    """
    Return kcal per 100 g for a food label.
    Prefers nutrition_db.py (if present); falls back to built-in dict.
    """
    try:
        import nutrition_db                             # noqa: PLC0415
        info = nutrition_db.get_nutrition(label)
        cal  = info.get("calories", 0)
        if cal > 0:
            return int(cal)
    except Exception:
        pass
    return FOOD_CALORIES_PER_100G.get(label, 200)


def load_model():
    """
    Load YOLO.
    Returns (model, is_custom_food101_model).
    """
    from ultralytics import YOLO                        # noqa: PLC0415
    if CUSTOM_MODEL.exists():
        print(f"[Model] ✓ Loaded Food-101 model: {CUSTOM_MODEL}")
        return YOLO(str(CUSTOM_MODEL)), True
    else:
        print(f"[Model] ℹ  models/best.pt not found — using COCO fallback ({COCO_FALLBACK}).")
        print("[Model]    Run  python train_model.py  to enable 101-class detection.")
        return YOLO(COCO_FALLBACK), False


def is_food(label: str, is_custom: bool) -> bool:
    """True if this YOLO detection label should be treated as food."""
    if is_custom:
        return True                          # custom model only knows food classes
    return label in COCO_FOOD_CLASSES or label in FOOD_CALORIES_PER_100G


def fmt(label: str) -> str:
    """'french_fries' → 'French Fries'"""
    return label.replace("_", " ").title()


# ──────────────────────────────────────────────────────────────────────────────
# CLI mode
# ──────────────────────────────────────────────────────────────────────────────

def run_cli(image_path: str) -> None:
    """
    Analyse a single image file.
    Prints a nutrition table and saves an annotated copy of the image.
    """
    import cv2                                          # noqa: PLC0415

    img_p = Path(image_path)
    if not img_p.exists():
        print(f"ERROR: File not found — {image_path}")
        sys.exit(1)

    frame = cv2.imread(str(img_p))
    if frame is None:
        print(f"ERROR: Could not read image — {image_path}")
        sys.exit(1)

    print(f"\nAnalysing: {img_p.name}  ({frame.shape[1]}×{frame.shape[0]})")
    model, is_custom = load_model()

    results = model(frame, conf=0.25, verbose=False)

    # Deduplicate: keep highest-confidence detection per label
    best: dict[str, float] = {}
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label  = model.names[cls_id]
            conf   = float(box.conf[0])
            if is_food(label, is_custom):
                if label not in best or conf > best[label]:
                    best[label] = conf

    # ── Print nutrition table ──────────────────────────────────────────────────
    print()
    print("─" * 58)
    print(f"  {'Food Item':<30} {'kcal/100g':>9}  {'Conf':>7}")
    print("─" * 58)

    total = 0
    if best:
        for label, conf in sorted(best.items(), key=lambda x: -x[1]):
            cal = get_calories(label)
            print(f"  {fmt(label):<30} {cal:>9}   {conf:>6.1%}")
            total += cal
    else:
        print("  No food detected in this image.")

    print("─" * 58)
    print(f"  {'Estimated total (per 100g each)':<30} {total:>9} kcal")
    print("─" * 58)

    # ── Save annotated image ───────────────────────────────────────────────────
    annotated = results[0].plot()
    out_path  = img_p.parent / f"{img_p.stem}_detected{img_p.suffix}"
    cv2.imwrite(str(out_path), annotated)
    print(f"\n  Annotated image → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# GUI mode — Tkinter + live webcam
# ──────────────────────────────────────────────────────────────────────────────

class FoodCalorieTracker:
    """
    Real-time food calorie tracker with Tkinter UI.

    Preserved public method names: setup_ui(), reset(), update_loop()
    Removed: HuggingFace / transformers dependency (caused the torchaudio crash).
    """

    def __init__(self, root) -> None:
        import tkinter as tk                            # noqa: PLC0415
        import cv2 as _cv2                             # noqa: PLC0415

        self.root   = root
        self.cv2    = _cv2
        self.root.title("AI Food Calorie Tracker")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1280x720")
        self.root.minsize(900, 600)

        self.model, self.is_custom = load_model()

        # ── Session/Training state ────────────────────────────────────────────
        self.total_calories: float = 0.0
        # detected_items: label → {"display": str, "calories": int}
        self.detected_items: dict[str, dict] = {}
        self.static_frame  = None          # set by upload_image()
        
        import queue
        self.train_queue = queue.Queue()
        self.train_proc = None
        self.is_training_cancelled = False

        self.setup_ui()

        self.cap = _cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.config(
                text="⚠ Webcam not found. Use Upload Image to analyse a file.",
                fg=WARNING_C,
            )
        self.update_loop()

    # ── UI construction (preserved method name) ───────────────────────────────

    def setup_ui(self) -> None:
        import tkinter as tk                            # noqa: PLC0415
        from tkinter import ttk                        # noqa: PLC0415

        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        # ── Left panel: video ──────────────────────────────────────────────────
        left = tk.Frame(self.root, bg=BG_DARK)
        left.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=(12, 6))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        if self.is_custom:
            nc = len(self.model.names)
            model_tag = f"Food-101 Model ({nc} classes)"
            fg_color = SUCCESS
        else:
            model_tag = "COCO fallback (7 classes)"
            fg_color = ACCENT_L

        self.title_label = tk.Label(
            left,
            text=f"🍽  AI Food Calorie Tracker   [{model_tag}]",
            font=("Segoe UI", 15, "bold"),
            bg=BG_DARK, fg=fg_color,
        )
        self.title_label.grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.video_label = tk.Label(left, bg="#000010", relief="flat")
        self.video_label.grid(row=1, column=0, sticky="nsew")

        # ── Right panel: sidebar ───────────────────────────────────────────────
        right = tk.Frame(self.root, bg=BG_SIDEBAR)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 12), pady=(12, 6))
        right.columnconfigure(0, weight=1)

        tk.Label(
            right, text="Detected Items",
            font=("Segoe UI", 13, "bold"),
            bg=BG_SIDEBAR, fg=TEXT_PRI,
        ).pack(anchor="w", padx=14, pady=(14, 4))
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", padx=10)

        # Scrollable item list
        list_f = tk.Frame(right, bg=BG_SIDEBAR)
        list_f.pack(fill="both", expand=True, padx=6, pady=6)

        self.item_canvas = tk.Canvas(
            list_f, bg=BG_SIDEBAR, highlightthickness=0, width=240,
        )
        sb = ttk.Scrollbar(list_f, orient="vertical",
                           command=self.item_canvas.yview)
        self.item_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.item_canvas.pack(side="left", fill="both", expand=True)

        self.items_inner = tk.Frame(self.item_canvas, bg=BG_SIDEBAR)
        self._cwin = self.item_canvas.create_window(
            (0, 0), window=self.items_inner, anchor="nw",
        )
        self.items_inner.bind(
            "<Configure>",
            lambda e: self.item_canvas.configure(
                scrollregion=self.item_canvas.bbox("all")
            ),
        )
        self.item_canvas.bind(
            "<Configure>",
            lambda e: self.item_canvas.itemconfig(self._cwin, width=e.width),
        )

        # Total calories display
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", padx=10)
        tot_f = tk.Frame(right, bg=BG_SIDEBAR)
        tot_f.pack(fill="x", padx=14, pady=(10, 2))
        tk.Label(tot_f, text="TOTAL",
                 font=("Segoe UI", 9, "bold"),
                 bg=BG_SIDEBAR, fg=TEXT_SEC).pack(anchor="w")
        self.total_label = tk.Label(
            tot_f, text="0 kcal",
            font=("Segoe UI", 26, "bold"),
            bg=BG_SIDEBAR, fg=SUCCESS,
        )
        self.total_label.pack(anchor="w")

        # Goal + progress bar
        tk.Label(right, text=f"Daily Goal: {DAILY_GOAL} kcal",
                 font=("Segoe UI", 9), bg=BG_SIDEBAR, fg=TEXT_SEC,
                 ).pack(anchor="w", padx=14)
        prog_outer = tk.Frame(right, bg=BORDER, height=10)
        prog_outer.pack(fill="x", padx=14, pady=(3, 12))
        self.prog_bar = tk.Frame(prog_outer, bg=ACCENT, height=10)
        self.prog_bar.place(x=0, y=0, relheight=1.0, relwidth=0.0)

        # Buttons
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", padx=10)
        btn_f = tk.Frame(right, bg=BG_SIDEBAR)
        btn_f.pack(fill="x", padx=14, pady=12)

        row_btn_f = tk.Frame(btn_f, bg=BG_SIDEBAR)
        row_btn_f.pack(fill="x", pady=(0, 6))

        self.upload_btn = self._btn(row_btn_f, "📁  Upload Image", self.upload_image, ACCENT)
        self.upload_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))

        self.train_btn = self._btn(row_btn_f, "⚙️  Train Model", self.open_train_popup, "#d81b60")
        self.train_btn.pack(side="right", fill="x", expand=True, padx=(3, 0))

        self.reset_btn = self._btn(btn_f, "🔄  Reset", self.reset, "#3a3a5a")
        self.reset_btn.pack(fill="x")

        # Status bar
        sbar = tk.Frame(self.root, bg="#0a0a18", height=28)
        sbar.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.status_label = tk.Label(
            sbar, text="Ready",
            font=("Segoe UI", 9),
            bg="#0a0a18", fg=TEXT_SEC, anchor="w",
        )
        self.status_label.pack(side="left", padx=12)
        self.count_label = tk.Label(
            sbar, text="0 items",
            font=("Segoe UI", 9),
            bg="#0a0a18", fg=TEXT_SEC,
        )
        self.count_label.pack(side="right", padx=12)

    def _btn(self, parent, text, cmd, bg_color):
        import tkinter as tk                            # noqa: PLC0415
        b = tk.Button(
            parent, text=text, command=cmd,
            font=("Segoe UI", 10, "bold"),
            bg=bg_color, fg=TEXT_PRI,
            activebackground=ACCENT, activeforeground="#ffffff",
            relief="flat", bd=0, padx=10, pady=8, cursor="hand2",
        )
        b.bind("<Enter>", lambda e: b.configure(bg=ACCENT))
        b.bind("<Leave>", lambda e: b.configure(bg=bg_color))
        return b

    # ── Reset (preserved method name) ────────────────────────────────────────

    def reset(self) -> None:
        """Clear all detected items and calorie total."""
        self.total_calories = 0.0
        self.detected_items.clear()
        self.static_frame = None
        self._refresh_sidebar()
        self.total_label.config(text="0 kcal", fg=SUCCESS)
        self.prog_bar.place(relwidth=0.0)
        self.count_label.config(text="0 items")
        self.status_label.config(text="Reset.", fg=TEXT_SEC)

    # ── Upload image ──────────────────────────────────────────────────────────

    def upload_image(self) -> None:
        from tkinter import filedialog                  # noqa: PLC0415
        path = filedialog.askopenfilename(
            title="Select food image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if path:
            frame = self.cv2.imread(path)
            if frame is not None:
                self.static_frame = frame
                self.status_label.config(
                    text=f"Image loaded: {Path(path).name}", fg=TEXT_SEC
                )
            else:
                self.status_label.config(
                    text="⚠ Could not read image file.", fg=DANGER
                )

    # ── Main loop (preserved method name) ─────────────────────────────────────

    def update_loop(self) -> None:
        from PIL import Image, ImageTk                  # noqa: PLC0415

        frame = None
        if self.static_frame is not None:
            frame = self.static_frame.copy()
        elif self.cap is not None and self.cap.isOpened():
            ret, cam = self.cap.read()
            if ret:
                frame = cam

        if frame is not None:
            results = self.model(frame, conf=0.35, verbose=False)
            frame   = self._process_detections(frame, results)

            # Convert + display
            rgb   = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            pil   = Image.fromarray(rgb)
            lw    = max(self.video_label.winfo_width(),  640)
            lh    = max(self.video_label.winfo_height(), 480)
            pil.thumbnail((lw, lh), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=pil)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            n = len(self.detected_items)
            self.count_label.config(text=f"{n} item{'s' if n != 1 else ''}")

        self.root.after(15, self.update_loop)

    # ── Detection logic ───────────────────────────────────────────────────────

    def _process_detections(self, frame, results):
        """Annotate frame and update calorie tracking."""
        sidebar_dirty = False
        h, w = frame.shape[:2]

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label  = self.model.names[cls_id]
                conf   = float(box.conf[0])

                if not is_food(label, self.is_custom):
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                display = fmt(label)

                # Draw bounding box + label tag
                self.cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_BGR, 2)
                tag = f"{display}  {conf:.0%}"
                (tw, th), _ = self.cv2.getTextSize(
                    tag, self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )
                self.cv2.rectangle(
                    frame,
                    (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1),
                    BOX_BGR, -1,
                )
                self.cv2.putText(
                    frame, tag,
                    (x1 + 4, max(th, y1 - 4)),
                    self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1,
                    self.cv2.LINE_AA,
                )

                # Track unique items
                if label not in self.detected_items:
                    cal = get_calories(label)
                    self.detected_items[label] = {
                        "display":  display,
                        "calories": cal,
                    }
                    self.total_calories += cal
                    sidebar_dirty = True

        if sidebar_dirty:
            self._refresh_sidebar()
            self._refresh_totals()
            print("\n--- Detected Ingredients ---")
            for lbl, info in self.detected_items.items():
                print(f"  - {info['display']}: {info['calories']} kcal/100g")
            print(f"Total Estimated Calories (per 100g each): {self.total_calories:.1f} kcal")
            print("----------------------------")

        return frame

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _refresh_sidebar(self) -> None:
        import tkinter as tk                            # noqa: PLC0415
        for widget in self.items_inner.winfo_children():
            widget.destroy()

        if not self.detected_items:
            tk.Label(
                self.items_inner,
                text="Point camera at food…",
                font=("Segoe UI", 10, "italic"),
                bg=BG_SIDEBAR, fg=TEXT_SEC,
            ).pack(padx=10, pady=20)
            return

        for label, info in self.detected_items.items():
            card = tk.Frame(self.items_inner, bg=BG_CARD, pady=6)
            card.pack(fill="x", padx=6, pady=3)
            row = tk.Frame(card, bg=BG_CARD)
            row.pack(fill="x", padx=10)
            tk.Label(row, text=info["display"],
                     font=("Segoe UI", 11, "bold"),
                     bg=BG_CARD, fg=TEXT_PRI).pack(side="left")
            tk.Label(row, text=f"{info['calories']} kcal/100g",
                     font=("Segoe UI", 10),
                     bg=BG_CARD, fg=SUCCESS).pack(side="right")

    def _refresh_totals(self) -> None:
        self.total_label.config(text=f"{self.total_calories:.0f} kcal")
        fg = (DANGER    if self.total_calories > DAILY_GOAL * 0.85 else
              WARNING_C if self.total_calories > DAILY_GOAL * 0.60 else SUCCESS)
        self.total_label.config(fg=fg)
        frac      = min(1.0, self.total_calories / DAILY_GOAL)
        bar_color = (DANGER    if frac > 0.85 else
                     WARNING_C if frac > 0.60 else ACCENT)
        self.prog_bar.place(relwidth=frac)
        self.prog_bar.config(bg=bar_color)


    def open_train_popup(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        if hasattr(self, "train_popup") and self.train_popup.winfo_exists():
            self.train_popup.lift()
            return

        self.train_btn.config(state="disabled")
        self.is_training_cancelled = False

        self.train_popup = tk.Toplevel(self.root)
        self.train_popup.title("Training YOLOv8n Model")
        self.train_popup.configure(bg=BG_SIDEBAR)
        self.train_popup.transient(self.root)
        self.train_popup.grab_set()

        # Size and center popup
        w_width, w_height = 600, 450
        x = self.root.winfo_x() + (self.root.winfo_width() - w_width) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - w_height) // 2
        self.train_popup.geometry(f"{w_width}x{w_height}+{x}+{y}")

        # Title Label
        tk.Label(
            self.train_popup, text="YOLOv8n Food-101 Training",
            font=("Segoe UI", 14, "bold"),
            bg=BG_SIDEBAR, fg=TEXT_PRI
        ).pack(anchor="w", padx=20, pady=(20, 10))

        # Status frame (Epoch / Loss)
        status_f = tk.Frame(self.train_popup, bg=BG_SIDEBAR)
        status_f.pack(fill="x", padx=20, pady=5)

        self.epoch_var = tk.StringVar(value="Epoch: 0/20")
        tk.Label(
            status_f, textvariable=self.epoch_var,
            font=("Segoe UI", 11, "bold"), bg=BG_SIDEBAR, fg=ACCENT_L
        ).pack(side="left")

        self.loss_var = tk.StringVar(value="Loss: N/A")
        tk.Label(
            status_f, textvariable=self.loss_var,
            font=("Segoe UI", 11, "bold"), bg=BG_SIDEBAR, fg=SUCCESS
        ).pack(side="right")

        # Progress bar
        self.train_progress = ttk.Progressbar(
            self.train_popup, orient="horizontal",
            length=560, mode="determinate", maximum=20
        )
        self.train_progress.pack(fill="x", padx=20, pady=10)

        # Log Text area
        log_f = tk.Frame(self.train_popup, bg=BG_DARK, bd=1, relief="solid")
        log_f.pack(fill="both", expand=True, padx=20, pady=10)

        self.log_text = tk.Text(
            log_f, bg=BG_DARK, fg=TEXT_PRI,
            font=("Consolas", 9), wrap="word", state="disabled",
            highlightthickness=0, bd=0
        )
        log_scroll = ttk.Scrollbar(log_f, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

        # Cancel button
        btn_f = tk.Frame(self.train_popup, bg=BG_SIDEBAR)
        btn_f.pack(fill="x", padx=20, pady=(10, 20))

        self.cancel_btn = tk.Button(
            btn_f, text="Cancel Training",
            command=self.cancel_training,
            font=("Segoe UI", 10, "bold"),
            bg=DANGER, fg=TEXT_PRI, relief="flat", bd=0,
            padx=15, pady=8, cursor="hand2"
        )
        self.cancel_btn.pack(side="right")

        # Protocol handlers
        self.train_popup.protocol("WM_DELETE_WINDOW", self.cancel_training)

        # Start thread
        self.start_training_process()

    def start_training_process(self) -> None:
        import threading
        self.train_queue.queue.clear()
        self.training_thread = threading.Thread(
            target=self._run_training_thread,
            daemon=True,
            name="train-worker"
        )
        self.training_thread.start()
        # Start checking queue
        self.root.after(100, self.check_training_queue)

    def _run_training_thread(self) -> None:
        import subprocess
        import sys
        import os

        py_exe = sys.executable

        # 1. Run prepare_dataset.py
        self.train_queue.put(("log", "Starting dataset preparation...\n"))
        try:
            self.train_proc = subprocess.Popen(
                [py_exe, "prepare_dataset.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )

            while self.train_proc.poll() is None:
                line = self.train_proc.stdout.readline()
                if line:
                    self.train_queue.put(("log", line))

            # read residual
            for line in self.train_proc.stdout:
                self.train_queue.put(("log", line))

            if self.is_training_cancelled:
                self.train_queue.put(("log", "\nTraining cancelled by user.\n"))
                self.train_queue.put(("done", False))
                return

            if self.train_proc.returncode != 0:
                self.train_queue.put(("log", f"\nDataset preparation failed with exit code {self.train_proc.returncode}\n"))
                self.train_queue.put(("done", False))
                return

        except Exception as e:
            self.train_queue.put(("log", f"\nError starting dataset preparation: {e}\n"))
            self.train_queue.put(("done", False))
            return

        # 2. Run train_model.py
        self.train_queue.put(("log", "\nDataset prepared successfully. Starting training...\n"))
        try:
            self.train_proc = subprocess.Popen(
                [py_exe, "train_model.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )

            while self.train_proc.poll() is None:
                line = self.train_proc.stdout.readline()
                if line:
                    self.train_queue.put(("log", line))

            # read residual
            for line in self.train_proc.stdout:
                self.train_queue.put(("log", line))

            if self.is_training_cancelled:
                self.train_queue.put(("log", "\nTraining cancelled by user.\n"))
                self.train_queue.put(("done", False))
                return

            success = (self.train_proc.returncode == 0)
            if success:
                self.train_queue.put(("log", "\nTraining finished successfully!\n"))
            else:
                self.train_queue.put(("log", f"\nTraining failed with exit code {self.train_proc.returncode}\n"))
            self.train_queue.put(("done", success))

        except Exception as e:
            self.train_queue.put(("log", f"\nError running training: {e}\n"))
            self.train_queue.put(("done", False))

    def check_training_queue(self) -> None:
        import queue
        if not hasattr(self, "train_popup") or not self.train_popup.winfo_exists():
            return

        try:
            while True:
                msg_type, val = self.train_queue.get_nowait()
                if msg_type == "log":
                    self.log_message(val)
                    self.parse_training_line(val)
                elif msg_type == "done":
                    self.on_training_complete(val)
                self.train_queue.task_done()
        except queue.Empty:
            pass

        # Schedule next check if process is still active
        is_active = False
        if hasattr(self, "training_thread") and self.training_thread and self.training_thread.is_alive():
            is_active = True

        if is_active:
            self.root.after(100, self.check_training_queue)

    def parse_training_line(self, line: str) -> None:
        import re
        # Match epoch progress, e.g. "  5/20"
        epoch_match = re.search(r'\b(\d+)/20\b', line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            self.epoch_var.set(f"Epoch: {epoch}/20")
            self.train_progress["value"] = epoch

            # Try to parse loss (the next float elements in the line)
            parts = line.strip().split()
            losses = []
            for p in parts:
                try:
                    val = float(p.replace('G', '').replace('M', ''))
                    if '.' in p:
                        losses.append(val)
                except ValueError:
                    pass
            if len(losses) >= 2:
                self.loss_var.set(f"Loss: {losses[0]:.4f} (box) | {losses[1]:.4f} (cls)")
            elif len(losses) == 1:
                self.loss_var.set(f"Loss: {losses[0]:.4f}")

    def cancel_training(self) -> None:
        self.is_training_cancelled = True
        self.log_message("\nCancelling training process...\n")
        if self.train_proc:
            try:
                self.train_proc.terminate()
                self.train_proc.wait(timeout=2)
            except Exception:
                try:
                    self.train_proc.kill()
                except Exception:
                    pass
        self.status_label.config(text="Training cancelled.", fg=WARNING_C)
        self.train_btn.config(state="normal")
        if hasattr(self, "train_popup") and self.train_popup.winfo_exists():
            self.train_popup.destroy()

    def on_training_complete(self, success: bool) -> None:
        if hasattr(self, "train_popup") and self.train_popup.winfo_exists():
            self.train_popup.destroy()

        self.train_btn.config(state="normal")

        if success:
            self.status_label.config(text="Training completed successfully!", fg=SUCCESS)
            # Reload model
            self.model, self.is_custom = load_model()
            if self.is_custom:
                nc = len(self.model.names)
                model_tag = f"Food-101 Model ({nc} classes)"
                self.title_label.config(
                    text=f"🍽  AI Food Calorie Tracker   [{model_tag}]",
                    fg=SUCCESS
                )
        else:
            if not self.is_training_cancelled:
                self.status_label.config(text="⚠ Training failed. Check logs.", fg=DANGER)

    def log_message(self, message: str) -> None:
        if hasattr(self, "log_text") and self.log_text.winfo_exists():
            self.log_text.configure(state="normal")
            self.log_text.insert("end", message)
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

    def __del__(self) -> None:
        if hasattr(self, "cap") and self.cap:
            self.cap.release()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _print_usage() -> None:
    print("Usage:")
    print("  python main.py <image_path>   — CLI: analyse an image file")
    print("  python main.py                — GUI: live webcam tracker")


if __name__ == "__main__":
    args = sys.argv[1:]

    if args and args[0] in ("-h", "--help"):
        _print_usage()
        sys.exit(0)

    if args:
        # ── CLI mode ──────────────────────────────────────────────────────────
        run_cli(args[0])
    else:
        # ── GUI mode ──────────────────────────────────────────────────────────
        import tkinter as tk
        root = tk.Tk()
        app  = FoodCalorieTracker(root)
        root.mainloop()