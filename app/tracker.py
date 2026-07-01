"""
app/tracker.py — FoodCalorieTracker GUI class.

The main Tkinter application window with:
  • Live webcam feed with YOLO detection overlay
  • Image upload mode
  • Sidebar with FoodCards showing macros + confidence badges
  • Meal totals with daily goal progress bar
  • FPS counter and device indicator
  • In-app model training popup
  • Proper resource cleanup on close

Preserved public method names: setup_ui(), reset(), update_loop()
"""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from app.constants import (
    ACCENT, ACCENT_L, BG_CARD, BG_DARK, BG_SIDEBAR, BORDER,
    BOX_BGR, DAILY_GOAL, DANGER, LOOP_INTERVAL_MS,
    SUCCESS, TEXT_PRI, TEXT_SEC, WARNING_C,
)
from app.detector import FoodDetector, Detection, FrameResult
from app.estimation import ServingEstimate, estimate_serving
from app.nutrition import get_nutrition
from app.widgets import FoodCard, MealTotals, StatusBar

logger = logging.getLogger("FoodTracker.GUI")


class FoodCalorieTracker:
    """
    Real-time food calorie tracker with Tkinter UI.

    Single inference pipeline:
        Camera/Image → YOLOv8 → Food Label → Nutrition DB → Calorie Estimation → UI
    """

    def __init__(self, root) -> None:
        import tkinter as tk
        import cv2 as _cv2

        self.root = root
        self.cv2  = _cv2
        self.root.title("AI Food Calorie Tracker")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1280x720")
        self.root.minsize(900, 600)

        # ── Detector (encapsulates YOLO + thresholds + smoothing) ─────────────
        self.detector = FoodDetector()

        # ── Session state ─────────────────────────────────────────────────────
        # tracked_items: label → ServingEstimate
        self.tracked_items: dict[str, ServingEstimate] = {}
        self.total_calories: float = 0.0
        self.total_protein: float = 0.0
        self.total_fat: float = 0.0
        self.total_carbs: float = 0.0

        self.static_frame = None
        self._static_processed = False
        self._static_annotated = None

        # ── Training state ────────────────────────────────────────────────────
        self.train_queue: queue.Queue = queue.Queue()
        self.train_proc = None
        self.is_training_cancelled = False
        self._train_total_epochs = 20

        # ── Build UI ──────────────────────────────────────────────────────────
        self.setup_ui()

        # ── Webcam ────────────────────────────────────────────────────────────
        self.cap = _cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_bar.set_status(
                "⚠ Webcam not found. Use Upload Image to analyse a file.",
                fg=WARNING_C,
            )

        # Show model error if load failed
        if self.detector.model_error:
            self.status_bar.set_status(
                f"⚠ {self.detector.model_error[:80]}", fg=DANGER
            )

        # ── Cleanup on close ──────────────────────────────────────────────────
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # ── Start loop ────────────────────────────────────────────────────────
        self.status_bar.set_device(self.detector.device_tag)
        self.update_loop()

    # ══════════════════════════════════════════════════════════════════════════
    # Cleanup
    # ══════════════════════════════════════════════════════════════════════════

    def _on_closing(self) -> None:
        """Release webcam and destroy the window."""
        logger.info("Shutting down — releasing resources.")
        try:
            if hasattr(self, "cap") and self.cap is not None and self.cap.isOpened():
                self.cap.release()
                logger.info("Webcam released.")
        except Exception as exc:
            logger.warning("Error releasing webcam: %s", exc)

        if hasattr(self, "train_proc") and self.train_proc is not None:
            try:
                self.train_proc.terminate()
                self.train_proc.wait(timeout=2)
            except Exception:
                try:
                    self.train_proc.kill()
                except Exception:
                    pass

        self.root.destroy()

    # ══════════════════════════════════════════════════════════════════════════
    # UI Construction (preserved method name)
    # ══════════════════════════════════════════════════════════════════════════

    def setup_ui(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)

        # ── Left panel: video ─────────────────────────────────────────────────
        left = tk.Frame(self.root, bg=BG_DARK)
        left.grid(row=0, column=0, sticky="nsew", padx=(12, 6), pady=(12, 6))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        # Title with model info
        model_tag = self.detector.model_tag
        fg_color = SUCCESS if self.detector.is_custom else ACCENT_L
        if self.detector.model_error:
            fg_color = DANGER

        self.title_label = tk.Label(
            left,
            text=f"🍽  AI Food Calorie Tracker   [{model_tag}]",
            font=("Segoe UI", 15, "bold"),
            bg=BG_DARK, fg=fg_color,
        )
        self.title_label.grid(row=0, column=0, sticky="w", pady=(0, 6))

        self.video_label = tk.Label(left, bg="#000010", relief="flat")
        self.video_label.grid(row=1, column=0, sticky="nsew")

        # ── Right panel: sidebar ──────────────────────────────────────────────
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
            list_f, bg=BG_SIDEBAR, highlightthickness=0, width=260,
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

        # ── Meal totals section ───────────────────────────────────────────────
        self.meal_totals = MealTotals(right)
        self.meal_totals.pack(fill="x", padx=14, pady=(4, 2))

        # Goal + progress bar
        tk.Label(
            right, text=f"Daily Goal: {DAILY_GOAL} kcal",
            font=("Segoe UI", 9), bg=BG_SIDEBAR, fg=TEXT_SEC,
        ).pack(anchor="w", padx=14)
        prog_outer = tk.Frame(right, bg=BORDER, height=10)
        prog_outer.pack(fill="x", padx=14, pady=(3, 12))
        self.prog_bar = tk.Frame(prog_outer, bg=ACCENT, height=10)
        self.prog_bar.place(x=0, y=0, relheight=1.0, relwidth=0.0)

        # ── Buttons ───────────────────────────────────────────────────────────
        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", padx=10)
        btn_f = tk.Frame(right, bg=BG_SIDEBAR)
        btn_f.pack(fill="x", padx=14, pady=12)

        row_btn_f = tk.Frame(btn_f, bg=BG_SIDEBAR)
        row_btn_f.pack(fill="x", pady=(0, 6))

        self.upload_btn = self._btn(
            row_btn_f, "📁  Upload Image", self.upload_image, ACCENT
        )
        self.upload_btn.pack(side="left", fill="x", expand=True, padx=(0, 3))

        self.train_btn = self._btn(
            row_btn_f, "⚙️  Train Model", self.open_train_popup, "#d81b60"
        )
        self.train_btn.pack(side="right", fill="x", expand=True, padx=(3, 0))

        self.reset_btn = self._btn(btn_f, "🔄  Reset", self.reset, "#3a3a5a")
        self.reset_btn.pack(fill="x")

        # ── Status bar ────────────────────────────────────────────────────────
        self.status_bar = StatusBar(self.root)
        self.status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _btn(self, parent, text, cmd, bg_color):
        import tkinter as tk
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

    # ══════════════════════════════════════════════════════════════════════════
    # Reset (preserved method name)
    # ══════════════════════════════════════════════════════════════════════════

    def reset(self) -> None:
        """Clear all detected items and calorie total."""
        self.total_calories = 0.0
        self.total_protein = 0.0
        self.total_fat = 0.0
        self.total_carbs = 0.0
        self.tracked_items.clear()
        self.static_frame = None
        self._static_processed = False
        self._static_annotated = None
        self.detector.clear_history()
        self._refresh_sidebar()
        self.meal_totals.update_totals(0, 0, 0, 0, DAILY_GOAL)
        self.prog_bar.place(relwidth=0.0)
        self.status_bar.set_count(0)
        self.status_bar.set_status("Reset.", fg=TEXT_SEC)

    # ══════════════════════════════════════════════════════════════════════════
    # Upload image
    # ══════════════════════════════════════════════════════════════════════════

    def upload_image(self) -> None:
        from tkinter import filedialog
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
                self._static_processed = False
                self._static_annotated = None
                self.status_bar.set_status(
                    f"Image loaded: {Path(path).name}", fg=TEXT_SEC
                )
            else:
                self.status_bar.set_status(
                    "⚠ Could not read image file.", fg=DANGER
                )

    # ══════════════════════════════════════════════════════════════════════════
    # Main loop (preserved method name)
    # ══════════════════════════════════════════════════════════════════════════

    def update_loop(self) -> None:
        from PIL import Image, ImageTk

        frame = None

        if self.static_frame is not None:
            if self._static_processed and self._static_annotated is not None:
                frame = self._static_annotated
            else:
                frame = self.static_frame.copy()
                result = self.detector.detect(frame)
                frame = self._process_result(frame, result, is_static=True)
                self._static_annotated = frame.copy()
                self._static_processed = True
        elif self.cap is not None and self.cap.isOpened():
            ret, cam = self.cap.read()
            if ret:
                frame = cam
                result = self.detector.detect(frame)
                frame = self._process_result(frame, result, is_static=False)
                # Update FPS in status bar
                self.status_bar.set_fps(self.detector.fps)

        if frame is not None:
            rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            lw = max(self.video_label.winfo_width(), 640)
            lh = max(self.video_label.winfo_height(), 480)
            pil.thumbnail((lw, lh), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=pil)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            self.status_bar.set_count(len(self.tracked_items))

        self.root.after(LOOP_INTERVAL_MS, self.update_loop)

    # ══════════════════════════════════════════════════════════════════════════
    # Detection processing
    # ══════════════════════════════════════════════════════════════════════════

    def _process_result(
        self, frame, result: FrameResult, *, is_static: bool
    ):
        """Annotate frame with detections and update tracking."""
        sidebar_dirty = False
        h, w = frame.shape[:2]

        # For webcam, use temporal smoothing; for static images, accept all
        if is_static:
            stable_labels = {d.label for d in result.detections}
        else:
            stable_labels = self.detector.get_stable_labels()

        for det in result.detections:
            # Draw bounding box on ALL detections (even unstable)
            self._draw_box(frame, det)

            # Only track items that pass temporal stability
            if det.label not in stable_labels:
                continue

            if det.label not in self.tracked_items:
                # Look up nutrition and estimate serving
                nutrition_info = get_nutrition(det.label)
                estimate = estimate_serving(
                    label=det.label,
                    display=det.display,
                    confidence=det.confidence,
                    bbox_area_ratio=det.bbox_area_ratio,
                    nutrition_info=nutrition_info,
                )
                self.tracked_items[det.label] = estimate
                self.total_calories += estimate.calories
                self.total_protein += estimate.protein
                self.total_fat += estimate.fat
                self.total_carbs += estimate.carbs
                sidebar_dirty = True

        if sidebar_dirty:
            self._refresh_sidebar()
            self._refresh_totals()
            logger.info("--- Detected Ingredients ---")
            for lbl, est in self.tracked_items.items():
                logger.info(
                    "  - %s: %.0f kcal (%s)",
                    est.display, est.calories, est.serving_desc,
                )
            logger.info("Meal Total: %.0f kcal", self.total_calories)

        return frame

    def _draw_box(self, frame, det: Detection) -> None:
        """Draw a bounding box with label tag on the frame."""
        self.cv2.rectangle(
            frame, (det.x1, det.y1), (det.x2, det.y2), BOX_BGR, 2
        )
        tag = f"{det.display}  {det.confidence:.0%}"
        (tw, th), _ = self.cv2.getTextSize(
            tag, self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        self.cv2.rectangle(
            frame,
            (det.x1, max(0, det.y1 - th - 10)),
            (det.x1 + tw + 8, det.y1),
            BOX_BGR, -1,
        )
        self.cv2.putText(
            frame, tag,
            (det.x1 + 4, max(th, det.y1 - 4)),
            self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1,
            self.cv2.LINE_AA,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Sidebar
    # ══════════════════════════════════════════════════════════════════════════

    def _refresh_sidebar(self) -> None:
        import tkinter as tk

        for widget in self.items_inner.winfo_children():
            widget.destroy()

        if not self.tracked_items:
            tk.Label(
                self.items_inner,
                text="Point camera at food…",
                font=("Segoe UI", 10, "italic"),
                bg=BG_SIDEBAR, fg=TEXT_SEC,
            ).pack(padx=10, pady=20)
            return

        for label, est in self.tracked_items.items():
            card = FoodCard(
                self.items_inner,
                display=est.display,
                confidence=est.confidence,
                serving_desc=est.serving_desc,
                calories=est.calories,
                protein=est.protein,
                fat=est.fat,
                carbs=est.carbs,
            )
            card.pack(fill="x", padx=4, pady=3)

    def _refresh_totals(self) -> None:
        self.meal_totals.update_totals(
            self.total_calories,
            self.total_protein,
            self.total_fat,
            self.total_carbs,
            DAILY_GOAL,
        )
        frac = min(1.0, self.total_calories / DAILY_GOAL)
        bar_color = (
            DANGER if frac > 0.85
            else WARNING_C if frac > 0.60
            else ACCENT
        )
        self.prog_bar.place(relwidth=frac)
        self.prog_bar.config(bg=bar_color)

    # ══════════════════════════════════════════════════════════════════════════
    # Training popup (preserved)
    # ══════════════════════════════════════════════════════════════════════════

    def open_train_popup(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        if hasattr(self, "train_popup") and self.train_popup.winfo_exists():
            self.train_popup.lift()
            return

        self.train_btn.config(state="disabled")
        self.is_training_cancelled = False
        self._train_total_epochs = 20

        self.train_popup = tk.Toplevel(self.root)
        self.train_popup.title("Training YOLOv8n Model")
        self.train_popup.configure(bg=BG_SIDEBAR)
        self.train_popup.transient(self.root)
        self.train_popup.grab_set()

        w_width, w_height = 600, 450
        x = self.root.winfo_x() + (self.root.winfo_width() - w_width) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - w_height) // 2
        self.train_popup.geometry(f"{w_width}x{w_height}+{x}+{y}")

        tk.Label(
            self.train_popup, text="YOLOv8n Food-101 Training",
            font=("Segoe UI", 14, "bold"),
            bg=BG_SIDEBAR, fg=TEXT_PRI,
        ).pack(anchor="w", padx=20, pady=(20, 10))

        status_f = tk.Frame(self.train_popup, bg=BG_SIDEBAR)
        status_f.pack(fill="x", padx=20, pady=5)

        self.epoch_var = tk.StringVar(value="Epoch: 0/?")
        tk.Label(
            status_f, textvariable=self.epoch_var,
            font=("Segoe UI", 11, "bold"), bg=BG_SIDEBAR, fg=ACCENT_L,
        ).pack(side="left")

        self.loss_var = tk.StringVar(value="Loss: N/A")
        tk.Label(
            status_f, textvariable=self.loss_var,
            font=("Segoe UI", 11, "bold"), bg=BG_SIDEBAR, fg=SUCCESS,
        ).pack(side="right")

        self.train_progress = ttk.Progressbar(
            self.train_popup, orient="horizontal",
            length=560, mode="determinate",
            maximum=self._train_total_epochs,
        )
        self.train_progress.pack(fill="x", padx=20, pady=10)

        log_f = tk.Frame(self.train_popup, bg=BG_DARK, bd=1, relief="solid")
        log_f.pack(fill="both", expand=True, padx=20, pady=10)

        self.log_text = tk.Text(
            log_f, bg=BG_DARK, fg=TEXT_PRI,
            font=("Consolas", 9), wrap="word", state="disabled",
            highlightthickness=0, bd=0,
        )
        log_scroll = ttk.Scrollbar(
            log_f, orient="vertical", command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self.log_text.pack(side="left", fill="both", expand=True)

        btn_f = tk.Frame(self.train_popup, bg=BG_SIDEBAR)
        btn_f.pack(fill="x", padx=20, pady=(10, 20))

        self.cancel_btn = tk.Button(
            btn_f, text="Cancel Training",
            command=self.cancel_training,
            font=("Segoe UI", 10, "bold"),
            bg=DANGER, fg=TEXT_PRI, relief="flat", bd=0,
            padx=15, pady=8, cursor="hand2",
        )
        self.cancel_btn.pack(side="right")

        self.train_popup.protocol("WM_DELETE_WINDOW", self.cancel_training)
        self._start_training_process()

    def _start_training_process(self) -> None:
        self.train_queue.queue.clear()
        self.training_thread = threading.Thread(
            target=self._run_training_thread,
            daemon=True,
            name="train-worker",
        )
        self.training_thread.start()
        self.root.after(100, self._check_training_queue)

    def _run_training_thread(self) -> None:
        py_exe = sys.executable

        # 1. prepare_dataset.py
        self.train_queue.put(("log", "Starting dataset preparation...\n"))
        try:
            self.train_proc = subprocess.Popen(
                [py_exe, "prepare_dataset.py"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                ),
            )
            while self.train_proc.poll() is None:
                line = self.train_proc.stdout.readline()
                if line:
                    self.train_queue.put(("log", line))
            for line in self.train_proc.stdout:
                self.train_queue.put(("log", line))

            if self.is_training_cancelled:
                self.train_queue.put(("log", "\nTraining cancelled by user.\n"))
                self.train_queue.put(("done", False))
                return
            if self.train_proc.returncode != 0:
                self.train_queue.put((
                    "log",
                    f"\nDataset preparation failed (exit {self.train_proc.returncode})\n",
                ))
                self.train_queue.put(("done", False))
                return
        except Exception as e:
            self.train_queue.put(("log", f"\nError: {e}\n"))
            self.train_queue.put(("done", False))
            return

        # 2. train_model.py
        self.train_queue.put(("log", "\nDataset ready. Starting training...\n"))
        try:
            self.train_proc = subprocess.Popen(
                [py_exe, "train_model.py"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                ),
            )
            while self.train_proc.poll() is None:
                line = self.train_proc.stdout.readline()
                if line:
                    self.train_queue.put(("log", line))
            for line in self.train_proc.stdout:
                self.train_queue.put(("log", line))

            if self.is_training_cancelled:
                self.train_queue.put(("log", "\nTraining cancelled by user.\n"))
                self.train_queue.put(("done", False))
                return

            success = self.train_proc.returncode == 0
            if success:
                self.train_queue.put(("log", "\nTraining finished successfully!\n"))
            else:
                self.train_queue.put((
                    "log",
                    f"\nTraining failed (exit {self.train_proc.returncode})\n",
                ))
            self.train_queue.put(("done", success))
        except Exception as e:
            self.train_queue.put(("log", f"\nError: {e}\n"))
            self.train_queue.put(("done", False))

    def _check_training_queue(self) -> None:
        import queue as _q
        if not hasattr(self, "train_popup") or not self.train_popup.winfo_exists():
            return
        try:
            while True:
                msg_type, val = self.train_queue.get_nowait()
                if msg_type == "log":
                    self._log_message(val)
                    self._parse_training_line(val)
                elif msg_type == "done":
                    self._on_training_complete(val)
                self.train_queue.task_done()
        except _q.Empty:
            pass

        is_active = (
            hasattr(self, "training_thread")
            and self.training_thread
            and self.training_thread.is_alive()
        )
        if is_active:
            self.root.after(100, self._check_training_queue)

    def _parse_training_line(self, line: str) -> None:
        import re
        epoch_match = re.search(r'\b(\d+)/(\d+)\b', line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            total = int(epoch_match.group(2))
            if total != self._train_total_epochs:
                self._train_total_epochs = total
                self.train_progress.configure(maximum=total)
            self.epoch_var.set(f"Epoch: {epoch}/{total}")
            self.train_progress["value"] = epoch

            parts = line.strip().split()
            losses = []
            for p in parts:
                try:
                    val = float(p.replace("G", "").replace("M", ""))
                    if "." in p:
                        losses.append(val)
                except ValueError:
                    pass
            if len(losses) >= 2:
                self.loss_var.set(
                    f"Loss: {losses[0]:.4f} (box) | {losses[1]:.4f} (cls)"
                )
            elif len(losses) == 1:
                self.loss_var.set(f"Loss: {losses[0]:.4f}")

    def cancel_training(self) -> None:
        self.is_training_cancelled = True
        self._log_message("\nCancelling training process...\n")
        if self.train_proc:
            try:
                self.train_proc.terminate()
                self.train_proc.wait(timeout=2)
            except Exception:
                try:
                    self.train_proc.kill()
                except Exception:
                    pass
        self.status_bar.set_status("Training cancelled.", fg=WARNING_C)
        self.train_btn.config(state="normal")
        if hasattr(self, "train_popup") and self.train_popup.winfo_exists():
            self.train_popup.destroy()

    def _on_training_complete(self, success: bool) -> None:
        if hasattr(self, "train_popup") and self.train_popup.winfo_exists():
            self.train_popup.destroy()
        self.train_btn.config(state="normal")

        if success:
            self.status_bar.set_status(
                "Training completed successfully!", fg=SUCCESS
            )
            self.detector.reload_model()
            model_tag = self.detector.model_tag
            fg_color = SUCCESS if self.detector.is_custom else ACCENT_L
            self.title_label.config(
                text=f"🍽  AI Food Calorie Tracker   [{model_tag}]",
                fg=fg_color,
            )
        else:
            if not self.is_training_cancelled:
                self.status_bar.set_status(
                    "⚠ Training failed. Check logs.", fg=DANGER
                )

    def _log_message(self, message: str) -> None:
        if hasattr(self, "log_text") and self.log_text.winfo_exists():
            self.log_text.configure(state="normal")
            self.log_text.insert("end", message)
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
