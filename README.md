# Production-Ready AI Food Calorie Tracker

A modular, real-time Computer Vision application that uses **YOLOv8** to identify food items via a webcam (or uploaded image) and estimates portion size and complete nutritional profiles.

---

## 🍽️ Features & Production Upgrades

- **Optimised YOLO Inference**: Custom-tuned confidence (0.45) and IoU NMS (0.50) thresholds to eliminate false positives and merge duplicates.
- **Webcam Temporal Smoothing**: Implements a sliding window queue filter preventing single-frame detection flickers.
- **Portion Size Estimation**: Evaluates bounding box area relative to the frame size to estimate multipliers (0.5×, 1.0×, 1.5×, 2.0×) and scale calories/macros accordingly.
- **Multi-Food deduplication**: Handles spatial deduplication correctly (e.g. counts two distinct pizzas as two separate items rather than skipping duplicates).
- **Full Macro Breakdown**: Tracks Calories, Protein, Fat, Carbohydrates, Fiber, Sugar, and Sodium.
- **Modernized Dark Theme GUI**: Displays real-time FPS, device engine badge (CPU vs GPU name), confidence status tags (HIGH/MED/LOW), meal totals, and custom progress indicators.
- **Modular Software Design**: Refactored into a structured Python package (`app/`) for strict separation of detection, nutrition, serving estimation, and UI layers.

---

## 🏗️ Architecture Diagram

```
 Webcam / Image File
        │
        ▼
 ┌──────────────┐
 │ FoodDetector │  ← app/detector.py (YOLOv8 + FP16 + NMS + Temporal Smoothing)
 └──────────────┘
        │ Stable Detection objects
        ▼
 ┌──────────────────┐
 │ serving_estimate │  ← app/estimation.py (Bbox Area portion sizing)
 └──────────────────┘
        │ Bbox Area Ratio + standard serving grams
        ▼
 ┌──────────────────┐
 │    nutrition     │  ← app/nutrition.py (nutrition_db JSON + USDA API fallback)
 └──────────────────┘
        │ {calories, protein, fat, carbs, fiber, sugar, sodium}
        ▼
 ┌──────────────────┐
 │    tracker UI    │  ← app/tracker.py & app/widgets.py (Tkinter frontend)
 └──────────────────┘
```

---

## 📁 Repository Structure

```
food-calorie-detector/
├── main.py                ← Production entry point
├── app/                   ← Core application package
│   ├── __init__.py
│   ├── constants.py       ← Color palette, thresholds, and configuration
│   ├── detector.py        ← YOLO inference, NMS duplicate merge, temporal filter
│   ├── estimation.py      ← Portion size scaling and multiplier mapping
│   ├── nutrition.py       ← Unified macros lookup
│   ├── tracker.py         ← Main GUI application controller
│   ├── widgets.py         ← Custom Tkinter views (cards, status bars)
│   └── cli.py             ← Command-line output parser
├── nutrition_data.json    ← Database containing macros for all 101 classes
├── nutrition_db.py        ← Backwards-compatible lookup module
├── train_model.py         ← Custom training pipeline (optimized hyperparameters)
├── prepare_dataset.py     ← Dataset splitter and label generator
├── evaluate_model.py      ← Accuracy, precision, recall, and mAP evaluation
├── tests/                 ← Unit test suite
│   ├── test_detector.py
│   ├── test_estimation.py
│   └── test_nutrition.py
└── requirements.txt
```

---

## 🛠️ Installation & Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run GUI (live webcam mode)**:
   ```bash
   python main.py
   ```

3. **Run CLI (static file analysis)**:
   ```bash
   python main.py reference_images/apple/sample.jpg
   ```

4. **Run Unit Tests**:
   ```bash
   python -m unittest discover -s tests
   ```

---

## ⚙️ Model Training & Augmentation (M1)

Our training pipeline in `train_model.py` is configured with production-grade hyperparameter tuning:
- **Image Size (`imgsz=416`)**: Larger canvas size for improved object detail recognition.
- **Augmentation Suite**: Includes MixUp (0.1) and Mosaic (1.0) combinations to make the model robust to scale variations and overlap.
- **Cosine Learning Rate Scheduler (`cos_lr=True`)**: Better convergence across the training runs.
- **Early Stopping Patience (`patience=10`)**: Automatic halt when validation performance plateaus.

---

## 🔮 Future Roadmap & Architecture (L2)

For production expansion, the following feature designs are supported by our code structure:
1. **Meal History Database**: Implement a local SQLite database to persist daily caloric logs.
2. **Weekly Analytics**: Integrate `matplotlib` graphs inside the Tkinter GUI to visualize diet habits.
3. **Report Generation**: Export daily summary reports to PDF or CSV format.
4. **Cloud Synchronisation**: Establish REST client sync with a remote backend.
