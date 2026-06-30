# AI Food Calorie Tracker

A real-time Computer Vision application that uses **YOLOv8** to identify food items through a webcam (or uploaded image) and instantly calculate nutritional data.

---

## 🚀 Key Features

- **YOLO Detection**: YOLOv8n detects and classifies food items in real time — either using a custom-trained Food-101 model or the COCO pretrained fallback.
- **Custom 15-Class Training**: Train a YOLOv8n model on 15 selected Food-101 classes for direct detection.
- **Nutrition Database Lookup**: Calories, protein, fat, and carbs from a local JSON database with optional live lookup via the **USDA FoodData Central API**.
- **Smart Deduplication**: Prevents calorie double-counting by tracking unique items per session.
- **Per-item Breakdown**: Sidebar shows each detected food with calories, serving size, and macros.
- **Daily Goal Progress Bar**: Visual progress towards a 2 000 kcal daily goal.
- **Upload Image**: Analyse a static photo in addition to live webcam.
- **In-App Training**: Train the YOLO model directly from the GUI with live progress.

---

## 🏗️ Architecture

```
Webcam / Uploaded Image
        │
        ▼
 ┌──────────────────────────┐
 │  YOLOv8n                 │  ← Object detection + classification
 │  (custom or COCO)        │     15 food classes (trained) or 80 COCO classes (fallback)
 └──────────────────────────┘
        │  food label + bounding box
        ▼
 ┌──────────────────────────┐
 │  Nutrition DB Lookup     │  ← nutrition_db.py
 │  local JSON → USDA API   │     3-tier: cache → local → API
 └──────────────────────────┘
        │  {calories, protein, fat, carbs, serving_g}
        ▼
 ┌──────────────────────────┐
 │  Tkinter UI              │  ← sidebar + progress bar + training popup
 └──────────────────────────┘
```

---

## 📁 Project Structure

```
food-calorie-detector/
├── main.py                ← Application entry point (GUI + detection loop)
├── nutrition_db.py        ← Nutrition lookup (local JSON + USDA API)
├── nutrition_data.json    ← Local nutrition database (101 classes + extras)
├── prepare_dataset.py     ← Dataset preparation (train/val split + YOLO labels)
├── train_model.py         ← YOLOv8n training script
├── requirements.txt       ← Python dependencies
├── food-101-overview.ipynb← Exploratory notebook
├── reference_images/      ← Sample images (apple, banana, orange)
│   ├── apple/
│   ├── banana/
│   └── orange/
├── models/                ← Trained model weights (auto-created)
│   └── best.pt            ← Custom YOLOv8n (after training)
├── food-101/              ← Dataset (auto-created by download/prepare scripts)
│   ├── images/
│   │   ├── train/         ← 80% split (12,000 images)
│   │   └── val/           ← 20% split (3,000 images)
│   └── labels/
│       ├── train/
│       └── val/
├── LICENSE
├── README.md
└── .gitignore
```

---

## 🎯 15 Training Classes

The custom YOLO model is trained on these Food-101 classes:

| # | Class | Calories (per 100g) |
|---|---|---|
| 0 | apple_pie | 237 kcal |
| 1 | chicken_curry | 150 kcal |
| 2 | chicken_wings | 290 kcal |
| 3 | dumplings | 232 kcal |
| 4 | french_toast | 229 kcal |
| 5 | fried_calamari | 175 kcal |
| 6 | fried_rice | 163 kcal |
| 7 | garlic_bread | 287 kcal |
| 8 | hamburger | 295 kcal |
| 9 | hot_and_sour_soup | 40 kcal |
| 10 | omelette | 154 kcal |
| 11 | pancakes | 227 kcal |
| 12 | pizza | 266 kcal |
| 13 | samosa | 262 kcal |
| 14 | spring_rolls | 200 kcal |

---

## 🛠️ Installation & Setup

### Prerequisites
Python 3.9+ is required.

```bash
pip install -r requirements.txt
```

### Run the App

```bash
python main.py
```

### Train the Custom Model (Optional)

```bash
# Step 1: Prepare dataset (train/val split + labels)
python prepare_dataset.py

# Step 2: Train YOLOv8n (GPU recommended)
python train_model.py
```

Training parameters: `epochs=20, imgsz=224, batch=16, patience=5`

> **Tip**: CPU training takes 5-7 hours. Use Google Colab (free GPU) for faster training, then copy `best.pt` to `models/`.

---

## 🌐 USDA API (Optional)

For live nutrition data beyond the local database, obtain a free API key at
[fdc.nal.usda.gov](https://fdc.nal.usda.gov/api-guide.html) and set:

```bash
set USDA_API_KEY=your_key_here   # Windows
```

Without a key, `DEMO_KEY` is used (1 000 requests/hour, no registration needed).

---

## 🔧 Configuration

| Variable | Default | Description |
|---|---|---|
| `DAILY_GOAL` | `2000` | Daily kcal goal for progress bar |
| `LOOP_INTERVAL_MS` | `33` | Detection loop interval in ms (~30 FPS ceiling) |
| `USDA_API_KEY` env var | `DEMO_KEY` | USDA FoodData Central API key |
