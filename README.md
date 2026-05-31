# 🚗 Tesla Autopilot Clone — YOLOv8 Object Detection System

> A real-time autonomous driving object detection pipeline built with YOLOv8, OpenCV, and Python — inspired by Tesla Autopilot's perception stack.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?style=flat-square&logo=opencv)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=flat-square)

---

## 📌 Overview

This project implements a **YOLOv8-based object detection system** for autonomous driving scenarios. It detects and classifies key road entities in real-time — vehicles, pedestrians, and traffic signs — simulating the core perception layer of a self-driving system like Tesla Autopilot.

---

## 🎯 Model Performance

| Metric / Class | Score |
|---|---|
| **mAP50** | **0.89** |
| **mAP50-95** | **0.74** |
| Vehicle | 0.92 |
| Pedestrian | 0.87 |
| Traffic Sign | 0.84 |

> Trained on YOLOv8n · 50 epochs · image size 640px · batch size 16

---

## 🗂️ Project Structure

```
tesla-autopilot-clone/
├── autopilot_clone.py    # Main script (train / evaluate / detect / report)
├── dataset.yaml          # Dataset configuration for YOLOv8 training
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Features

| Feature | Description |
|---|---|
| 🏋️ **Train** | Fine-tune YOLOv8n on custom autonomous driving dataset |
| 📊 **Evaluate** | Get mAP50, mAP50-95, precision, recall metrics |
| 🎥 **Video Detection** | Real-time frame-by-frame detection on video files |
| 🖼️ **Image Detection** | Single image inference with annotated output saved |
| 📄 **HTML Report** | Auto-generated styled report with metric visualizations |

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Gowdham-Kumar-C-dev/tesla-autopilot-clone.git
cd tesla-autopilot-clone
```

### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🗃️ Dataset Setup

Organize your dataset in YOLO format:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each `.txt` label file (YOLO format):
```
class_id  x_center  y_center  width  height
```

Update `dataset.yaml` with your dataset path if needed.

> **Recommended datasets:** [KITTI](http://www.cvlibs.net/datasets/kitti/), [BDD100K](https://bdd-data.berkeley.edu/), or [COCO](https://cocodataset.org/) filtered for road objects.

---

## ▶️ Usage

Run the interactive menu:

```bash
python autopilot_clone.py
```

```
========================================
   TESLA AUTOPILOT CLONE
   YOLOv8 Object Detection System
========================================

  1. Train Model
  2. Evaluate Model
  3. Run Video Detection
  4. Detect Single Image
  5. Generate HTML Report
  q. Quit

Select Option:
```

### Quick commands (without menu)

```python
# Train
from autopilot_clone import train_model
train_model()

# Evaluate
from autopilot_clone import evaluate_model
evaluate_model("runs/detect/autopilot_clone/weights/best.pt")

# Generate Report
from autopilot_clone import generate_report
generate_report()
```

---

## 📄 HTML Report

Running option **5** generates `inference_report.html` — a dark-themed report with:
- Summary stat boxes (mAP50, mAP50-95, class count)
- Per-class performance table with visual progress bars
- Color-coded scores (green ≥ 0.85 / yellow ≥ 0.75 / red < 0.75)

---

## 🧠 How It Works

```
Input (Video / Image)
        ↓
  YOLOv8 Inference
        ↓
  Bounding Box + Class + Confidence
        ↓
  OpenCV Annotation (plot)
        ↓
  Display / Save Output
```

- YOLOv8 runs inference on each frame
- Results include bounding boxes, class labels, and confidence scores
- OpenCV draws annotations on the frame
- Video: displayed live in window (press `q` to quit)
- Image: window + saved as `detected_<filename>`

---

## 📦 Dependencies

| Library | Version | Purpose |
|---|---|---|
| ultralytics | ≥ 8.0.0 | YOLOv8 model |
| opencv-python | ≥ 4.8.0 | Video/image I/O and display |
| torch | ≥ 2.0.0 | Deep learning backend |
| torchvision | ≥ 0.15.0 | Vision utilities |
| numpy | ≥ 1.24.0 | Array operations |

---

## 👨‍💻 Author

**Gowdham Kumar C**
- 🎓 B.Sc Computer Science & Applications — SKASC, Coimbatore (Reg: 24BCC021)
- 💼 IoT Head — Google On Campus, SKASC
- 🌐 [gowdhamkumar.netlify.app](https://gowdhamkumar.netlify.app)
- 🔗 [LinkedIn](https://www.linkedin.com/in/gowdham-kumar)
- 🐙 [GitHub](https://github.com/Gowdham-Kumar-C-dev)
- 📧 gowdhamkumarc@gmail.com

---

## 📄 License

This project is licensed under the MIT License — free to use, modify, and share.
