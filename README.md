# PCB Defect Inspection System
**Course:** BMDS2133 Image Processing

**Name:** Tan Jin Yuan (25WMR12328), Khoo Hou Sheng (25WMR03473), Chen Xiang Hui (25WMR12896)

---

## Group Members & Responsibilities

| Name | Folder | Primary Responsibility |
|---|---|---|
| **Chen Xiang Hui** | `Jayden/` | Sobel Image Edge Detection & Preprocessing Experiments |
| **Tan Jin Yuan** | `ktyj/` | CLAHE Pipeline, YOLOv8 Training, Streamlit App & Evaluation |
| **Khoo Hou Sheng** | `KHS/` | PiDiNet Edge Detection & Deep Edge Feature Fusion |

---

## Project Overview

This project implements an automated PCB defect detection and classification system using classical image processing techniques combined with a fine-tuned YOLOv8 object detection model. The system detects six categories of PCB manufacturing defects:

`mouse_bite` · `spur` · `missing_hole` · `short` · `open_circuit` · `spurious_copper`

---

## Repository Structure
Note: The dataset will not be included in submission due to size constraints. 
```
Y3S1_Image_Processing/
│
├── README.md                             ← You are here
├── flowchart.png                         ← System architecture flowchart
├── training.py                           ← YOLOv8 model training script
│
├── runs/                                 ← Trained YOLOv8 models & evaluation runs (weights)
│
├── ktyj/                                 ← Tan Jin Yuan
│   ├── pipeline1.ipynb                   ← Main experiment notebook (pipelines + training)
│   ├── batch_processing_app.py           ← Streamlit GUI application (batch & video inference)
│   ├── samples/                          ← Sample images for app testing
│   └── beta_samples/                     ← Additional test samples
│
├── KHS/                                  ← Khoo Hou Sheng
│   ├── demo.ipynb                        ← PiDiNet edge detection demo notebook
│   ├── marge_pidinet.py                  ← PiDiNet edge map + weighted feature fusion (0.8/0.2)
│   ├── preprocessing_storing.py          ← Saves preprocessed PiDiNet dataset to disk
│   ├── preprocessing_testing.py          ← Tests PiDiNet preprocessing on sample images
│   ├── edge_detection.py                 ← Standalone edge detection utility
│   ├── function/                         ← Shared helper functions
│   └── pidinet/                          ← PiDiNet model weights and architecture
│
├── Jayden/                               ← Chen Xiang Hui
│   ├── preprocessing_with_median_filtering_(K=9)_and_CLAHE.py   ← OPTIMAL pipeline
│   ├── preprocessing_with_median_filtering_(K=3)_and_CLAHE.py   ← Variant (smaller kernel)
│   ├── preprocessing_with_CLAHE.py                               ← CLAHE only
│   ├── preprocessing_with_median_filtering_(K=9).py              ← Median blur only
│   ├── preprocessing_with_Sharpening.py                          ← Sharpening experiments
│   ├── validation.py                                             ← Model validation script
│   └── [other preprocessing variants...]
│
├── pcb-defect-dataset/                   ← Base dataset (YOLO format, NC=6)
│   └── data.yaml                         ← Dataset config (paths + class names)
│
├── pcb-defect-dataset-sobel-solder-mask/ ← Preprocessed Dataset (Median + Sobel + Solder Mask Threshold)
└── pcb-defect-dataset-solder-mask/       ← Preprocessed Dataset (Median + Solder Mask Threshold)
```

---

## How to Run

### Prerequisites
```bash
pip install ultralytics opencv-python streamlit numpy
```

### 1. Streamlit Inference App
```bash
cd ktyj
streamlit run batch_processing_app.py
```
Launches the GUI where you can upload PCB images or video for real-time defect detection.

### 2. Re-train the Model
```bash
python training.py
```
Ensure paths in `pcb-defect-dataset/data.yaml` are correct before running.


## Key Results

- **Best Preprocessing Pipeline:** Median Blur (K=9) + CLAHE (LAB Color Space)
- **Model Architecture:** YOLOv8s (`yolov8s.pt`)
- **Defect Classes:** 6 (mouse_bite, spur, missing_hole, short, open_circuit, spurious_copper)
- **Evaluation Metrics:** Precision, Recall, mAP@0.5, mAP@0.5:0.95