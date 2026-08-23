# ============================================================
# EXPERIMENT 1 — BASELINE YOLOv8m
# NO IMAGE PREPROCESSING
# ============================================================

import os
from ultralytics import YOLO


# ============================================================
# 1. ORIGINAL DATASET
# ============================================================

DATASET_ROOT = "./pcb-defect-dataset"

DATA_YAML = os.path.join(
    DATASET_ROOT,
    "data.yaml"
)


print("\n==============================================")
print("BASELINE DATASET")
print("==============================================")

print("Dataset:", DATASET_ROOT)
print("Dataset exists:", os.path.exists(DATASET_ROOT))
print("data.yaml:", DATA_YAML)
print("data.yaml exists:", os.path.exists(DATA_YAML))


# ============================================================
# 2. CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_ROOT):

    raise FileNotFoundError(
        f"Dataset not found: {DATASET_ROOT}"
    )


if not os.path.exists(DATA_YAML):

    raise FileNotFoundError(
        f"data.yaml not found: {DATA_YAML}"
    )

# ============================================================
# 4. TRAIN BASELINE YOLOv8m
# ============================================================

print("\n==============================================")
print("TRAINING BASELINE YOLOv8m")
print("==============================================")

model = YOLO(
    "yolov8m.pt"
)

RESULTS_DIR = (
    "./pcb-yolo-results"
)


model.train(

    data=DATA_YAML,

    epochs=100,

    imgsz=640,

    batch=16,

    pretrained=True,

    project=RESULTS_DIR,

    name="baseline",

    exist_ok=True,

    patience=20,

    save=True,

    plots=True

)