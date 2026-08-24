# ============================================================
# EXPERIMENT 1 — BASELINE YOLOv8m
# NO IMAGE PREPROCESSING
# ============================================================

import os
from ultralytics import YOLO


# ============================================================
# 1. ORIGINAL DATASET
# ============================================================

DATASET_ROOT = os.path.join(os.getcwd(), "pcb-defect-dataset")

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