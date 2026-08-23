from ultralytics import YOLO
import os

# =========================
# 1. Load YOLO Model
# =========================
model = YOLO("yolov8s.pt")
dataset_path = os.


# =========================
# 2. Train
# =========================
results = model.train(
    data=r"C:\Y3S1\image processing\assignment\pcb-defect-preprocessed\data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=0,
    cache=True,
    project="runs",
    name="pcb_defect"
)


# =========================
# 3. Validate
# =========================
metrics = model.val(
    data=r"C:\Y3S1\image processing\assignment\pcb-defect-preprocessed\data.yaml",
    cache=True,
    device=0,
    workers=0
)

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)