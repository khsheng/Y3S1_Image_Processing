from ultralytics import YOLO
import os

# =========================
# 1. Load YOLO Model
# =========================
model = YOLO("yolov8s.pt")
dataset_path = os.path.join(os.getcwd(), "pcb-defect-preprocessed")
data_yaml_path = os.path.join(dataset_path, "data.yaml")

result_folder_name = "pcb_preprocessed"


# =========================
# 2. Train
# =========================
results = model.train(
    data=data_yaml_path,
    epochs=30,
    imgsz=640,
    batch=16,
    device=0,
    workers=0,
    cache=True,
    project="runs_khs",
    name=result_folder_name
)


# =========================
# 3. Validate
# =========================
metrics = model.val(
    data=data_yaml_path,
    cache=True,
    device=0,
    workers=0
)

print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)