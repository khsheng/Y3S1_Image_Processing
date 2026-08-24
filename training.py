from ultralytics import YOLO
import os

if __name__ == "__main__":
  # =========================
  # 1. Load YOLO Model
  # =========================
  model = YOLO("yolov8s.pt")
  dataset_path = os.path.join(os.getcwd(), "pcb-defect-dataset")
  data_yaml_path = os.path.join(dataset_path, "data.yaml")

  result_folder_name = "yolov8s_with_pcb-defect-dataset"

  # =========================
  # 2. Train
  # =========================
  results = model.train(
      data=data_yaml_path,
      epochs=30,
      imgsz=640,
      batch=16,
      device=0,
      workers=4,
      cache=False,
      project="runs",
      name=result_folder_name,
  )

  # =========================
  # 3. Validate
  # =========================
  metrics = model.val(data=data_yaml_path, cache=False, device=0, workers=4)

  print("mAP50:", metrics.box.map50)
  print("mAP50-95:", metrics.box.map)