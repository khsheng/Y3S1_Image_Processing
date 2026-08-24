# ============================================================
# BASELINE FASTER R-CNN — PCB DEFECT DETECTION
# ============================================================
#
# Model:
#   Faster R-CNN + ResNet-50 FPN
#
# Dataset:
#   pcb-defect-dataset
#
# Annotation:
#   YOLO .txt format
#
# Metrics:
#   Precision
#   Recall
#   mAP50
#   mAP50-95
#
# ============================================================


import os
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import functional as F

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)

from torchmetrics.detection.mean_ap import MeanAveragePrecision


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATASET_PATH = os.path.join(
    os.getcwd(),
    "pcb-defect-dataset"
)

DATA_YAML_PATH = os.path.join(
    DATASET_PATH,
    "data.yaml"
)

RESULT_FOLDER = os.path.join(
    "runs",
    "baseline_faster_rcnn_pcb"
)


# -------------------------
# Training parameters
# -------------------------

EPOCHS = 30

BATCH_SIZE = 4

LEARNING_RATE = 0.005

MOMENTUM = 0.9

WEIGHT_DECAY = 0.0005

NUM_WORKERS = 0

CONFIDENCE_THRESHOLD = 0.5


# -------------------------
# Device
# -------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 2. CREATE RESULT FOLDER
# ============================================================

os.makedirs(
    RESULT_FOLDER,
    exist_ok=True
)


# ============================================================
# 3. PRINT CONFIGURATION
# ============================================================

print("\n")
print("=" * 60)
print("BASELINE FASTER R-CNN")
print("PCB DEFECT DETECTION")
print("=" * 60)

print("Dataset:")
print(DATASET_PATH)

print("\nDevice:")
print(DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 4. LOAD data.yaml
# ============================================================

print("\nLoading data.yaml...")

with open(
    DATA_YAML_PATH,
    "r"
) as file:

    data_config = yaml.safe_load(file)


# ============================================================
# 5. GET CLASS NAMES
# ============================================================

names = data_config["names"]


if isinstance(names, dict):

    names = [
        names[i]
        for i in sorted(names.keys())
    ]


NUM_CLASSES = len(names)


print("\nClasses:")

for i, name in enumerate(names):

    print(
        f"{i}: {name}"
    )


print(
    "\nNumber of defect classes:",
    NUM_CLASSES
)


# ============================================================
# 6. DATASET CLASS
# ============================================================

class PCBDefectDataset(Dataset):

    def __init__(
        self,
        image_dir,
        label_dir
    ):

        self.image_dir = image_dir

        self.label_dir = label_dir

        valid_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        )

        self.images = sorted(
            [
                file
                for file in os.listdir(
                    image_dir
                )
                if file.lower().endswith(
                    valid_extensions
                )
            ]
        )


    # --------------------------------------------------------
    # Number of images
    # --------------------------------------------------------

    def __len__(self):

        return len(self.images)


    # --------------------------------------------------------
    # Load one image
    # --------------------------------------------------------

    def __getitem__(
        self,
        index
    ):

        image_name = self.images[index]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )


        # ====================================================
        # LOAD IMAGE
        # ====================================================

        image = Image.open(
            image_path
        ).convert("RGB")


        width, height = image.size


        # ====================================================
        # FIND YOLO LABEL
        # ====================================================

        label_name = (
            os.path.splitext(
                image_name
            )[0]
            + ".txt"
        )


        label_path = os.path.join(
            self.label_dir,
            label_name
        )


        boxes = []

        labels = []


        # ====================================================
        # READ YOLO ANNOTATIONS
        # ====================================================

        if os.path.exists(label_path):

            with open(
                label_path,
                "r"
            ) as file:

                for line in file:

                    line = line.strip()

                    if not line:
                        continue


                    values = line.split()


                    # YOLO format:
                    #
                    # class
                    # x_center
                    # y_center
                    # width
                    # height

                    class_id = int(
                        values[0]
                    )

                    x_center = float(
                        values[1]
                    )

                    y_center = float(
                        values[2]
                    )

                    box_width = float(
                        values[3]
                    )

                    box_height = float(
                        values[4]
                    )


                    # ====================================================
                    # CONVERT NORMALIZED YOLO COORDINATES
                    # TO PIXEL COORDINATES
                    # ====================================================

                    x_center *= width

                    y_center *= height

                    box_width *= width

                    box_height *= height


                    xmin = (
                        x_center
                        - box_width / 2
                    )

                    ymin = (
                        y_center
                        - box_height / 2
                    )

                    xmax = (
                        x_center
                        + box_width / 2
                    )

                    ymax = (
                        y_center
                        + box_height / 2
                    )


                    # ====================================================
                    # CLIP BOXES
                    # ====================================================

                    xmin = max(
                        0,
                        xmin
                    )

                    ymin = max(
                        0,
                        ymin
                    )

                    xmax = min(
                        width,
                        xmax
                    )

                    ymax = min(
                        height,
                        ymax
                    )


                    # ====================================================
                    # REMOVE INVALID BOXES
                    # ====================================================

                    if (
                        xmax <= xmin
                        or
                        ymax <= ymin
                    ):

                        continue


                    boxes.append(
                        [
                            xmin,
                            ymin,
                            xmax,
                            ymax
                        ]
                    )


                    # ====================================================
                    # IMPORTANT:
                    #
                    # Faster R-CNN reserves:
                    #
                    # class 0 = background
                    #
                    # Therefore:
                    #
                    # YOLO class 0 → Faster R-CNN class 1
                    # YOLO class 1 → Faster R-CNN class 2
                    # etc.
                    # ====================================================

                    labels.append(
                        class_id + 1
                    )


        # ====================================================
        # CONVERT TO TENSOR
        # ====================================================

        boxes = torch.as_tensor(
            boxes,
            dtype=torch.float32
        )

        labels = torch.as_tensor(
            labels,
            dtype=torch.int64
        )


        # ====================================================
        # EMPTY ANNOTATION HANDLING
        # ====================================================

        if len(boxes) == 0:

            boxes = torch.zeros(
                (
                    0,
                    4
                ),
                dtype=torch.float32
            )

            labels = torch.zeros(
                (
                    0,
                ),
                dtype=torch.int64
            )


        # ====================================================
        # AREA
        # ====================================================

        area = (
            (
                boxes[:, 2]
                -
                boxes[:, 0]
            )
            *
            (
                boxes[:, 3]
                -
                boxes[:, 1]
            )
        )


        # ====================================================
        # CROWD
        # ====================================================

        iscrowd = torch.zeros(
            (
                len(boxes),
            ),
            dtype=torch.int64
        )


        # ====================================================
        # IMAGE ID
        # ====================================================

        image_id = torch.tensor(
            [
                index
            ]
        )


        # ====================================================
        # TARGET
        # ====================================================

        target = {

            "boxes":
                boxes,

            "labels":
                labels,

            "image_id":
                image_id,

            "area":
                area,

            "iscrowd":
                iscrowd
        }


        # ====================================================
        # IMAGE → TENSOR
        # ====================================================

        image = F.to_tensor(
            image
        )


        return image, target


# ============================================================
# 7. COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    return tuple(
        zip(*batch)
    )


# ============================================================
# 8. DATASET DIRECTORIES
# ============================================================

TRAIN_IMAGE_DIR = os.path.join(
    DATASET_PATH,
    "train",
    "images"
)

TRAIN_LABEL_DIR = os.path.join(
    DATASET_PATH,
    "train",
    "labels"
)


VALID_IMAGE_DIR = os.path.join(
    DATASET_PATH,
    "val",
    "images"
)

VALID_LABEL_DIR = os.path.join(
    DATASET_PATH,
    "val",
    "labels"
)


# ============================================================
# 9. CREATE DATASETS
# ============================================================

print("\n")
print("=" * 60)
print("LOADING DATASET")
print("=" * 60)


train_dataset = PCBDefectDataset(
    TRAIN_IMAGE_DIR,
    TRAIN_LABEL_DIR
)


valid_dataset = PCBDefectDataset(
    VALID_IMAGE_DIR,
    VALID_LABEL_DIR
)


print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(valid_dataset)
)


# ============================================================
# 10. CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=NUM_WORKERS,

    collate_fn=collate_fn
)


valid_loader = DataLoader(

    valid_dataset,

    batch_size=1,

    shuffle=False,

    num_workers=NUM_WORKERS,

    collate_fn=collate_fn
)


# ============================================================
# 11. LOAD PRETRAINED FASTER R-CNN
# ============================================================

print("\n")
print("=" * 60)
print("LOADING FASTER R-CNN")
print("=" * 60)


weights = (
    FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)


model = fasterrcnn_resnet50_fpn(
    weights=weights
)


# ============================================================
# 12. REPLACE CLASSIFIER
# ============================================================

# Background + PCB defect classes

NUM_MODEL_CLASSES = (
    NUM_CLASSES + 1
)


in_features = (
    model.roi_heads
    .box_predictor
    .cls_score
    .in_features
)


model.roi_heads.box_predictor = (
    FastRCNNPredictor(
        in_features,
        NUM_MODEL_CLASSES
    )
)


# ============================================================
# 13. MOVE MODEL TO GPU
# ============================================================

model = model.to(
    DEVICE
)


print(
    "Faster R-CNN loaded successfully."
)


# ============================================================
# 14. OPTIMIZER
# ============================================================

params = [

    parameter

    for parameter
    in model.parameters()

    if parameter.requires_grad
]


optimizer = torch.optim.SGD(

    params,

    lr=LEARNING_RATE,

    momentum=MOMENTUM,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 15. LEARNING RATE SCHEDULER
# ============================================================

scheduler = (
    torch.optim.lr_scheduler.StepLR(

        optimizer,

        step_size=10,

        gamma=0.1
    )
)


# ============================================================
# 16. mAP METRIC
# ============================================================

map_metric = MeanAveragePrecision(

    box_format="xyxy",

    iou_type="bbox",

    class_metrics=True
)


# ============================================================
# 17. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0


    for batch_index, (
        images,
        targets
    ) in enumerate(loader):


        # ----------------------------------------------------
        # Move images to GPU
        # ----------------------------------------------------

        images = [

            image.to(device)

            for image in images
        ]


        # ----------------------------------------------------
        # Move targets to GPU
        # ----------------------------------------------------

        targets = [

            {
                key: value.to(device)

                for key, value
                in target.items()
            }

            for target in targets
        ]


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        loss_dict = model(
            images,
            targets
        )


        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

        losses = sum(
            loss
            for loss
            in loss_dict.values()
        )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        losses.backward()

        optimizer.step()


        total_loss += (
            losses.item()
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_index + 1
        ) % 20 == 0:

            print(

                f"  Batch "
                f"{batch_index + 1}/"
                f"{len(loader)} "
                f"Loss: "
                f"{losses.item():.4f}"
            )


    average_loss = (
        total_loss
        /
        len(loader)
    )


    return average_loss


# ============================================================
# 18. VALIDATION FUNCTION
# ============================================================

def validate(
    model,
    loader,
    device
):

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Faster R-CNN needs eval mode during inference.
    # --------------------------------------------------------

    model.eval()


    map_metric.reset()


    total_ground_truth = 0

    total_predictions = 0


    with torch.no_grad():

        for images, targets in loader:


            # ------------------------------------------------
            # Move images
            # ------------------------------------------------

            images = [

                image.to(device)

                for image in images
            ]


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            predictions = model(
                images
            )


            # ------------------------------------------------
            # Prepare predictions
            # ------------------------------------------------

            prediction_list = []

            target_list = []


            for prediction, target in zip(
                predictions,
                targets
            ):


                # --------------------------------------------
                # Move prediction to CPU
                # --------------------------------------------

                prediction = {

                    key: value.cpu()

                    for key, value
                    in prediction.items()
                }


                # --------------------------------------------
                # Confidence filtering
                # --------------------------------------------

                keep = (
                    prediction["scores"]
                    >= CONFIDENCE_THRESHOLD
                )


                prediction = {

                    "boxes":
                        prediction["boxes"][keep],

                    "scores":
                        prediction["scores"][keep],

                    "labels":
                        prediction["labels"][keep]
                }


                target = {

                    "boxes":
                        target["boxes"].cpu(),

                    "labels":
                        target["labels"].cpu()
                }


                prediction_list.append(
                    prediction
                )

                target_list.append(
                    target
                )


                total_predictions += len(
                    prediction["boxes"]
                )


                total_ground_truth += len(
                    target["boxes"]
                )


            # --------------------------------------------
            # Update mAP
            # --------------------------------------------

            map_metric.update(
                prediction_list,
                target_list
            )


    # ========================================================
    # COMPUTE METRICS
    # ========================================================

    metrics = map_metric.compute()


    map50 = metrics[
        "map_50"
    ].item()


    map5095 = metrics[
        "map"
    ].item()


    # --------------------------------------------------------
    # Precision / Recall
    #
    # TorchMetrics provides detailed statistics.
    # --------------------------------------------------------

    precision = 0.0

    recall = 0.0


    if (
        "precision"
        in metrics
    ):

        precision_tensor = (
            metrics["precision"]
        )


        if precision_tensor.numel() > 0:

            valid_precision = (
                precision_tensor[
                    precision_tensor >= 0
                ]
            )


            if (
                valid_precision.numel()
                > 0
            ):

                precision = (
                    valid_precision
                    .mean()
                    .item()
                )


    if (
        "recall"
        in metrics
    ):

        recall_tensor = (
            metrics["recall"]
        )


        if recall_tensor.numel() > 0:

            valid_recall = (
                recall_tensor[
                    recall_tensor >= 0
                ]
            )


            if (
                valid_recall.numel()
                > 0
            ):

                recall = (
                    valid_recall
                    .mean()
                    .item()
                )


    return {

        "precision":
            precision,

        "recall":
            recall,

        "mAP50":
            map50,

        "mAP50-95":
            map5095,

        "ground_truth":
            total_ground_truth,

        "predictions":
            total_predictions
    }


# ============================================================
# 19. TRAINING
# ============================================================

print("\n")
print("=" * 60)
print("START TRAINING")
print("=" * 60)


history = []


best_map50 = -1


for epoch in range(
    EPOCHS
):


    print("\n")
    print(
        f"Epoch "
        f"{epoch + 1}/{EPOCHS}"
    )

    print(
        "-" * 50
    )


    # ========================================================
    # TRAIN
    # ========================================================

    train_loss = train_one_epoch(

        model,

        train_loader,

        optimizer,

        DEVICE
    )


    # ========================================================
    # VALIDATE
    # ========================================================

    validation_results = validate(

        model,

        valid_loader,

        DEVICE
    )


    precision = (
        validation_results[
            "precision"
        ]
    )


    recall = (
        validation_results[
            "recall"
        ]
    )


    map50 = (
        validation_results[
            "mAP50"
        ]
    )


    map5095 = (
        validation_results[
            "mAP50-95"
        ]
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print(
        "Training Loss:",
        f"{train_loss:.4f}"
    )

    print(
        "Precision:",
        f"{precision:.4f}"
    )

    print(
        "Recall:",
        f"{recall:.4f}"
    )

    print(
        "mAP50:",
        f"{map50:.4f}"
    )

    print(
        "mAP50-95:",
        f"{map5095:.4f}"
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    history.append({

        "epoch":
            epoch + 1,

        "train_loss":
            train_loss,

        "precision":
            precision,

        "recall":
            recall,

        "mAP50":
            map50,

        "mAP50-95":
            map5095
    })


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if map50 > best_map50:

        best_map50 = map50


        best_model_path = os.path.join(

            RESULT_FOLDER,

            "best_model.pth"
        )


        torch.save({

            "epoch":
                epoch + 1,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "mAP50":
                map50,

            "mAP50-95":
                map5095,

            "classes":
                names

        }, best_model_path)


        print(
            "\n✓ Best model saved!"
        )

        print(
            "Best mAP50:",
            f"{best_map50:.4f}"
        )


    # ========================================================
    # UPDATE LEARNING RATE
    # ========================================================

    scheduler.step()


# ============================================================
# 20. SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(

    RESULT_FOLDER,

    "final_model.pth"
)


torch.save({

    "model_state_dict":
        model.state_dict(),

    "classes":
        names,

    "num_classes":
        NUM_MODEL_CLASSES

}, final_model_path)


print("\n")
print(
    "Final model saved:"
)

print(
    final_model_path
)


# ============================================================
# 21. SAVE RESULTS TO CSV
# ============================================================

results_df = pd.DataFrame(
    history
)


csv_path = os.path.join(

    RESULT_FOLDER,

    "training_results.csv"
)


results_df.to_csv(
    csv_path,
    index=False
)


print(
    "\nResults saved:"
)

print(
    csv_path
)


# ============================================================
# 22. PLOT TRAINING LOSS
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    results_df["epoch"],

    results_df["train_loss"],

    marker="o"
)


plt.title(
    "Faster R-CNN Training Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Training Loss"
)


plt.grid(
    True
)


loss_plot_path = os.path.join(

    RESULT_FOLDER,

    "training_loss.png"
)


plt.savefig(
    loss_plot_path,
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# 23. PLOT mAP
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    results_df["epoch"],

    results_df["mAP50"],

    marker="o",

    label="mAP50"
)


plt.plot(

    results_df["epoch"],

    results_df["mAP50-95"],

    marker="o",

    label="mAP50-95"
)


plt.title(
    "Faster R-CNN Validation mAP"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "mAP"
)


plt.legend()


plt.grid(
    True
)


map_plot_path = os.path.join(

    RESULT_FOLDER,

    "validation_map.png"
)


plt.savefig(

    map_plot_path,

    dpi=300,

    bbox_inches="tight"
)


plt.show()


# ============================================================
# 24. PLOT PRECISION AND RECALL
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    results_df["epoch"],

    results_df["precision"],

    marker="o",

    label="Precision"
)


plt.plot(

    results_df["epoch"],

    results_df["recall"],

    marker="o",

    label="Recall"
)


plt.title(
    "Faster R-CNN Precision and Recall"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Score"
)


plt.legend()


plt.grid(
    True
)


pr_plot_path = os.path.join(

    RESULT_FOLDER,

    "precision_recall.png"
)


plt.savefig(

    pr_plot_path,

    dpi=300,

    bbox_inches="tight"
)


plt.show()


# ============================================================
# 25. FINAL RESULTS
# ============================================================

best_epoch = results_df.loc[
    results_df["mAP50"].idxmax()
]


print("\n")
print("=" * 60)
print("FINAL BASELINE FASTER R-CNN RESULTS")
print("=" * 60)


print(
    "Best Epoch:",
    int(best_epoch["epoch"])
)


print(
    "Precision:",
    f"{best_epoch['precision']:.4f}"
)


print(
    "Recall:",
    f"{best_epoch['recall']:.4f}"
)


print(
    "mAP50:",
    f"{best_epoch['mAP50']:.4f}"
)


print(
    "mAP50-95:",
    f"{best_epoch['mAP50-95']:.4f}"
)


print("\n")
print(
    "Results folder:"
)

print(
    RESULT_FOLDER
)


print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)