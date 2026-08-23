import os
import cv2
import numpy as np
import shutil
import yaml
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ============================================================
# DATASET PATH
# ============================================================

DATASET_ROOT = os.path.join(os.getcwd(), "pcb-defect-dataset")

print("Dataset exists:", os.path.exists(DATASET_ROOT))

# ============================================================
# EXPERIMENT 3 — SHARPENING
# ============================================================


# ============================================================
# 1. SHARPENING DATASET PATH
# ============================================================

SHARPEN_DATASET_ROOT = "./pcb-defect-dataset-sharpening"


# ============================================================
# 2. CREATE TRAIN / VAL / TEST FOLDERS
# ============================================================

for split in [
    "train",
    "val",
    "test"
]:

    os.makedirs(
        os.path.join(
            SHARPEN_DATASET_ROOT,
            split,
            "images"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            SHARPEN_DATASET_ROOT,
            split,
            "labels"
        ),
        exist_ok=True
    )


print("\n==============================================")
print("SHARPENING DATASET")
print("==============================================")

print(
    "Dataset:",
    SHARPEN_DATASET_ROOT
)


# ============================================================
# 3. SHARPENING FUNCTION
# ============================================================

def apply_sharpening(image):

    kernel = np.array([
        [0, -1,  0],
        [-1, 5, -1],
        [0, -1,  0]
    ])

    sharpened = cv2.filter2D(
        image,
        -1,
        kernel
    )

    return sharpened


# ============================================================
# 4. PREPROCESS EACH SPLIT
# ============================================================

def preprocess_sharpening_split(split):

    source_images = os.path.join(
        DATASET_ROOT,
        split,
        "images"
    )

    source_labels = os.path.join(
        DATASET_ROOT,
        split,
        "labels"
    )

    output_images = os.path.join(
        SHARPEN_DATASET_ROOT,
        split,
        "images"
    )

    output_labels = os.path.join(
        SHARPEN_DATASET_ROOT,
        split,
        "labels"
    )

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )

    image_files = [
        f
        for f in os.listdir(source_images)
        if f.lower().endswith(
            image_extensions
        )
    ]

    processed = 0

    for filename in image_files:

        # ----------------------------------------------------
        # Read original image
        # ----------------------------------------------------

        input_path = os.path.join(
            source_images,
            filename
        )

        image = cv2.imread(
            input_path
        )

        if image is None:

            print(
                "Could not read:",
                filename
            )

            continue


        # ----------------------------------------------------
        # Apply sharpening
        # ----------------------------------------------------

        sharpened_image = apply_sharpening(
            image
        )


        # ----------------------------------------------------
        # Save sharpened image
        # ----------------------------------------------------

        output_path = os.path.join(
            output_images,
            filename
        )

        cv2.imwrite(
            output_path,
            sharpened_image
        )


        # ----------------------------------------------------
        # Copy YOLO label
        # ----------------------------------------------------

        label_filename = (
            os.path.splitext(filename)[0]
            + ".txt"
        )

        source_label = os.path.join(
            source_labels,
            label_filename
        )

        output_label = os.path.join(
            output_labels,
            label_filename
        )

        if os.path.exists(source_label):

            shutil.copy2(
                source_label,
                output_label
            )

        processed += 1


    print(
        f"{split}: {processed} images processed"
    )


# ============================================================
# 5. APPLY SHARPENING TO TRAIN / VAL / TEST
# ============================================================

print("\n==============================================")
print("STARTING SHARPENING PREPROCESSING")
print("==============================================")


for split in [
    "train",
    "val",
    "test"
]:

    print(
        f"\nProcessing {split.upper()}..."
    )

    preprocess_sharpening_split(
        split
    )


print("\nSharpening preprocessing completed.")


# ============================================================
# 6. CREATE SHARPENING DATA.YAML
# ============================================================

SHARPEN_DATA_YAML = os.path.join(
    SHARPEN_DATASET_ROOT,
    "data.yaml"
)


sharpen_data = {

    "path": SHARPEN_DATASET_ROOT,

    "train": "train",

    "val": "val",

    "test": "test",

    "names": {

        0: "mouse_bite",

        1: "spur",

        2: "missing_hole",

        3: "short",

        4: "open_circuit",

        5: "spurious_copper"

    }
}


with open(
    SHARPEN_DATA_YAML,
    "w"
) as f:

    yaml.dump(
        sharpen_data,
        f,
        sort_keys=False
    )


print("\n==============================================")
print("SHARPENING DATA.YAML")
print("==============================================")

print(
    SHARPEN_DATA_YAML
)

print()

with open(
    SHARPEN_DATA_YAML,
    "r"
) as f:

    print(
        f.read()
    )

# ============================================================
# 7. DISPLAY ORIGINAL VS SHARPENED
# ============================================================
TRAIN_IMAGES = os.path.join(DATASET_ROOT, "train", "images")
sample_files = []

for filename in os.listdir(
    TRAIN_IMAGES
):

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    ):

        sample_files.append(
            filename
        )

        if len(sample_files) >= 3:
            break


print("\n==============================================")
print("ORIGINAL VS SHARPENING")
print("==============================================")


for filename in sample_files:

    original_path = os.path.join(
        TRAIN_IMAGES,
        filename
    )

    sharpen_path = os.path.join(
        SHARPEN_DATASET_ROOT,
        "train",
        "images",
        filename
    )


    original = cv2.imread(
        original_path
    )

    sharpened = cv2.imread(
        sharpen_path
    )


    original = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    sharpened = cv2.cvtColor(
        sharpened,
        cv2.COLOR_BGR2RGB
    )


    plt.figure(
        figsize=(12, 5)
    )


    plt.subplot(
        1,
        2,
        1
    )

    plt.imshow(
        original
    )

    plt.title(
        "Original"
    )

    plt.axis(
        "off"
    )


    plt.subplot(
        1,
        2,
        2
    )

    plt.imshow(
        sharpened
    )

    plt.title(
        "Sharpened"
    )

    plt.axis(
        "off"
    )


    plt.tight_layout()

    plt.show()

# ============================================================
# 8. TRAIN YOLOv8m
# ============================================================

print("\n==============================================")
print("TRAINING SHARPENING YOLOv8m")
print("==============================================")


model_sharpening = YOLO(
    "yolov8m.pt"
)


SHARPEN_RESULTS_DIR = (
    "./pcb-yolo-results-sharpening"
)


model_sharpening.train(

    data=SHARPEN_DATA_YAML,

    epochs=100,

    imgsz=640,

    batch=16,

    pretrained=True,

    project=SHARPEN_RESULTS_DIR,

    name="sharpening",

    exist_ok=True,

    patience=20,

    save=True,

    plots=True

)