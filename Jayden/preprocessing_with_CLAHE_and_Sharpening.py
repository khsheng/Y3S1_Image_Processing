# ============================================================
# EXPERIMENT 4
# CLAHE → SHARPENING → YOLOv8m
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO


# ============================================================
# 1. DATASET PATH
# ============================================================

DATASET_ROOT = (
    "/content/drive/MyDrive/"
    "pcb-defect-dataset"
)


# ============================================================
# 2. HYBRID DATASET PATH
# ============================================================

HYBRID_DATASET_ROOT = (
    "/content/drive/MyDrive/"
    "pcb-defect-dataset-hybrid"
)


# ============================================================
# 3. DATA YAML
# ============================================================

ORIGINAL_DATA_YAML = os.path.join(
    DATASET_ROOT,
    "data.yaml"
)

HYBRID_DATA_YAML = os.path.join(
    HYBRID_DATASET_ROOT,
    "data.yaml"
)


# ============================================================
# 4. ORIGINAL DATASET PATHS
# ============================================================

TRAIN_IMAGES = os.path.join(
    DATASET_ROOT,
    "train",
    "images"
)

TRAIN_LABELS = os.path.join(
    DATASET_ROOT,
    "train",
    "labels"
)

VAL_IMAGES = os.path.join(
    DATASET_ROOT,
    "val",
    "images"
)

VAL_LABELS = os.path.join(
    DATASET_ROOT,
    "val",
    "labels"
)

TEST_IMAGES = os.path.join(
    DATASET_ROOT,
    "test",
    "images"
)

TEST_LABELS = os.path.join(
    DATASET_ROOT,
    "test",
    "labels"
)


# ============================================================
# 5. HYBRID DATASET PATHS
# ============================================================

HYBRID_TRAIN_IMAGES = os.path.join(
    HYBRID_DATASET_ROOT,
    "train",
    "images"
)

HYBRID_TRAIN_LABELS = os.path.join(
    HYBRID_DATASET_ROOT,
    "train",
    "labels"
)

HYBRID_VAL_IMAGES = os.path.join(
    HYBRID_DATASET_ROOT,
    "val",
    "images"
)

HYBRID_VAL_LABELS = os.path.join(
    HYBRID_DATASET_ROOT,
    "val",
    "labels"
)

HYBRID_TEST_IMAGES = os.path.join(
    HYBRID_DATASET_ROOT,
    "test",
    "images"
)

HYBRID_TEST_LABELS = os.path.join(
    HYBRID_DATASET_ROOT,
    "test",
    "labels"
)


# ============================================================
# 6. DATASET PATH CHECK
# ============================================================

paths = {

    "Original Dataset":
        DATASET_ROOT,

    "Original data.yaml":
        ORIGINAL_DATA_YAML,

    "Hybrid Dataset":
        HYBRID_DATASET_ROOT,

    "Hybrid data.yaml":
        HYBRID_DATA_YAML,

    "Train Images":
        TRAIN_IMAGES,

    "Train Labels":
        TRAIN_LABELS,

    "Validation Images":
        VAL_IMAGES,

    "Validation Labels":
        VAL_LABELS,

    "Test Images":
        TEST_IMAGES,

    "Test Labels":
        TEST_LABELS

}


print("\n==============================================")
print("DATASET PATH CHECK")
print("==============================================")


for name, path in paths.items():

    print(
        f"{name:25s}: {path}"
    )

    print(
        "Exists:",
        os.path.exists(path)
    )

    print()


# ============================================================
# 7. EXISTING data.yaml
# ============================================================

print("==============================================")
print("EXISTING data.yaml")
print("==============================================")


if os.path.exists(ORIGINAL_DATA_YAML):

    with open(
        ORIGINAL_DATA_YAML,
        "r"
    ) as f:

        print(
            f.read()
        )

else:

    raise FileNotFoundError(
        "Original data.yaml was not found!"
    )


# ============================================================
# 8. GET IMAGE FILES
# ============================================================

def get_image_files(folder):

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )

    files = []

    if not os.path.exists(folder):

        return files

    for file in os.listdir(folder):

        if file.lower().endswith(
            valid_extensions
        ):

            files.append(
                os.path.join(
                    folder,
                    file
                )
            )

    return sorted(files)


# ============================================================
# 9. GET LABEL FILES
# ============================================================

def get_label_files(folder):

    if not os.path.exists(folder):

        return []

    files = []

    for file in os.listdir(folder):

        if file.lower().endswith(
            ".txt"
        ):

            files.append(
                os.path.join(
                    folder,
                    file
                )
            )

    return sorted(files)


# ============================================================
# 10. COUNT ORIGINAL DATASET
# ============================================================

train_images = get_image_files(
    TRAIN_IMAGES
)

val_images = get_image_files(
    VAL_IMAGES
)

test_images = get_image_files(
    TEST_IMAGES
)


train_labels = get_label_files(
    TRAIN_LABELS
)

val_labels = get_label_files(
    VAL_LABELS
)

test_labels = get_label_files(
    TEST_LABELS
)


print("\n==============================================")
print("IMAGE COUNT")
print("==============================================")


print(
    "Training images   :",
    len(train_images)
)

print(
    "Validation images :",
    len(val_images)
)

print(
    "Testing images    :",
    len(test_images)
)


print("\n==============================================")
print("LABEL COUNT")
print("==============================================")


print(
    "Training labels   :",
    len(train_labels)
)

print(
    "Validation labels :",
    len(val_labels)
)

print(
    "Testing labels    :",
    len(test_labels)
)


# ============================================================
# 11. CLAHE + SHARPENING FUNCTION
# ============================================================

def clahe_sharpen(image):

    # --------------------------------------------------------
    # STEP 1: CLAHE
    # --------------------------------------------------------

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(
        l_channel
    )

    lab = cv2.merge(
        (
            l_channel,
            a_channel,
            b_channel
        )
    )

    clahe_image = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    # --------------------------------------------------------
    # STEP 2: SHARPENING
    # --------------------------------------------------------

    sharpening_kernel = np.array(
        [
            [0, -1,  0],
            [-1, 5, -1],
            [0, -1,  0]
        ]
    )

    sharpened_image = cv2.filter2D(
        clahe_image,
        -1,
        sharpening_kernel
    )

    return sharpened_image


# ============================================================
# 12. PREPROCESS DATASET
# ============================================================

def preprocess_split(split):

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
        HYBRID_DATASET_ROOT,
        split,
        "images"
    )

    output_labels = os.path.join(
        HYBRID_DATASET_ROOT,
        split,
        "labels"
    )


    # --------------------------------------------------------
    # CREATE OUTPUT FOLDERS
    # --------------------------------------------------------

    os.makedirs(
        output_images,
        exist_ok=True
    )

    os.makedirs(
        output_labels,
        exist_ok=True
    )


    # --------------------------------------------------------
    # IMAGE EXTENSIONS
    # --------------------------------------------------------

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )


    image_files = [

        f

        for f in os.listdir(
            source_images
        )

        if f.lower().endswith(
            image_extensions
        )

    ]


    print(
        f"\nProcessing {split.upper()}..."
    )

    print(
        "Images:",
        len(image_files)
    )


    # --------------------------------------------------------
    # PROCESS IMAGES
    # --------------------------------------------------------

    for i, filename in enumerate(
        image_files,
        start=1
    ):

        source_path = os.path.join(
            source_images,
            filename
        )

        output_path = os.path.join(
            output_images,
            filename
        )


        # Read image

        image = cv2.imread(
            source_path
        )


        if image is None:

            print(
                "WARNING: Could not read:",
                filename
            )

            continue


        # ----------------------------------------------------
        # CLAHE → SHARPENING
        # ----------------------------------------------------

        processed_image = clahe_sharpen(
            image
        )


        # ----------------------------------------------------
        # SAVE PROCESSED IMAGE
        # ----------------------------------------------------

        success = cv2.imwrite(
            output_path,
            processed_image
        )


        if not success:

            print(
                "WARNING: Could not save:",
                filename
            )


        if i % 100 == 0:

            print(
                f"Processed "
                f"{i}/{len(image_files)}"
            )


    # --------------------------------------------------------
    # COPY YOLO LABELS
    # --------------------------------------------------------

    if os.path.exists(
        source_labels
    ):

        label_files = [

            f

            for f in os.listdir(
                source_labels
            )

            if f.lower().endswith(
                ".txt"
            )

        ]


        for filename in label_files:

            source_label = os.path.join(
                source_labels,
                filename
            )

            output_label = os.path.join(
                output_labels,
                filename
            )


            # Copy label without changing it

            import shutil

            shutil.copy2(
                source_label,
                output_label
            )


        print(
            "Labels copied:",
            len(label_files)
        )


# ============================================================
# 13. START PREPROCESSING
# ============================================================

print("\n==============================================")
print("CLAHE + SHARPENING PREPROCESSING")
print("==============================================")


for split in [

    "train",
    "val",
    "test"

]:

    preprocess_split(
        split
    )


# ============================================================
# 14. CREATE HYBRID data.yaml
# ============================================================

print("\n==============================================")
print("CREATING HYBRID data.yaml")
print("==============================================")


yaml_content = f"""
path: {HYBRID_DATASET_ROOT}

train: train/images
val: val/images
test: test/images

names:
  0: mouse_bite
  1: spur
  2: missing_hole
  3: short
  4: open_circuit
  5: spurious_copper
"""


with open(
    HYBRID_DATA_YAML,
    "w"
) as f:

    f.write(
        yaml_content.strip()
    )


print(
    "Created:",
    HYBRID_DATA_YAML
)


print("\n==============================================")
print("HYBRID data.yaml")
print("==============================================")


with open(
    HYBRID_DATA_YAML,
    "r"
) as f:

    print(
        f.read()
    )


# ============================================================
# 15. COUNT HYBRID DATASET
# ============================================================

hybrid_train_images = get_image_files(
    HYBRID_TRAIN_IMAGES
)

hybrid_val_images = get_image_files(
    HYBRID_VAL_IMAGES
)

hybrid_test_images = get_image_files(
    HYBRID_TEST_IMAGES
)


hybrid_train_labels = get_label_files(
    HYBRID_TRAIN_LABELS
)

hybrid_val_labels = get_label_files(
    HYBRID_VAL_LABELS
)

hybrid_test_labels = get_label_files(
    HYBRID_TEST_LABELS
)


print("\n==============================================")
print("HYBRID DATASET COUNT")
print("==============================================")


print(
    "Training images   :",
    len(hybrid_train_images)
)

print(
    "Validation images :",
    len(hybrid_val_images)
)

print(
    "Testing images    :",
    len(hybrid_test_images)
)


print(
    "\nTraining labels   :",
    len(hybrid_train_labels)
)

print(
    "Validation labels :",
    len(hybrid_val_labels)
)

print(
    "Testing labels    :",
    len(hybrid_test_labels)
)


# ============================================================
# 16. WINDOWS / COLAB EXECUTION
# ============================================================

if __name__ == "__main__":


    # ========================================================
    # 17. LOAD YOLOv8m
    # ========================================================

    print("\n==============================================")
    print("LOADING YOLOv8m")
    print("==============================================")


    model = YOLO(
        "yolov8m.pt"
    )


    # ========================================================
    # 18. TRAIN HYBRID MODEL
    # ========================================================

    print("\n==============================================")
    print("TRAINING CLAHE + SHARPENING YOLOv8m")
    print("==============================================")


    RESULTS_DIR = (
        "./pcb-yolo-results-clahe-sharpening"
    )


    results = model.train(

        data=HYBRID_DATA_YAML,

        epochs=100,

        imgsz=640,

        batch=16,

        pretrained=True,

        project=RESULTS_DIR,

        name="clahe_sharpening",

        exist_ok=True,

        patience=20,

        save=True,

        plots=True,

        workers=8,

        device=0,

        cache=True

    )