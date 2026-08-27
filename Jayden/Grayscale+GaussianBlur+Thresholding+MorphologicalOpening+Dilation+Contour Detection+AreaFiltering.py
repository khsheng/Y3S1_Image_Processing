# ============================================================
# EXPERIMENT 1
# GRAYSCALE → GAUSSIAN BLUR → THRESHOLDING
# → MORPHOLOGICAL OPENING → DILATION
# → CONTOUR DETECTION → AREA FILTERING → BOUNDING BOXES
# → YOLOv8m
# ============================================================

import os
import cv2
import shutil
import numpy as np
from ultralytics import YOLO


# ============================================================
# 1. ORIGINAL DATASET
# ============================================================

DATASET_ROOT = os.path.join(
    os.getcwd(),
    "pcb-defect-dataset"
)

DATA_YAML = os.path.join(
    DATASET_ROOT,
    "data.yaml"
)


print("\n==============================================")
print("PCB DATASET")
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
# 3. PREPROCESSED DATASET
# ============================================================

PREPROCESSED_ROOT = os.path.join(
    os.getcwd(),
    "pcb-defect-dataset-Grayscale+GaussianBlur+Thresholding+MorphologicalOpening+Dilation+ContourDetection+AreaFiltering"
)

print("\n==============================================")
print("PREPROCESSED DATASET")
print("==============================================")

print(
    "Output:",
    PREPROCESSED_ROOT
)


# ============================================================
# 4. IMAGE PREPROCESSING FUNCTION
# ============================================================

def preprocess_image(image):
    """
    PCB Image
        ↓
    Grayscale
        ↓
    Gaussian Blur
        ↓
    Thresholding
        ↓
    Morphological Opening
        ↓
    Dilation
        ↓
    Contour Detection
        ↓
    Area Filtering
        ↓
    Bounding Boxes
    """

    # --------------------------------------------------------
    # STEP 1 — GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # STEP 2 — GAUSSIAN BLUR
    # --------------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    # --------------------------------------------------------
    # STEP 3 — THRESHOLDING
    # --------------------------------------------------------

    _, threshold = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )


    # --------------------------------------------------------
    # STEP 4 — MORPHOLOGICAL OPENING
    # --------------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    opened = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )


    # --------------------------------------------------------
    # STEP 5 — DILATION
    # --------------------------------------------------------

    dilated = cv2.dilate(
        opened,
        kernel,
        iterations=1
    )


    # --------------------------------------------------------
    # STEP 6 — CONTOUR DETECTION
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # --------------------------------------------------------
    # STEP 7 — AREA FILTERING
    # --------------------------------------------------------

    boxes = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore very small regions
        if area < 20:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        boxes.append(
            (x, y, w, h, area)
        )


    # --------------------------------------------------------
    # STEP 8 — DRAW BOUNDING BOXES
    # --------------------------------------------------------

    result = image.copy()

    for x, y, w, h, area in boxes:

        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


    return result, boxes


# ============================================================
# 5. PROCESS DATASET IMAGES
# ============================================================

def process_dataset_split(split):

    input_images = os.path.join(
        DATASET_ROOT,
        split,
        "images"
    )

    output_images = os.path.join(
        PREPROCESSED_ROOT,
        split,
        "images"
    )

    input_labels = os.path.join(
        DATASET_ROOT,
        split,
        "labels"
    )

    output_labels = os.path.join(
        PREPROCESSED_ROOT,
        split,
        "labels"
    )


    os.makedirs(
        output_images,
        exist_ok=True
    )

    os.makedirs(
        output_labels,
        exist_ok=True
    )


    if not os.path.exists(input_images):

        print(
            f"WARNING: {input_images} does not exist."
        )

        return


    image_files = [
        f for f in os.listdir(input_images)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        )
    ]


    print(
        f"\nProcessing {split}: "
        f"{len(image_files)} images"
    )


    for filename in image_files:

        input_path = os.path.join(
            input_images,
            filename
        )

        output_path = os.path.join(
            output_images,
            filename
        )


        # Read image
        image = cv2.imread(
            input_path
        )


        if image is None:

            print(
                f"WARNING: Cannot read {filename}"
            )

            continue


        # Apply preprocessing
        processed_image, boxes = preprocess_image(
            image
        )


        # Save processed image
        cv2.imwrite(
            output_path,
            processed_image
        )


        # ----------------------------------------------------
        # IMPORTANT:
        # Copy original YOLO labels.
        #
        # We are NOT replacing YOLO ground-truth labels
        # with contour-generated boxes.
        # ----------------------------------------------------

        label_filename = os.path.splitext(
            filename
        )[0] + ".txt"

        input_label_path = os.path.join(
            input_labels,
            label_filename
        )

        output_label_path = os.path.join(
            output_labels,
            label_filename
        )


        if os.path.exists(input_label_path):

            shutil.copy2(
                input_label_path,
                output_label_path
            )


    print(
        f"Completed: {split}"
    )


# ============================================================
# 6. PROCESS TRAIN / VAL / TEST
# ============================================================

for split in [
    "train",
    "val",
    "test"
]:

    process_dataset_split(
        split
    )


# ============================================================
# 7. CREATE DATA.YAML
# ============================================================

import yaml


with open(
    DATA_YAML,
    "r"
) as f:

    data_config = yaml.safe_load(f)


new_data_yaml = os.path.join(
    PREPROCESSED_ROOT,
    "data.yaml"
)


data_config["path"] = PREPROCESSED_ROOT

data_config["train"] = "train/images"

data_config["val"] = "val/images"

data_config["test"] = "test/images"


with open(
    new_data_yaml,
    "w"
) as f:

    yaml.dump(
        data_config,
        f,
        sort_keys=False
    )


print("\n==============================================")
print("PREPROCESSING COMPLETED")
print("==============================================")

print(
    "Preprocessed dataset:",
    PREPROCESSED_ROOT
)

print(
    "New data.yaml:",
    new_data_yaml
)
