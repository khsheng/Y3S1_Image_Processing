# ============================================================
# EXPERIMENT 3 — YOLOv8m
# MEDIAN K=9 + CLAHE + MILD UNSHARP + EDGE ENHANCEMENT
# ============================================================

import os
import cv2
import shutil
import yaml
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
print("ORIGINAL DATASET")
print("==============================================")

print("Dataset:", DATASET_ROOT)
print("Dataset exists:", os.path.exists(DATASET_ROOT))
print("data.yaml:", DATA_YAML)
print("data.yaml exists:", os.path.exists(DATA_YAML))


if not os.path.exists(DATASET_ROOT):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_ROOT}"
    )

if not os.path.exists(DATA_YAML):
    raise FileNotFoundError(
        f"data.yaml not found: {DATA_YAML}"
    )


# ============================================================
# 2. E3 ENHANCED DATASET
# ============================================================

ENHANCED_ROOT = os.path.join(
    os.getcwd(),
    "pcb-defect-dataset-median_filtering_(K=9)_and_CLAHE+MildUnsharp+EdgeEnhancement"
)

print("\n==============================================")
print("E3 ENHANCED DATASET")
print("==============================================")

print("Enhanced dataset:", ENHANCED_ROOT)


# ============================================================
# 3. MILD UNSHARP MASK
# ============================================================

def mild_unsharp(
    image,
    sigma=1.0,
    amount=0.4
):

    # Slight Gaussian blur
    blurred = cv2.GaussianBlur(
        image,
        (0, 0),
        sigma
    )

    # Unsharp mask
    enhanced = cv2.addWeighted(
        image,
        1.0 + amount,
        blurred,
        -amount,
        0
    )

    return enhanced


# ============================================================
# 4. EDGE ENHANCEMENT
# ============================================================

def edge_enhancement(
    image,
    strength=0.25
):

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Detect edges using Laplacian
    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F,
        ksize=3
    )

    # Convert to 8-bit
    laplacian = cv2.convertScaleAbs(
        laplacian
    )

    # Convert edge map to 3 channels
    edges = cv2.cvtColor(
        laplacian,
        cv2.COLOR_GRAY2BGR
    )

    # Add edges to image
    enhanced = cv2.addWeighted(
        image,
        1.0,
        edges,
        strength,
        0
    )

    return enhanced


# ============================================================
# 5. COMPLETE E3 IMAGE ENHANCEMENT PIPELINE
# ============================================================

def enhance_image(image):

    # --------------------------------------------------------
    # STEP 1 — MEDIAN FILTER K=9
    # --------------------------------------------------------

    median = cv2.medianBlur(
        image,
        9
    )


    # --------------------------------------------------------
    # STEP 2 — CLAHE
    # Apply CLAHE to LAB L-channel
    # --------------------------------------------------------

    lab = cv2.cvtColor(
        median,
        cv2.COLOR_BGR2LAB
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_enhanced = clahe.apply(
        l_channel
    )

    lab_enhanced = cv2.merge(
        [
            l_enhanced,
            a_channel,
            b_channel
        ]
    )

    clahe_image = cv2.cvtColor(
        lab_enhanced,
        cv2.COLOR_LAB2BGR
    )


    # --------------------------------------------------------
    # STEP 3 — MILD UNSHARP
    # --------------------------------------------------------

    unsharp_image = mild_unsharp(
        clahe_image,
        sigma=1.0,
        amount=0.4
    )


    # --------------------------------------------------------
    # STEP 4 — EDGE ENHANCEMENT
    # --------------------------------------------------------

    enhanced = edge_enhancement(
        unsharp_image,
        strength=0.25
    )


    return enhanced


# ============================================================
# 6. CREATE E3 ENHANCED DATASET
# ============================================================

for split in [
    "train",
    "val",
    "test"
]:

    original_images = os.path.join(
        DATASET_ROOT,
        split,
        "images"
    )

    original_labels = os.path.join(
        DATASET_ROOT,
        split,
        "labels"
    )

    enhanced_images = os.path.join(
        ENHANCED_ROOT,
        split,
        "images"
    )

    enhanced_labels = os.path.join(
        ENHANCED_ROOT,
        split,
        "labels"
    )


    # Create directories

    os.makedirs(
        enhanced_images,
        exist_ok=True
    )

    os.makedirs(
        enhanced_labels,
        exist_ok=True
    )


    # --------------------------------------------------------
    # PROCESS IMAGES
    # --------------------------------------------------------

    if os.path.exists(
        original_images
    ):

        image_files = [
            f
            for f in os.listdir(
                original_images
            )
            if f.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp"
                )
            )
        ]

        print(
            f"\n{split.upper()} images: "
            f"{len(image_files)}"
        )


        for filename in image_files:

            input_path = os.path.join(
                original_images,
                filename
            )

            output_path = os.path.join(
                enhanced_images,
                filename
            )


            image = cv2.imread(
                input_path
            )


            if image is None:

                print(
                    "WARNING: Could not read:",
                    input_path
                )

                continue


            # Apply E3 pipeline

            enhanced = enhance_image(
                image
            )


            # Save enhanced image

            cv2.imwrite(
                output_path,
                enhanced
            )


    # --------------------------------------------------------
    # COPY LABELS
    # Labels remain unchanged
    # --------------------------------------------------------

    if os.path.exists(
        original_labels
    ):

        label_files = [
            f
            for f in os.listdir(
                original_labels
            )
            if f.lower().endswith(
                ".txt"
            )
        ]


        for filename in label_files:

            source_label = os.path.join(
                original_labels,
                filename
            )

            destination_label = os.path.join(
                enhanced_labels,
                filename
            )


            shutil.copy2(
                source_label,
                destination_label
            )


# ============================================================
# 7. CREATE NEW data.yaml
# ============================================================

with open(
    DATA_YAML,
    "r"
) as f:

    original_yaml = yaml.safe_load(
        f
    )


enhanced_yaml = original_yaml.copy()

enhanced_yaml["path"] = ENHANCED_ROOT


ENHANCED_YAML = os.path.join(
    ENHANCED_ROOT,
    "data.yaml"
)


with open(
    ENHANCED_YAML,
    "w"
) as f:

    yaml.dump(
        enhanced_yaml,
        f,
        sort_keys=False
    )


print("\n==============================================")
print("E3 DATASET CREATED")
print("==============================================")

print(
    "Location:",
    ENHANCED_ROOT
)

print(
    "data.yaml:",
    ENHANCED_YAML
)