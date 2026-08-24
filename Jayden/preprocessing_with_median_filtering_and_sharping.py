import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import yaml

from ultralytics import YOLO


DATASET_ROOT = os.path.join(os.getcwd(), "pcb-defect-dataset")

DATA_YAML = os.path.join(
    DATASET_ROOT,
    "data.yaml"
)

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
        if file.lower().endswith(valid_extensions):
            files.append(
                os.path.join(folder, file)
            )
    return sorted(files)


train_images = get_image_files(TRAIN_IMAGES)
val_images = get_image_files(VAL_IMAGES)
test_images = get_image_files(TEST_IMAGES)


# ============================================================
# 7. COUNT IMAGES
# ============================================================

print("\n==============================================")
print("IMAGE COUNT")
print("==============================================")

print("Training images   :", len(train_images))
print("Validation images :", len(val_images))
print("Testing images    :", len(test_images))


# ============================================================
# 8. GET LABEL FILES
# ============================================================

def get_label_files(folder):
    if not os.path.exists(folder):
        return []
    files = []
    for file in os.listdir(folder):
        if file.lower().endswith(".txt"):
            files.append(
                os.path.join(folder, file)
            )
    return sorted(files)


train_labels = get_label_files(TRAIN_LABELS)
val_labels = get_label_files(VAL_LABELS)
test_labels = get_label_files(TEST_LABELS)


print("\n==============================================")
print("LABEL COUNT")
print("==============================================")

print("Training labels   :", len(train_labels))
print("Validation labels :", len(val_labels))
print("Testing labels    :", len(test_labels))


# ============================================================
# 9. APPLY COMBINED MEDIAN FILTER + SHARPENING FUNCTION
# ============================================================

def apply_median_and_sharpen(
    image,
    kernel_size=3,
    alpha=1.5
):
    """
    1. Apply Median Filter to remove impulse noise.
    2. Apply a Sharpening filter using a 2D convolution kernel to enhance edge details.
    """
    # Step 1: Median Filtering
    median_filtered = cv2.medianBlur(image, kernel_size)

    # Step 2: Sharpening Kernel
    # A standard 3x3 sharpening kernel
    kernel = np.array([
        [ 0, -1,  0],
        [-1,  4 + alpha, -1],
        [ 0, -1,  0]
    ])
    
    # Alternatively, a stronger laplacian-based sharpen:
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    sharpened = cv2.filter2D(median_filtered, -1, kernel)

    return sharpened


# ============================================================
# 10. CREATE COMBINED DATASET DIRECTORY
# ============================================================

COMBINED_DATASET_ROOT = os.path.join(os.getcwd(), "pcb-defect-dataset-median-sharpen")


for split in [
    "train",
    "val",
    "test"
]:
    os.makedirs(
        os.path.join(
            COMBINED_DATASET_ROOT,
            split,
            "images"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            COMBINED_DATASET_ROOT,
            split,
            "labels"
        ),
        exist_ok=True
    )


print("\n==============================================")
print("MEDIAN + SHARPEN DATASET")
print("==============================================")

print("Combined dataset:", COMBINED_DATASET_ROOT)


# ============================================================
# 11. PREPROCESS EACH SPLIT
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
        COMBINED_DATASET_ROOT,
        split,
        "images"
    )
    output_labels = os.path.join(
        COMBINED_DATASET_ROOT,
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
        if f.lower().endswith(image_extensions)
    ]

    processed = 0

    for filename in image_files:
        input_path = os.path.join(
            source_images,
            filename
        )
        image = cv2.imread(input_path)

        if image is None:
            print("Could not read:", filename)
            continue

        # Apply Median Filter + Sharpening
        processed_image = apply_median_and_sharpen(
            image,
            kernel_size=3,
            alpha=1.0
        )

        # Save processed image
        output_path = os.path.join(
            output_images,
            filename
        )
        cv2.imwrite(
            output_path,
            processed_image
        )

        # Copy YOLO label
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
# 12. PROCESS TRAIN / VAL / TEST
# ============================================================

print("\n==============================================")
print("STARTING MEDIAN + SHARPEN PREPROCESSING")
print("==============================================")


for split in [
    "train",
    "val",
    "test"
]:
    print(f"\nProcessing {split.upper()}...")
    preprocess_split(split)


print("\nMedian + Sharpen preprocessing completed.")


# ============================================================
# 13. CREATE COMBINED DATA.YAML
# ============================================================

COMBINED_DATA_YAML = os.path.join(
    COMBINED_DATASET_ROOT,
    "data.yaml"
)


combined_data = {
    "path": COMBINED_DATASET_ROOT,
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
    COMBINED_DATA_YAML,
    "w"
) as f:
    yaml.dump(
        combined_data,
        f,
        sort_keys=False
    )


print("\n==============================================")
print("MEDIAN + SHARPEN DATA.YAML")
print("==============================================")

print(COMBINED_DATA_YAML)
print()

with open(
    COMBINED_DATA_YAML,
    "r"
) as f:
    print(f.read())


# ============================================================
# 14. DISPLAY ORIGINAL VS MEDIAN + SHARPEN
# ============================================================

sample_files = []

for filename in os.listdir(TRAIN_IMAGES):
    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    ):
        sample_files.append(filename)
        if len(sample_files) >= 3:
            break


print("\n==============================================")
print("ORIGINAL VS MEDIAN + SHARPEN")
print("==============================================")


for filename in sample_files:
    original_path = os.path.join(
        TRAIN_IMAGES,
        filename
    )
    processed_path = os.path.join(
        COMBINED_DATASET_ROOT,
        "train",
        "images",
        filename
    )

    original = cv2.imread(original_path)
    processed_image = cv2.imread(processed_path)

    original = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )
    processed_image = cv2.cvtColor(
        processed_image,
        cv2.COLOR_BGR2RGB
    )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(processed_image)
    plt.title("Median + Sharpen")
    plt.axis("off")

    plt.tight_layout()
    plt.show()