from KHS.function.preprocessing_function import (
    clahe_rgb,
    grayscale,
    average_filter,
    median_filter,
    median_filter,
    morphological_erosion,
    clahe,
    morphological_opening
)
import cv2
import shutil
from pathlib import Path

# =========================================================
# 1. PATH CONFIGURATION
# =========================================================

INPUT_DATASET = Path.cwd() / "pcb-defect-dataset"

OUTPUT_DATASET = Path.cwd() / "pcb-defect-testing"

# =========================================================
# 2. PREPROCESS IMAGE
# =========================================================

def preprocess_image(image):

    # median = median_filter(image)

    # opened = morphological_opening(median)

    clahe_image = clahe_rgb(image)

    return (
        clahe_image
    )


# =========================================================
# 3. PROCESS ONE IMAGE
# =========================================================

def process_image(input_path, output_path):

    image = cv2.imread(str(input_path))

    if image is None:
        print(f"WARNING: Cannot load image: {input_path}")
        return False

    # Preprocess
    processed = preprocess_image(image)

    # Save processed image
    success = cv2.imwrite(
        str(output_path),
        processed
    )

    if not success:
        print(f"WARNING: Cannot save image: {output_path}")
        return False

    return True


# =========================================================
# 4. COPY LABEL
# =========================================================

def copy_label(input_label, output_label):
    
    if input_label.exists():

        shutil.copy2(
            input_label,
            output_label
        )

        return True

    return False


# =========================================================
# 5. PROCESS DATASET SPLIT
# =========================================================

def process_dataset_split(split):

    input_images = INPUT_DATASET / split / "images"
    input_labels = INPUT_DATASET / split / "labels"

    output_images = OUTPUT_DATASET / split / "images"
    output_labels = OUTPUT_DATASET / split / "labels"

    # Create output folders
    output_images.mkdir(
        parents=True,
        exist_ok=True
    )

    output_labels.mkdir(
        parents=True,
        exist_ok=True
    )

    # Get all image files
    image_files = []

    for extension in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:

        image_files.extend(
            input_images.glob(extension)
        )

    print()
    print("=" * 60)
    print(f"Processing: {split}")
    print(f"Images found: {len(image_files)}")
    print("=" * 60)

    processed_count = 0
    failed_count = 0

    # Process every image
    for index, image_path in enumerate(image_files, start=1):

        # Output image path
        output_image = (
            output_images / image_path.name
        )

        # Process image
        success = process_image(
            image_path,
            output_image
        )

        if success:
            processed_count += 1
        else:
            failed_count += 1

        # Copy corresponding YOLO label
        label_path = (
            input_labels /
            f"{image_path.stem}.txt"
        )

        output_label = (
            output_labels /
            f"{image_path.stem}.txt"
        )

        copy_label(
            label_path,
            output_label
        )

        # Progress
        if index % 100 == 0 or index == len(image_files):

            print(
                f"Progress: {index}/{len(image_files)}"
            )

    print()
    print(f"{split} completed.")
    print(f"Processed: {processed_count}")
    print(f"Failed: {failed_count}")


# =========================================================
# 6. MAIN
# =========================================================

def main():

    print("=" * 60)
    print("PCB DATASET PREPROCESSING")
    print("=" * 60)

    print()
    print("Input dataset:")
    print(INPUT_DATASET)

    print()
    print("Output dataset:")
    print(OUTPUT_DATASET)

    # Process train
    process_dataset_split("train")

    # Process validation
    process_dataset_split("val")

    # Process test
    process_dataset_split("test")

    print()
    print("=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)

    print()
    print("Preprocessed dataset saved at:")
    print(OUTPUT_DATASET)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()