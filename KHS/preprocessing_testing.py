from KHS.function.preprocessing_function import (
    clahe_rgb,
    grayscale,
    average_filter,
    median_filter,
    morphological_erosion,
    morphological_opening,
    morphological_closing,
    clahe
)

import cv2
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# PREPROCESSING PIPELINE
# =========================================================

def preprocess_image(image):

    median = median_filter(image)

    opened = morphological_opening(median)

    clahe_image = clahe_rgb(opened)

    return (
        median,
        opened,
        clahe_image
    )


# =========================================================
# DISPLAY RESULTS
# =========================================================

def display_results(images, titles):

    plt.figure(figsize=(15, 10))

    for i in range(len(images)):

        plt.subplot(2, 4, i + 1)

        if i == 0:
            plt.imshow(
                cv2.cvtColor(
                    images[i],
                    cv2.COLOR_BGR2RGB
                )
            )
        else:
            plt.imshow(
                images[i],
                cmap="gray"
            )

        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# =========================================================
# MAIN
# =========================================================

def main():

    # Image path
    image_path = Path(r"C:\Y3S1\image_processing\assignment\pcb-defect-dataset\test\images\l_light_01_missing_hole_04_2_600.jpg")

    # =====================================================
    # Load Image
    # =====================================================

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Cannot load image: {image_path}"
        )

    # =====================================================
    # Run Preprocessing
    # =====================================================

    (
        # gray,
        median,
        opened,
        clahe_image,
    ) = preprocess_image(image)

    # =====================================================
    # Prepare Results
    # =====================================================

    images = [
        image,
        median,
        opened,
        clahe_image
    ]

    titles = [
        "Original",
        "Median Filtering",
        "Morphological Opening",
        "CLAHE"
    ]

    # =====================================================
    # Display
    # =====================================================

    display_results(
        images,
        titles
    )


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()