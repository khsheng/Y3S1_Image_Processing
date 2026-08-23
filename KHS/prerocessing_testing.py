from KHS.preprocessing_function import (
    grayscale,
    average_filter,
    median_filter,
    morphological_erosion,
    morphological_opening,
    morphological_closing,
    clahe,
    eet
)

import cv2
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# PREPROCESSING PIPELINE
# =========================================================

def preprocess_image(image):

    # Step 1: Grayscale
    gray = grayscale(image)

    # Step 2: Median Filtering
    median = median_filter(gray)

    # Step 3: Morphological Opening
    opened = morphological_opening(median)

    # Step 5: CLAHE
    clahe_image = clahe(opened)

    # Step 6: EET
    edge_enhanced = eet(clahe_image)

    return (
        gray,
        median,
        opened,
        clahe_image,
        edge_enhanced
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
    image_path = Path(r"C:\Y3S1\image processing\assignment\pcb-defect-dataset\test\images\l_light_04_mouse_bite_03_1_600.jpg")

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
        gray,
        median,
        opened,
        clahe_image,
        edge_enhanced
    ) = preprocess_image(image)

    # =====================================================
    # Prepare Results
    # =====================================================

    images = [
        image,
        gray,
        median,
        opened,
        clahe_image,
        edge_enhanced
    ]

    titles = [
        "Original",
        "Grayscale",
        "Median Filtering",
        "Morphological Opening",
        "Morphological Closing",
        "CLAHE",
        "EET"
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