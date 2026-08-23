import cv2
import matplotlib.pyplot as plt


# =========================
# 1. Load Image
# =========================
def load_image(path):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(
            f"Cannot load image: {path}"
        )

    return image


# =========================
# 2. Grayscale
# =========================
def grayscale(image):
    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


# =========================
# 3. Average Filtering
# =========================
def average_filter(image):
    return cv2.blur(
        image,
        (9, 9)
    )

def median_filter(image):
    return cv2.medianBlur(
        image,
        9
    )


# =========================
# 4. Morphological Erosion
# =========================
def morphological_erosion(image):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    return cv2.erode(
        image,
        kernel,
        iterations=1
    )


# =========================
# 5. CLAHE
# =========================
def clahe(image):
    clahe_processor = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe_processor.apply(image)


# =========================
# 6. EET
# =========================
def eet(image):
    blurred = cv2.GaussianBlur(
        image,
        (3, 3),
        0
    )

    return cv2.addWeighted(
        image,
        1.5,
        blurred,
        -0.5,
        0
    )

def morphological_opening(image):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    return cv2.morphologyEx(
        image,
        cv2.MORPH_OPEN,
        kernel
    )

def morphological_closing(image):

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    return cv2.morphologyEx(
        image,
        cv2.MORPH_CLOSE,
        kernel
    )


# =========================
# 8. Display Results
# =========================
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


# =========================
# MAIN PIPELINE
# =========================
def main():

    image_path = (
        r"C:\Y3S1\image processing\assignment"
        r"\pcb-defect-dataset\test\images"
        r"\l_light_01_mouse_bite_05_2_600.jpg"
    )

    # Pipeline
    image = load_image(image_path)

    gray = grayscale(image)

    average = average_filter(gray)

    eroded = morphological_erosion(average)

    clahe_image = clahe(eroded)

    edge_enhanced = eet(clahe_image)

    # Results
    images = [
        image,
        gray,
        average,
        eroded,
        clahe_image,
        edge_enhanced
    ]

    titles = [
        "Original",
        "Grayscale",
        "Average Filtering",
        "Morphological Erosion",
        "CLAHE",
        "EET"
    ]

    display_results(images, titles)


# =========================
# Run Program
# =========================
if __name__ == "__main__":
    main()