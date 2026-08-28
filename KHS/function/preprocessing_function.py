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

def clahe_rgb(image):
    # RGB -> LAB
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Split L, A, B
    l, a, b = cv2.split(lab)

    # CLAHE only on L channel
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    l = clahe.apply(l)

    # Merge back
    lab = cv2.merge((l, a, b))

    # LAB -> RGB
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return result

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