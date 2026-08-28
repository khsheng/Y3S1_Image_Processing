from scipy.io import loadmat
from PIL import Image
import numpy as np
from pathlib import Path
from KHS.function.preprocessing_function import morphological_opening
import shutil



def enhance_image(
    image_path,
    mat_path,
    output_path,
    threshold,
    original_weight,
    edge_weight
):
    """
    Enhance an image using a PiDiNet edge map.

    Parameters
    ----------
    image_path : str
        Path to the original image.

    mat_path : str
        Path to the corresponding PiDiNet .mat file.

    output_path : str
        Path to save the enhanced image.

    threshold : float
        PiDiNet threshold.
        Values >= threshold become 255.
        Values < threshold become 0.

    original_weight : float
        Weight of the original image.

    edge_weight : float
        Weight of the binary edge map.
    """

    # Load original image
    original = np.array(
        Image.open(image_path).convert("RGB")
    ).astype(np.float32)

    # Load PiDiNet prediction
    mat = loadmat(mat_path)
    edge = mat["img"].astype(np.float32)

    # Convert to binary edge map
    # edge = (edge >= threshold).astype(np.float32)
    edge = (1.0 - edge) * 255.0

    # Combine original image and edge
    result = (
        original * original_weight
        + edge[..., np.newaxis] * edge_weight
    )

    # Morphological opening
    result = morphological_opening(result)

    # Keep values between 0 and 255
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Save
    Image.fromarray(result).save(output_path)


def process_input(
    image_input,
    mat_input,
    output_dir,
    threshold,
    original_weight,
    edge_weight
):
    """
    Process either:
        - single image + single MAT
        - folder of images + folder of MAT files
    """

    image_input = Path(image_input)
    mat_input = Path(mat_input)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    # ==================================================
    # Single image + single MAT
    # ==================================================
    if image_input.is_file() and mat_input.is_file():

        output_path = (
            output_dir /
            f"{image_input.stem}_enhanced.png"
        )

        enhance_image(
            image_path=image_input,
            mat_path=mat_input,
            output_path=output_path,
            threshold=threshold,
            original_weight=original_weight,
            edge_weight=edge_weight
        )

        print("Done! 0 images left.")

        return

    # ==================================================
    # Folder of images + folder of MAT files
    # ==================================================
    if image_input.is_dir() and mat_input.is_dir():

        image_files = sorted([
            p for p in image_input.iterdir()
            if p.suffix.lower() in image_extensions
        ])

        mat_files = {
            p.stem: p
            for p in mat_input.glob("*.mat")
        }

        total = len(image_files)

        for index, image_path in enumerate(image_files, start=1):

            mat_path = mat_files.get(image_path.stem)

            if mat_path is None:
                remaining = total - index

                print(
                    f"[{index}/{total}] "
                    f"MAT not found - {remaining} images left"
                )

                continue

            output_path = (
                output_dir /
                f"{image_path.stem}_enhanced.png"
            )

            enhance_image(
                image_path=image_path,
                mat_path=mat_path,
                output_path=output_path,
                threshold=threshold,
                original_weight=original_weight,
                edge_weight=edge_weight
            )

            remaining = total - index

            print(
                f"[{index}/{total}] "
                f"{remaining} images left"
            )

        print("\nAll images processed!")

        return

    raise ValueError(
        "image_input and mat_input must both be files "
        "or both be folders."
    )


def main(
    image_input,
    mat_input,
    output_dir,
    threshold=0.85,
    original_weight=0.8,
    edge_weight=0.2
):
    """
    Main function.

    image_input:
        Original JPG/image file OR folder.

    mat_input:
        PiDiNet MAT file OR folder.

    output_dir:
        Output folder.

    threshold:
        Binary edge threshold.

    original_weight:
        Weight of original image.

    edge_weight:
        Weight of edge map.
    """

    process_input(
        image_input=image_input,
        mat_input=mat_input,
        output_dir=output_dir,
        threshold=threshold,
        original_weight=original_weight,
        edge_weight=edge_weight
    )


if __name__ == "__main__":

    type_of_image = ["test", "train", "val"]

    image_foulder_name = "pcb-defect-claheRGB"

    mat_foulder_name = "pcb-defect-pidinet-dataset(edge_id=2)"

    new_dataset_name = f"{image_foulder_name}-edgeEnhanced(edge_id=2)(without_threshold)"

    for img_type in type_of_image:

        # =========================
        # Image paths
        # =========================
        image_input = rf"C:\Y3S1\image_processing\assignment\{image_foulder_name}\{img_type}\images"

        mat_input = rf"C:\Y3S1\image_processing\assignment\edge_detection\{mat_foulder_name}\{img_type}\output_mats"

        output_dir = rf"C:\Y3S1\image_processing\assignment\{new_dataset_name}\{img_type}\images"

        # =========================
        # Enhance images
        # =========================
        main(
            image_input=image_input,
            mat_input=mat_input,
            output_dir=output_dir,
            threshold=0.85,
            original_weight=0.8,
            edge_weight=0.2
        )

        # =========================
        # Copy labels
        # =========================
        label_input = rf"C:\Y3S1\image_processing\assignment\pcb-defect-median-opened-claheRGB\{img_type}\labels"
        label_output = rf"C:\Y3S1\image_processing\assignment\{new_dataset_name}\{img_type}\labels"

        shutil.copytree(
            label_input,
            label_output,
            dirs_exist_ok=True
        )

        print(f"Finished {img_type}")