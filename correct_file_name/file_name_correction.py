from pathlib import Path


def fix_256_labels(dataset_split):

    images_dir = dataset_split / "images"
    labels_dir = dataset_split / "labels"

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    # Get all image stems
    image_stems = {
        image.stem
        for image in images_dir.iterdir()
        if image.is_file()
        and image.suffix.lower() in image_extensions
    }

    # Check every label
    for label_path in labels_dir.glob("*.txt"):

        label_stem = label_path.stem

        # Already has a corresponding image
        if label_stem in image_stems:
            continue

        # Only fix labels ending with _256
        if not label_stem.endswith("_256"):
            continue

        # Change _256 -> _600
        new_stem = label_stem[:-4] + "_600"

        # Check if _600 image exists
        if new_stem in image_stems:

            new_label_path = (
                labels_dir /
                f"{new_stem}.txt"
            )

            print(
                f"FIX: {label_path.name} "
                f"-> {new_label_path.name}"
            )

            label_path.rename(new_label_path)

        else:

            print(
                f"WARNING: {label_path.name} "
                f"cannot be fixed because "
                f"{new_stem} image does not exist."
            )

def main():

    dataset_names = [
        "pcb-defect-claheRGB",
        "pcb-defect-dataset"
    ]

    data_types = [
        "test",
        "train",
        "val"
    ]

    for dataset_name in dataset_names:

        dataset = Path.cwd() / dataset_name

        for data_type in data_types:

            dataset_split = dataset / data_type

            print()
            print("=" * 60)
            print(f"Dataset: {dataset_name}")
            print(f"Split:   {data_type}")
            print("=" * 60)

            fix_256_labels(dataset_split)
            


if __name__ == "__main__":
    main()