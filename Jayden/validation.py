import os
from ultralytics import YOLO


# ============================================================
# UNIVERSAL YOLOv8m VALIDATION
# ============================================================
#
# This file validates all experiments using their best.pt.
#
# Experiments:
# 1. Baseline
# 2. Sharpening
# 3. CLAHE
# 4. CLAHE + Sharpening
#
# You only need to change the paths below if your folders
# have different names.
# ============================================================


# ============================================================
# 1. EXPERIMENT PATHS
# ============================================================

EXPERIMENTS = {

    "Baseline": {
        "model": "./pcb-yolo-results/baseline/weights/best.pt",
        "data": "./pcb-defect-dataset/data.yaml"
    },

    "Sharpening": {
        "model": "./pcb-yolo-results-sharpening/sharpening/weights/best.pt",
        "data": "./pcb-defect-dataset-sharpening/data.yaml"
    },

    "CLAHE": {
        "model": "./pcb-yolo-results-clahe/clahe/weights/best.pt",
        "data": "./pcb-defect-dataset-clahe/data.yaml"
    },

    "CLAHE + Sharpening": {
        "model": "./pcb-yolo-results-clahe-sharpening/"
                 "clahe_sharpening/weights/best.pt",
        "data": "./pcb-defect-dataset-clahe-sharpening/data.yaml"
    }

}


# ============================================================
# 2. VALIDATION SETTINGS
# ============================================================

VAL_SPLIT = "val"


# ============================================================
# 3. RESULTS STORAGE
# ============================================================

results = []


# ============================================================
# 4. VALIDATE EACH EXPERIMENT
# ============================================================

print("\n")
print("============================================================")
print("              YOLOv8m ALL EXPERIMENT VALIDATION")
print("============================================================")


for experiment_name, paths in EXPERIMENTS.items():

    model_path = paths["model"]
    data_yaml = paths["data"]

    print("\n")
    print("============================================================")
    print(f"EXPERIMENT: {experiment_name}")
    print("============================================================")

    print("\nModel:")
    print(model_path)

    print("\nData:")
    print(data_yaml)


    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(model_path):

        print("\nWARNING: Model not found.")
        print("Skipping this experiment.")

        results.append({
            "Experiment": experiment_name,
            "Precision": None,
            "Recall": None,
            "mAP50": None,
            "mAP50-95": None
        })

        continue


    # --------------------------------------------------------
    # Check data.yaml
    # --------------------------------------------------------

    if not os.path.exists(data_yaml):

        print("\nWARNING: data.yaml not found.")
        print("Skipping this experiment.")

        results.append({
            "Experiment": experiment_name,
            "Precision": None,
            "Recall": None,
            "mAP50": None,
            "mAP50-95": None
        })

        continue


    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = YOLO(model_path)

    print("Model loaded successfully.")


    # --------------------------------------------------------
    # Run validation
    # --------------------------------------------------------

    print("\nRunning validation...")

    metrics = model.val(
        data=data_yaml,
        split=VAL_SPLIT
    )


    # --------------------------------------------------------
    # Extract metrics
    # --------------------------------------------------------

    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map5095 = metrics.box.map


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results.append({

        "Experiment": experiment_name,

        "Precision": precision,

        "Recall": recall,

        "mAP50": map50,

        "mAP50-95": map5095

    })


    # --------------------------------------------------------
    # Print experiment results
    # --------------------------------------------------------

    print("\n------------------------------------------------------------")
    print(f"{experiment_name} RESULTS")
    print("------------------------------------------------------------")

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"mAP50     : {map50:.4f}")
    print(f"mAP50-95  : {map5095:.4f}")

    print("\nPercentage:")

    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"mAP50     : {map50 * 100:.2f}%")
    print(f"mAP50-95  : {map5095 * 100:.2f}%")


# ============================================================
# 5. FINAL COMPARISON TABLE
# ============================================================

print("\n\n")
print("============================================================")
print("                 FINAL COMPARISON")
print("============================================================")


print(
    f"{'Experiment':<25}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'mAP50':>12}"
    f"{'mAP50-95':>12}"
)

print("-" * 73)


for result in results:

    experiment = result["Experiment"]

    precision = result["Precision"]
    recall = result["Recall"]
    map50 = result["mAP50"]
    map5095 = result["mAP50-95"]


    if precision is None:

        print(
            f"{experiment:<25}"
            f"{'NOT FOUND':>12}"
            f"{'NOT FOUND':>12}"
            f"{'NOT FOUND':>12}"
            f"{'NOT FOUND':>12}"
        )

    else:

        print(
            f"{experiment:<25}"
            f"{precision:>12.4f}"
            f"{recall:>12.4f}"
            f"{map50:>12.4f}"
            f"{map5095:>12.4f}"
        )


# ============================================================
# 6. BEST EXPERIMENT
# ============================================================

valid_results = [

    result
    for result in results
    if result["mAP50-95"] is not None

]


if len(valid_results) > 0:

    best_result = max(
        valid_results,
        key=lambda x: x["mAP50-95"]
    )


    print("\n")
    print("============================================================")
    print("                    BEST EXPERIMENT")
    print("============================================================")

    print(
        f"\nBest Experiment : "
        f"{best_result['Experiment']}"
    )

    print(
        f"Precision       : "
        f"{best_result['Precision']:.4f}"
    )

    print(
        f"Recall          : "
        f"{best_result['Recall']:.4f}"
    )

    print(
        f"mAP50           : "
        f"{best_result['mAP50']:.4f}"
    )

    print(
        f"mAP50-95        : "
        f"{best_result['mAP50-95']:.4f}"
    )


# ============================================================
# 7. COMPLETED
# ============================================================

print("\n")
print("============================================================")
print("             ALL VALIDATION COMPLETED")
print("============================================================")