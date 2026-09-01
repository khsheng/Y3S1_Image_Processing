import streamlit as st
import cv2
import numpy as np
import tempfile
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# 2. MODEL PATHS
# ============================================================

MODEL_PATHS = {
    "Median (K=9) + CLAHE  ✨ Best": (
        "../runs/runs/"
        "yolov8s_with_pcb-defect-dataset-median-(K=9)-clahe/"
        "weights/best.pt"
    ),
    "Sobel + Solder Mask": (
        "../runs/runs/"
        "yolov8s_with_pcb-defect-dataset-sobel-solder-mask/"
        "weights/best.pt"
    ),
}


@st.cache_resource
def load_model(path: str):
    return YOLO(path)


# ============================================================
# 3. PREPROCESSING PIPELINES
# ============================================================

def median_clahe(
    image,
    kernel_size=9,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
):
    """
    Best-performing pipeline: Median (K=9) + CLAHE.

    Steps
    -----
    1. Median blur (k=9) -- removes salt-and-pepper noise while
       preserving PCB trace edges.
    2. Convert BGR -> LAB colour space.
    3. Apply CLAHE on the L (luminance) channel -- enhances local
       contrast without over-brightening saturated colours.
    4. Merge LAB back -> BGR.

    Returns a BGR image of the same size as the input.
    """

    # Step 1: Median filtering
    median_filtered = cv2.medianBlur(image, kernel_size)

    # Step 2: BGR -> LAB
    lab = cv2.cvtColor(median_filtered, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Step 3: CLAHE on luminance channel
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    l = clahe.apply(l)

    # Step 4: Merge and convert back
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


def sobel_solder_mask(
    image,
    kernel_size=9,
    lower=52,
    upper=78
):
    """
    Original pipeline: Sobel edge map + Solder Mask highlight.

    Steps
    -----
    1. Median blur (k=9).
    2. Grayscale conversion.
    3. Solder-mask isolation via pixel-range threshold.
    4. Sobel X + Y gradient magnitude (normalised to uint8).
    5. Overlay solder-mask pixels in RED on the edge map.

    Returns a BGR image of the same size as the input.
    """

    # Median filtering
    filtered = cv2.medianBlur(image, kernel_size)

    # Grayscale
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

    # Solder mask threshold
    sold_mask = cv2.inRange(gray, lower, upper)

    # Sobel X/Y
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Sobel magnitude
    sobel = np.abs(sobel_x) + np.abs(sobel_y)

    # Normalize
    sobel = cv2.normalize(
        sobel, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Convert to BGR
    overlay = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

    # Highlight solder-mask regions RED
    overlay[sold_mask > 0] = [0, 0, 255]

    return overlay


# Map pipeline name -> function
PIPELINES = {
    "Median (K=9) + CLAHE  ✨ Best": median_clahe,
    "Sobel + Solder Mask": sobel_solder_mask,
}


# ============================================================
# 4. IMAGE PROCESSING
# ============================================================

def process_image(image_bytes, pipeline_fn, model):

    file_bytes = np.asarray(
        bytearray(image_bytes),
        dtype=np.uint8
    )

    original = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if original is None:
        return None, None

    # Preprocessing (selected pipeline)
    processed = pipeline_fn(original)

    # YOLO inference
    results = model(
        processed,
        conf=0.25,
        iou=0.45,
        verbose=False,
        device=0
    )

    # Draw boxes on ORIGINAL
    annotated = results[0].plot(
        img=original
    )

    return processed, annotated


# ============================================================
# 5. VIDEO PROCESSING
# ============================================================

def process_video(input_path, output_path, pipeline_fn, model):

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        return False, "Cannot open video."

    # ----------------------------------------
    # Video properties
    # ----------------------------------------

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    # Prevent invalid FPS
    if fps <= 0:
        fps = 30

    # ----------------------------------------
    # VideoWriter
    # ----------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        cap.release()
        return False, "Cannot create output video."

    # ----------------------------------------
    # Progress bar
    # ----------------------------------------

    progress = st.progress(0)

    status = st.empty()

    frame_count = 0

    # ----------------------------------------
    # Frame processing
    # ----------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ------------------------------------
        # Preprocessing (selected pipeline)
        # ------------------------------------

        processed_frame = pipeline_fn(
            frame
        )

        # ------------------------------------
        # YOLO detection
        # ------------------------------------

        results = model(
            processed_frame,
            conf=0.25,
            iou=0.45,
            verbose=False,
            device=0
        )

        # ------------------------------------
        # Draw detection on ORIGINAL frame
        # ------------------------------------

        annotated_frame = results[0].plot(
            img=frame
        )

        # ------------------------------------
        # Write frame
        # ------------------------------------


        out.write(
            annotated_frame
        )

        # ------------------------------------
        # Update progress
        # ------------------------------------

        frame_count += 1

        if total_frames > 0:

            percentage = (
                frame_count / total_frames
            )

            progress.progress(
                min(percentage, 1.0)
            )

            status.text(
                f"Processing frame "
                f"{frame_count}/{total_frames}"
            )

    # ----------------------------------------
    # Release resources
    # ----------------------------------------

    cap.release()
    out.release()

    progress.progress(1.0)

    status.success(
        f"Completed {frame_count} frames."
    )


    return True, None


# ============================================================
# 6. SIDEBAR -- PIPELINE & MODEL SELECTION
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuration")

    selected_pipeline = st.selectbox(
        "Preprocessing Pipeline",
        list(MODEL_PATHS.keys()),
        index=0,          # Default: best model
        help=(
            "Choose the preprocessing pipeline.\n\n"
            "**Median (K=9) + CLAHE** — best overall metrics "
            "(highest mAP50 / F1).\n\n"
            "**Sobel + Solder Mask** — edge-based pipeline with "
            "explicit solder-mask highlighting."
        )
    )

    st.divider()
    st.caption(
        "**Median (K=9) + CLAHE** applies median blur then "
        "CLAHE contrast enhancement in LAB space.\n\n"
        "**Sobel + Solder Mask** computes Sobel edge gradients "
        "and highlights solder-mask regions in red."
    )

pipeline_fn = PIPELINES[selected_pipeline]
model = load_model(MODEL_PATHS[selected_pipeline])

# Friendly short name for captions
_short = (
    "Median K=9 + CLAHE"
    if "CLAHE" in selected_pipeline
    else "Sobel + Solder Mask"
)


# ============================================================
# 7. USER INTERFACE
# ============================================================

st.title("🔍 PCB Defect Detection System")

st.write(
    f"YOLOv8s PCB defect detection · active pipeline: **{_short}**"
)


# ============================================================
# 8. SELECT INPUT TYPE
# ============================================================

input_type = st.radio(
    "Select input type:",
    ["Images", "Video"],
    horizontal=True
)


# ============================================================
# 9. IMAGE MODE
# ============================================================

if input_type == "Images":

    uploaded_files = st.file_uploader(
        "Upload PCB images",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp"
        ],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} image(s) uploaded."
        )

        for uploaded_file in uploaded_files:

            st.divider()

            st.subheader(
                uploaded_file.name
            )

            processed, annotated = process_image(
                uploaded_file.getvalue(),
                pipeline_fn,
                model
            )

            if processed is None:

                st.error(
                    "Unable to process this image."
                )

                continue

            # BGR → RGB
            processed_rgb = cv2.cvtColor(
                processed,
                cv2.COLOR_BGR2RGB
            )

            annotated_rgb = cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB
            )

            # ------------------------------------
            # Display
            # ------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    processed_rgb,
                    caption=f"Preprocessed — {_short}",
                    use_container_width=True
                )

            with col2:

                st.image(
                    annotated_rgb,
                    caption="YOLO Detection",
                    use_container_width=True
                )

            # ------------------------------------
            # Download
            # ------------------------------------

            success, encoded = cv2.imencode(
                ".jpg",
                annotated
            )

            if success:

                st.download_button(
                    "Download Detection Result",
                    data=encoded.tobytes(),
                    file_name=(
                        f"detected_{uploaded_file.name}"
                    ),
                    mime="image/jpeg"
                )


# ============================================================
# 10. VIDEO MODE
# ============================================================

else:

    uploaded_video = st.file_uploader(
        "Upload PCB inspection video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ],
        accept_multiple_files=False
    )

    if uploaded_video:

        st.video(
            uploaded_video
        )

        if st.button(
            "▶ Start Video Detection",
            type="primary"
        ):

            # ------------------------------------
            # Save uploaded video temporarily
            # ------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_input:

                temp_input.write(
                    uploaded_video.getvalue()
                )

                input_path = temp_input.name

            # ------------------------------------
            # Output temporary file
            # ------------------------------------

            output_path = tempfile.NamedTemporaryFile(
                delete=False,
                suffix="_detection.mp4"
            ).name

            # ------------------------------------
            # Process
            # ------------------------------------

            with st.spinner(
                "Processing video..."
            ):

                success, error = process_video(
                    input_path,
                    output_path,
                    pipeline_fn,
                    model
                )

            if success:

                st.success(
                    "Video processing completed!"
                )

                # --------------------------------
                # Display processed video
                # --------------------------------

                st.subheader(
                    "YOLO Detection Result"
                )

                with open(
                    output_path,
                    "rb"
                ) as video_file:

                    video_bytes = (
                        video_file.read()
                    )

                st.video(
                    video_bytes
                )

                # --------------------------------
                # Download
                # --------------------------------

                st.download_button(
                    label="⬇ Download Processed Video",
                    data=video_bytes,
                    file_name="pcb_yolo_detection.mp4",
                    mime="video/mp4"
                )

            else:

                st.error(error)