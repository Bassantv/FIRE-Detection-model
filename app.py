import streamlit as st
import os
from PIL import Image
from ultralytics import YOLO
import tempfile
import cv2
import torch

st.set_page_config(page_title="Fire Detection Dashboard", page_icon=":fire:")
st.title(":fire: Fire Detection MODEL ")

# Sidebar selection
option = st.sidebar.selectbox("Select Mode", ["Data Sample", "Training Data Visualization", "Image & Video Detection"])

# --------------------- #
# 🔹 Training Dashboard #
# --------------------- #
if option == "Data Sample":
    st.header("🗂️ Sample Annotated Images")

    # Folder paths
    image_folder = "sample/images"  # path to your images
    label_folder = "sample/labels"  # corrected path to .txt files

    # Check if folders exist
    if not os.path.exists(image_folder) or not os.path.exists(label_folder):
        st.error("Sample data folders not found. Please ensure 'sample_data/images' and 'sample_data/labels' exist.")
        st.stop()

    # List images
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        st.warning("No image files found in the specified folder.")
        st.stop()

    # Let user choose image
    selected_image_name = st.selectbox("Choose an image to display", image_files)

    if selected_image_name:
        image_path = os.path.join(image_folder, selected_image_name)
        label_path = os.path.join(label_folder, selected_image_name.rsplit('.', 1)[0] + ".txt")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    _, x_center, y_center, w, h = map(float, parts)

                    # Convert to pixel values
                    x1 = int((x_center - w / 2) * width)
                    y1 = int((y_center - h / 2) * height)
                    x2 = int((x_center + w / 2) * width)
                    y2 = int((y_center + h / 2) * height)

                    # Draw box and label
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image, "fire", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        st.image(image, caption="Image with Fire Annotations", use_container_width=True)

# --------------------- #
# 🔹 Training Dashboard #
# --------------------- #
elif option == "Training Data Visualization":
    st.header("📊 Display Pre-Generated Plots")  # Fixed emoji here
    plots ={
        "results": "results.png",
        "BoxF1_curve": "BoxF1_curve.png",
        "BoxPR_curve": "BoxPR_curve.png",
        "BoxR_curve": "BoxR_curve.png",
        "confusion_matrix": "confusion_matrix.png"
    }

    # Move checkboxes to sidebar
    st.sidebar.subheader("Select Plots to Show :sparkles:")
    selected_plots = [plot_name for plot_name in plots if st.sidebar.checkbox(plot_name)]

    # Show selected plots
    for plot_name in selected_plots:
        plot_path = os.path.join("plots", plots[plot_name])  # updated to use plots/ folder
        if os.path.exists(plot_path):
            st.image(plot_path, caption=plot_name, use_container_width=True)
        else:
            st.warning(f"{plot_name} image not found at {plot_path}")


# ------------------------------ #
# 🔹 Inference: Image + Video    #
# ------------------------------ #
elif option == "Image & Video Detection":
    st.sidebar.header("Model Inference Settings")
    torch.serialization.add_safe_globals([__import__('ultralytics.nn.tasks').nn.tasks.DetectionModel])

    # Then load the model
    model_path = "best.pt"
    model = YOLO(model_path)
    # Confidence threshold slider
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, step=0.05)

    # --------------------- #
    # 🔹 IMAGE DETECTION    #
    # --------------------- #
    st.header("🖼️ Image Detection")
    image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if image_file:
        # Save uploaded image to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            tmp_img.write(image_file.read())
            tmp_img_path = tmp_img.name

        # Run model on image path (most reliable)
        results = model.predict(source=tmp_img_path, conf=conf_threshold)
        detections = results[0].boxes

        # Load image back to display
        original_image = Image.open(tmp_img_path)

        # Draw detections
        image_cv = cv2.imread(tmp_img_path)
        if detections is not None and len(detections) > 0:
            boxes = detections.data.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                label = f"fire {conf:.2f}"

                cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

        else:
            st.warning("⚠️ No fire detected in this image.")

        # Display result
        result_image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        result_image_pil = Image.fromarray(result_image_rgb)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🖼️ Original Image")
            st.image(original_image, use_container_width=True)
        with col2:
            st.markdown("### 🔍 Detection Output")
            st.image(result_image_pil, use_container_width=True)

    # --------------------- #
    # 🔹 VIDEO DETECTION    #
    # --------------------- #
    st.header("🎥 Video Detection")
    video_file = st.file_uploader("Upload Video for Detection", type=["mp4", "avi", "mov"])

    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(video_file.read())
            video_path = tmp_video.name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        st.sidebar.success("Running video detection...")
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, conf=conf_threshold, verbose=False)
            result_frame = results[0].plot()

            rgb_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            stframe.image(rgb_frame, channels="RGB", use_container_width=True)
            out.write(result_frame)

        cap.release()
        out.release()

        with open(output_path, "rb") as f:
            video_bytes = f.read()
            st.download_button(
                label="⬇️ Download Processed Video",
                data=video_bytes,
                file_name="detected_video.mp4",
                mime="video/mp4"
            )

        os.remove(video_path)
        os.remove(output_path)




