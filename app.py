
import os
import time
import tempfile
import streamlit as st
import numpy as np
import pandas as pd

from PIL import Image
from datetime import datetime
from ultralytics import YOLO

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Smart Surveillance System",
    page_icon="📹",
    layout="wide"
)

st.title("📹 Smart Surveillance System")
st.subheader(
    "Employee Monitoring and Indexing using Deep Learning"
)

# =====================================
# DATABASE
# =====================================

DATABASE = "employee_records.csv"

COLUMNS = [
    "Date",
    "Time",
    "Person_Count",
    "Tracking_IDs",
    "Source"
]

def initialize_database():
    if not os.path.exists(DATABASE):
        pd.DataFrame(columns=COLUMNS).to_csv(
            DATABASE, index=False
        )

def save_record(count, ids, source):
    initialize_database()

    record = pd.DataFrame([{
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Person_Count": count,
        "Tracking_IDs": ", ".join(map(str, ids)),
        "Source": source
    }])

    record.to_csv(
        DATABASE,
        mode="a",
        header=False,
        index=False
    )

# =====================================
# LOAD YOLO11 MODEL
# =====================================

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

initialize_database()

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("📌 Navigation")

module = st.sidebar.selectbox(
    "Choose Module",
    [
        "Home",
        "Image Detection",
        "Video Monitoring",
        "Employee Records"
    ]
)

# =====================================
# LOAD MODEL
# =====================================

try:
    with st.spinner("Loading AI Model..."):
        model = load_model()

    st.sidebar.success("AI Model Loaded Successfully")

except Exception as e:
    st.error("AI Model Loading Failed")
    st.exception(e)
    st.stop()

# =====================================
# HOME DASHBOARD
# =====================================

if module == "Home":

    st.header("🏠 Home Dashboard")

    df = pd.read_csv(DATABASE)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("AI Model", "YOLO11")

    with col2:
        st.metric("Total Records", len(df))

    with col3:
        st.metric("System Status", "Active")

    st.divider()

    st.subheader("Project Features")

    st.write("""
    - AI Based Person Detection
    - Employee Tracking
    - Image Detection
    - Video Monitoring
    - Person Counting
    - Employee Records
    - CSV Report Generation
    """)

    st.info(
        "Select a module from the sidebar to start monitoring."
    )

# =====================================
# IMAGE DETECTION
# =====================================

elif module == "Image Detection":

    st.header("🖼️ Image Detection")

    uploaded_file = st.file_uploader(
        "Upload CCTV Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("🔍 Detect Persons"):

            with st.spinner("AI Detecting Persons..."):

                results = model.predict(
                    image,
                    classes=[0],
                    conf=0.4,
                    verbose=False
                )

                result = results[0]

                # Convert BGR to RGB without importing cv2
                detected_image = result.plot()
                detected_image = detected_image[:, :, ::-1]

                person_count = len(result.boxes)

                st.success("Detection Completed!")

                st.image(
                    detected_image,
                    caption="Detection Result",
                    use_container_width=True
                )

                st.metric(
                    "Total Persons Detected",
                    person_count
                )

                save_record(
                    person_count,
                    [],
                    "Image"
                )

# =====================================
# VIDEO MONITORING
# =====================================

elif module == "Video Monitoring":

    st.header("🎥 Video Monitoring")

    uploaded_video = st.file_uploader(
        "Upload CCTV Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button("▶️ Start Monitoring"):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ) as temp_file:

                temp_file.write(
                    uploaded_video.getbuffer()
                )

                video_path = temp_file.name

            video_display = st.empty()
            status_display = st.empty()

            total_frames = 0
            last_saved = 0

            try:

                # Ultralytics internally uses OpenCV
                from ultralytics.utils.checks import check_requirements
                from ultralytics.data.loaders import LoadImages

                loader = LoadImages(video_path)

                for frame_data in loader:

                    if len(frame_data) == 3:
                        path, frame, original_frame = frame_data
                    else:
                        continue

                    results = model.track(
                        original_frame,
                        persist=True,
                        classes=[0],
                        conf=0.4,
                        tracker="bytetrack.yaml",
                        verbose=False
                    )

                    result = results[0]

                    annotated_frame = result.plot()

                    # BGR to RGB
                    annotated_frame = annotated_frame[:, :, ::-1]

                    tracking_ids = []

                    if (
                        result.boxes is not None
                        and result.boxes.id is not None
                    ):

                        tracking_ids = (
                            result.boxes.id
                            .cpu()
                            .numpy()
                            .astype(int)
                            .tolist()
                        )

                    person_count = len(result.boxes)

                    video_display.image(
                        annotated_frame,
                        channels="RGB",
                        use_container_width=True
                    )

                    status_display.info(
                        f"Persons Detected: {person_count} | "
                        f"Tracking IDs: {tracking_ids}"
                    )

                    current_time = time.time()

                    if current_time - last_saved >= 1:

                        save_record(
                            person_count,
                            tracking_ids,
                            "Video"
                        )

                        last_saved = current_time

                    total_frames += 1

                st.success(
                    f"Monitoring Completed! Frames: {total_frames}"
                )

            except Exception as e:

                st.error("Video Processing Error")
                st.exception(e)

            finally:

                if os.path.exists(video_path):
                    os.remove(video_path)

# =====================================
# EMPLOYEE RECORDS
# =====================================

elif module == "Employee Records":

    st.header("📊 Employee Records")

    df = pd.read_csv(DATABASE)

    if not df.empty:

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Records", len(df))

        with col2:
            st.metric(
                "Total Person Detections",
                int(df["Person_Count"].sum())
            )

        st.dataframe(
            df,
            use_container_width=True
        )

        csv_data = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Surveillance Report",
            data=csv_data,
            file_name="employee_records.csv",
            mime="text/csv"
        )

        if st.button("Clear All Records"):

            pd.DataFrame(
                columns=COLUMNS
            ).to_csv(DATABASE, index=False)

            st.success("Records Cleared Successfully!")

            st.rerun()

    else:

        st.info("No Employee Records Available.")

# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "Smart Surveillance System | "
    "Python | YOLO11 | Deep Learning"
)
