
import os
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from datetime import datetime
from PIL import Image
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

        pd.DataFrame(
            columns=COLUMNS
        ).to_csv(DATABASE, index=False)


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

    model = YOLO("yolo11n.pt")

    return model


initialize_database()

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("📌 Navigation")

module = st.sidebar.selectbox(
    "Select Module",
    [
        "Home",
        "Image Detection",
        "Video Monitoring",
        "Employee Records"
    ]
)

# =====================================
# MODEL LOADING
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

    st.header("🏠 Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "AI Model",
            "YOLO11"
        )

    with col2:

        st.metric(
            "Technology",
            "Deep Learning"
        )

    with col3:

        st.metric(
            "System Status",
            "Active"
        )

    st.divider()

    st.subheader("Project Features")

    st.write("""
    1. AI Based Person Detection
    2. Employee Tracking
    3. Image Detection
    4. Video Monitoring
    5. Person Counting
    6. Employee Records
    7. CSV Report Generation
    """)

    st.info(
        "Select a module from the sidebar."
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

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("🔍 Detect Persons"):

            with st.spinner("AI is detecting persons..."):

                results = model.predict(
                    image,
                    classes=[0],
                    conf=0.4,
                    verbose=False
                )

                result = results[0]

                detected_image = result.plot()

                detected_image = cv2.cvtColor(
                    detected_image,
                    cv2.COLOR_BGR2RGB
                )

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

    if uploaded_video:

        if st.button("▶️ Start Monitoring"):

            os.makedirs(
                "outputs",
                exist_ok=True
            )

            video_path = "outputs/input_video.mp4"

            with open(video_path, "wb") as file:

                file.write(
                    uploaded_video.getbuffer()
                )

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():

                st.error("Unable to open video.")

                st.stop()

            fps = cap.get(
                cv2.CAP_PROP_FPS
            )

            if not fps or fps <= 0:

                fps = 25

            frame_delay = 1 / fps

            video_display = st.empty()

            status_display = st.empty()

            last_saved = 0

            total_frames = 0

            try:

                while cap.isOpened():

                    start_time = time.time()

                    ret, frame = cap.read()

                    if not ret:

                        break

                    results = model.track(
                        frame,
                        persist=True,
                        classes=[0],
                        conf=0.4,
                        tracker="bytetrack.yaml",
                        verbose=False
                    )

                    result = results[0]

                    annotated_frame = result.plot()

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

                    annotated_frame = cv2.cvtColor(
                        annotated_frame,
                        cv2.COLOR_BGR2RGB
                    )

                    video_display.image(
                        annotated_frame,
                        channels="RGB",
                        use_container_width=True
                    )

                    status_display.info(
                        f"Persons: {person_count} | "
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

                    processing_time = (
                        time.time() - start_time
                    )

                    time.sleep(
                        max(
                            0,
                            frame_delay - processing_time
                        )
                    )

            except Exception as e:

                st.error("Video Processing Error")

                st.exception(e)

            finally:

                cap.release()

            st.success(
                f"Monitoring Completed! Total Frames: {total_frames}"
            )


# =====================================
# EMPLOYEE RECORDS
# =====================================

elif module == "Employee Records":

    st.header("📊 Employee Records")

    if os.path.exists(DATABASE):

        df = pd.read_csv(DATABASE)

        if not df.empty:

            st.dataframe(
                df,
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Total Records",
                    len(df)
                )

            with col2:

                st.metric(
                    "Total Person Detections",
                    int(df["Person_Count"].sum())
                )

            csv_data = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="⬇️ Download Surveillance Report",
                data=csv_data,
                file_name="employee_records.csv",
                mime="text/csv"
            )

            if st.button("🗑️ Clear All Records"):

                pd.DataFrame(
                    columns=COLUMNS
                ).to_csv(
                    DATABASE,
                    index=False
                )

                st.success("All records cleared!")

                st.rerun()

        else:

            st.info("No records available.")

    else:

        st.info("Database not found.")


# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "Smart Surveillance System | "
    "Python | YOLO11 | OpenCV | Deep Learning"
)

