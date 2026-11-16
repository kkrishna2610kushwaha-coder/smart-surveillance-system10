
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Fake Currency Detection System",
    page_icon="💵",
    layout="wide"
)

# File paths
MODEL_PATH = "currency_model.keras"
CLASSES_PATH = "classes.json"
HISTORY_PATH = "detection_history.csv"

IMAGE_SIZE = (224, 224)


# Load AI Model
@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model(MODEL_PATH)


# Load class names
def load_classes():
    with open(CLASSES_PATH, "r") as file:
        data = json.load(file)
    return data["class_names"]


# Save detection history
def save_record(filename, prediction, confidence):

    record = pd.DataFrame([{
        "Date": datetime.now().strftime("%d-%m-%Y"),
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Image": filename,
        "Prediction": prediction,
        "Confidence": round(confidence * 100, 2)
    }])

    if os.path.exists(HISTORY_PATH):
        record.to_csv(
            HISTORY_PATH,
            mode="a",
            header=False,
            index=False
        )
    else:
        record.to_csv(HISTORY_PATH, index=False)


# Sidebar
st.sidebar.title("💵 Currency Detection")

menu = st.sidebar.selectbox(
    "Choose Module",
    [
        "Home",
        "Currency Detection",
        "Detection History",
        "About Project"
    ]
)


# HOME PAGE
if menu == "Home":

    st.title("💵 Fake Currency Detection System")

    st.subheader(
        "Indian Currency Detection using Deep Learning"
    )

    st.write(
        "This AI system uses a CNN model to classify "
        "currency note images into Fake and Genuine categories."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("AI Technology", "CNN")
    col2.metric("Framework", "TensorFlow")
    col3.metric("System", "Active")

    st.divider()

    st.header("Project Features")

    st.markdown("""
    - Upload currency note images
    - AI-based currency classification
    - Fake and Genuine prediction
    - Confidence score
    - Detection history
    - Download detection report
    """)

    st.info(
        "Upload an image from the Currency Detection module "
        "to start the prediction."
    )


# CURRENCY DETECTION PAGE
elif menu == "Currency Detection":

    st.title("🔍 Currency Detection")

    uploaded_file = st.file_uploader(
        "Upload Currency Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Currency Image",
                use_container_width=True
            )

        with col2:

            if st.button("Detect Currency", type="primary"):

                if not os.path.exists(MODEL_PATH):
                    st.error("AI Model not found. Run train.py first.")

                elif not os.path.exists(CLASSES_PATH):
                    st.error("classes.json file not found.")

                else:

                    with st.spinner("AI is analysing the image..."):

                        model = load_ai_model()
                        class_names = load_classes()

                        # Image preprocessing
                        image = image.resize(IMAGE_SIZE)

                        image_array = np.array(
                            image,
                            dtype=np.float32
                        )

                        image_array = np.expand_dims(
                            image_array,
                            axis=0
                        )

                        # Model prediction
                        score = float(
                            model.predict(
                                image_array,
                                verbose=0
                            )[0][0]
                        )

                        # Sigmoid output represents class index 1
                        probabilities = [1 - score, score]

                        prediction_index = int(
                            np.argmax(probabilities)
                        )

                        prediction = class_names[prediction_index]

                        confidence = probabilities[prediction_index]

                        st.divider()

                        if prediction.lower() == "genuine":

                            st.success(
                                "✅ AI Prediction: GENUINE CURRENCY"
                            )

                        else:

                            st.error(
                                "⚠️ AI Prediction: FAKE CURRENCY"
                            )

                        st.metric(
                            "Prediction Confidence",
                            f"{confidence * 100:.2f}%"
                        )

                        st.progress(float(confidence))

                        st.caption(
                            "Confidence is the model's score, "
                            "not a guarantee of authenticity."
                        )

                        # Save record
                        save_record(
                            uploaded_file.name,
                            prediction,
                            confidence
                        )

                        st.success("Detection record saved!")


# DETECTION HISTORY PAGE
elif menu == "Detection History":

    st.title("📊 Detection History")

    if os.path.exists(HISTORY_PATH):

        df = pd.read_csv(HISTORY_PATH)

        st.dataframe(
            df,
            use_container_width=True
        )

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Detection Report",
            data=csv,
            file_name="currency_detection_report.csv",
            mime="text/csv"
        )

    else:

        st.info("No detection records available yet.")


# ABOUT PROJECT PAGE
elif menu == "About Project":

    st.title("About Fake Currency Detection System")

    st.write("""
    This project is developed using Artificial Intelligence
    and Deep Learning.

    A Convolutional Neural Network (CNN) is trained to
    classify currency images into two categories:

    1. Fake Currency
    2. Genuine Currency

    Technologies Used:

    - Python
    - TensorFlow
    - CNN
    - NumPy
    - Pandas
    - Pillow
    - Streamlit
    """)

    st.warning(
        "This is an educational image-classification prototype. "
        "It cannot verify all physical security features of a banknote. "
        "Do not use its prediction as proof of authenticity."
    )


# Footer
st.divider()

st.caption(
    "Fake Currency Detection System | "
    "Python | CNN | TensorFlow | Streamlit"
)
