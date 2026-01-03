
    # app.py
import streamlit as st
import numpy as np
import tensorflow as tf
import json
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PlantGuard - Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #22c55e;
        font-size: 3rem;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/plant_disease_recog_model_pwp.keras",
        compile=False
    )

model = load_model()

# ---------------- LOAD DISEASE JSON ----------------
with open("plant_disease.json", "r") as f:
    disease_data = json.load(f)

CLASS_NAMES = [item["name"] for item in disease_data]

# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((160, 160))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------- PREDICTION ----------------
def predict_disease(image):
    img = preprocess_image(image)
    predictions = model.predict(img)
    index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][index])

    # Safety check
    if index >= len(CLASS_NAMES) or confidence < 0.80:
        return {
            "name": "Unknown Disease",
            "confidence": confidence,
            "severity": "Unknown",
            "cause": "Disease not confidently recognized.",
            "cure": "Consult an agricultural expert."
        }

    disease = disease_data[index]

    return {
        "name": disease["name"],
        "confidence": confidence,
        "severity": "None" if "healthy" in disease["name"].lower() else "High",
        "cause": disease["cause"],
        "cure": disease["cure"]
    }

# ---------------- UI ----------------
st.markdown('<h1 class="main-header">🌿 PlantGuard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Plant Disease Detection</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Plant", use_container_width=True):
        with st.spinner("Analyzing image..."):
            result = predict_disease(image)

        st.success("Analysis Complete!")

        col1, col2, col3 = st.columns(3)
        col1.metric("Detected Class", result["name"])
        col2.metric("Confidence", f"{result['confidence']*100:.2f}%")
        col3.metric("Severity", result["severity"])

        st.subheader("📋 Cause")
        st.write(result["cause"])

        st.subheader("💊 Cure / Treatment")
        st.write(result["cure"])
