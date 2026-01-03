import streamlit as st
import numpy as np
import json
from PIL import Image
import tensorflow as tf

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Plant Disease Detection System")
st.write("Upload a clear leaf image to get disease details.")

# --------------------------------------------------
# LOAD TFLITE MODEL
# --------------------------------------------------
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(
        model_path="models/plant_disease_fp16.tflite"
    )
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --------------------------------------------------
# LABELS (ORDER MUST MATCH TRAINING)
# --------------------------------------------------
LABELS = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Background_without_leaves',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# --------------------------------------------------
# LOAD DISEASE JSON
# --------------------------------------------------
@st.cache_data
def load_disease_data():
    with open("plant_disease.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["name"]: item for item in data}

DISEASE_DATA = load_disease_data()

# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------
def preprocess_image(image):
    image = image.resize((160, 160))
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------
def predict(image):
    input_data = preprocess_image(image)

    interpreter.set_tensor(
        input_details[0]['index'],
        input_data
    )
    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]['index']
    )

    index = int(np.argmax(output))
    confidence = float(output[0][index]) * 100

    return LABELS[index], confidence

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Detecting disease..."):
        label, confidence = predict(image)

    # --------------------------------------------------
    # BACKGROUND CHECK
    # --------------------------------------------------
    if label == "Background_without_leaves":
        st.warning("⚠️ No leaf detected. Please upload a clear leaf image.")
        st.stop()

    # --------------------------------------------------
    # MATCH JSON
    # --------------------------------------------------
    if label not in DISEASE_DATA:
        st.error("❌ Disease information not found in JSON.")
        st.stop()

    info = DISEASE_DATA[label]

    # SPLIT PLANT & DISEASE
    plant, disease = label.split("___")
    disease = disease.replace("_", " ")

    # --------------------------------------------------
    # RESULT DISPLAY
    # --------------------------------------------------
    st.success("✅ Detection Result")

    st.markdown(f"### 🌱 Plant Name")
    st.write(plant)

    st.markdown(f"### 🦠 Disease Name")
    st.write(disease)

    st.markdown(f"### 📊 Affected Percentage")
    st.write(f"{confidence:.2f}%")

    st.markdown("### ❓ Cause")
    st.write(info.get("cause", "Not available"))

    st.markdown("### 💊 Cure / Prevention")
    st.write(info.get("cure", "Not available"))

    st.progress(int(confidence))

else:
    st.info("👆 Upload an image to start detection.")
