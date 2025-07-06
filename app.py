import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Optional: suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the fine-tuned model
model_path = r"C:\Users\nancy\Downloads\mobilenetv2_finetuned.h5" # ✅ updated to your best model
model = load_model(model_path)

# Class labels
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Streamlit UI
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload an MRI scan and the AI model will classify the tumor type.")

# Upload input image
uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Load and display image
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

        # Preprocess image
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        with st.spinner("🔍 Classifying..."):
            prediction = model.predict(img_array)[0]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction)

        # Display result
        st.success(f"🩺 Predicted Tumor Type: `{predicted_class}`")
        st.info(f"📊 Confidence Score: `{confidence * 100:.2f}%`")

    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
