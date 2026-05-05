# ==============================
# 🍓 Strawberry Disease Detector
# ==============================

import streamlit as st
from PIL import Image
import numpy as np
import joblib
import cv2
from skimage.feature import local_binary_pattern

# ==============================
# 📌 Load Model + Scaler
# ==============================
model = joblib.load("model.pkl")
label_map = joblib.load("label_map.pkl")
scaler = joblib.load("scaler.pkl")

reverse_map = {v: k for k, v in label_map.items()}

# ==============================
# 📌 Feature Extraction
# ==============================
def extract_features(image):

    # RGB
    mean = image.mean(axis=(0,1))
    std = image.std(axis=(0,1))

    # HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_mean = hsv.mean(axis=(0,1))
    hsv_std = hsv.std(axis=(0,1))

    # LBP
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")

    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0,10))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    # Edges
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    # HSV Histogram
    hsv_hist = cv2.calcHist([hsv], [0,1], None, [8,8], [0,180,0,256])
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()

    # White pixels (Powdery Mildew)
    white_pixels = np.sum(
        (image[:,:,0] > 200) &
        (image[:,:,1] > 200) &
        (image[:,:,2] > 200)
    ) / image.size

    return np.concatenate([
        mean, std,
        hsv_mean, hsv_std,
        hist,
        hsv_hist,
        [edge_density],
        [white_pixels]
    ])

# ==============================
# 📌 Treatments
# ==============================
treatments = {
    "healthy": "Plant is healthy, no treatment needed",
    "Leaf Spot": "Use resistant plants",
    "Gray Mold": "Improve ventilation",
    "Blossom Blight": "Apply fungicide",
    "Anthracnose Fruit Rot": "Remove infected fruits",
    "Angular Leafspot": "Use copper fungicide",
    "Powdery Mildew Fruit": "Apply sulfur spray",
    "Powdery Mildew Leaf": "Avoid humidity"
}

# ==============================
# 📌 UI
# ==============================
st.set_page_config(page_title="Strawberry Disease Detector", page_icon="🍓")

st.title("🍓 Strawberry Disease Detector")
st.markdown("## 🌿 AI Strawberry Disease Detection")
st.info("📸 Upload a clear image of strawberry leaf or fruit")

file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# ==============================
# 📌 Prediction
# ==============================
if file is not None:
    try:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Image")

        img = image.resize((224, 224))
        img = np.array(img)

        features = extract_features(img)
        features = scaler.transform([features])

        probs = model.predict_proba(features)[0]
        pred = np.argmax(probs)

        confidence = probs[pred]
        disease = reverse_map[pred]

        st.success(f"🦠 Disease: {disease}")



        st.info(f"💊 Treatment: {treatments.get(disease, 'No treatment found')}")

        # Similar classes warning
        sorted_probs = np.sort(probs)[::-1]
        if abs(sorted_probs[0] - sorted_probs[1]) < 0.1:
            st.warning("⚠️ Similar diseases detected, result may be uncertain")

    except Exception as e:
        st.error(f"❌ Error: {e}")