"""
Retina AI - Medical-Grade Deep Learning for Retinitis Pigmentosa Classification
Production Streamlit Application for HuggingFace Spaces

This application provides real-time detection of Retinitis Pigmentosa from fundus photographs.
Uses embedded model inference - no external API required.
"""

import streamlit as st
import torch
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import timm
import torch.nn as nn
from torchvision import transforms
import cv2

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Try to find the model
MODEL_PATH = "models/VIT_rp.pth"
if not os.path.exists(MODEL_PATH):
    # Try alternative paths for HuggingFace
    for alt_path in [
        "VIT_rp.pth",
        "../VIT_rp.pth",
        "/home/user/VIT_rp.pth",
    ]:
        if os.path.exists(alt_path):
            MODEL_PATH = alt_path
            break

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🏥 Retina AI Diagnostic System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
    <style>
    .main { background-color: #0a0e27; color: #e0e0e0; }
    .header-gradient {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px; border-radius: 10px; margin-bottom: 20px;
    }
    .alert-rp {
        background-color: #ff4444; color: white; padding: 15px;
        border-radius: 8px; border-left: 5px solid #dd0000;
    }
    .alert-normal {
        background-color: #44ff44; color: black; padding: 15px;
        border-radius: 8px; border-left: 5px solid #00aa00;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model():
    """Load the Vision Transformer model"""
    try:
        # Create model using TIMM
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2)
        
        # Load checkpoint if exists
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
        
        model = model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocessing
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

CLASS_NAMES = ["Normal", "Retinitis Pigmentosa"]

def predict_image(image: Image.Image, model) -> dict:
    """Run inference on an image"""
    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, pred_class].item()
    
    return {
        "predicted_class": CLASS_NAMES[pred_class],
        "confidence": confidence,
        "is_rp": pred_class == 1,
        "all_probabilities": {
            "Normal": probabilities[0, 0].item(),
            "Retinitis Pigmentosa": probabilities[0, 1].item(),
        }
    }

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("⚙️ Settings")
    
    # Model status
    try:
        model = load_model()
        if model is not None:
            st.success("✅ Model Loaded")
            st.caption(f"Device: {DEVICE}")
            st.caption(f"Model: Vision Transformer")
        else:
            st.warning("⚠️ Model not loaded")
    except Exception as e:
        st.error(f"❌ Error: {e}")
    
    st.divider()
    
    st.subheader("👨‍⚕️ Doctor Information")
    doctor_name = st.text_input("Doctor Name", value="Dr. Smith")
    doctor_id = st.text_input("Doctor ID", value="DOC001")

    st.divider()
    
    st.caption(f"**Version:** 1.0.0")
    st.caption(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")

# ============================================================================
# MAIN APP
# ============================================================================

st.markdown("""
    <div class="header-gradient">
        <h1>🏥 Retina AI Diagnostic System</h1>
        <p>Real-time Detection of Retinitis Pigmentosa using Deep Learning</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Upload Fundus Image")
    
    uploaded_file = st.file_uploader(
        "Choose a fundus image (PNG, JPG)",
        type=["png", "jpg", "jpeg"],
        help="Upload a color fundus photograph for analysis"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Fundus Image", width=400)

        patient_id = st.text_input("Patient ID", value="PAT001")

        if st.button("🔬 Run Analysis", use_container_width=True, type="primary"):
            if model is None:
                st.error("Model not loaded. Please refresh the page.")
            else:
                with st.spinner("Analyzing image..."):
                    try:
                        prediction = predict_image(image, model)
                        st.session_state.last_prediction = prediction
                        st.session_state.patient_id = patient_id
                        st.success("✅ Analysis complete!")
                    except Exception as e:
                        st.error(f"Error: {e}")

with col2:
    st.subheader("🔍 Analysis Results")

    if "last_prediction" in st.session_state:
        pred = st.session_state.last_prediction
        
        predicted_class = pred["predicted_class"]
        confidence = pred["confidence"]
        is_rp = pred["is_rp"]

        if is_rp:
            st.markdown(
                f"""
                <div class="alert-rp">
                <h3>⚠️ POSITIVE - Retinitis Pigmentosa Detected</h3>
                <p>Confidence: <strong>{confidence:.1%}</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="alert-normal">
                <h3>✅ NEGATIVE - Normal Retina</h3>
                <p>Confidence: <strong>{confidence:.1%}</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.divider()
        st.subheader("📊 Confidence Scores")

        probs = pred["all_probabilities"]
        col_a, col_b = st.columns(2)

        with col_a:
            st.metric("Normal", f"{probs['Normal']:.1%}")
        with col_b:
            st.metric("Retinitis Pigmentosa", f"{probs['Retinitis Pigmentosa']:.1%}")

        fig = go.Figure(data=[
            go.Bar(
                x=list(probs.keys()),
                y=list(probs.values()),
                marker_color=["#00ff00", "#ff0000"],
                text=[f"{v:.1%}" for v in probs.values()],
                textposition="auto",
            )
        ])

        fig.update_layout(
            title="Class Probability Distribution",
            xaxis_title="Diagnosis",
            yaxis_title="Probability",
            height=300,
            template="plotly_dark",
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Upload an image and click 'Run Analysis' to see results")

# ============================================================================
# DISCLAIMER
# ============================================================================

st.divider()
st.markdown("""
### ⚠️ Disclaimer

This tool is for **research and educational purposes only**. 
It is NOT intended to be used as a substitute for professional medical diagnosis, 
advice, or treatment. Always consult with a qualified healthcare provider for 
medical decisions.

**Version:** 1.0.0 | **Model:** Vision Transformer (ViT)
""")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("🏥 Retina AI Diagnostic System v1.0.0")

with col2:
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col3:
    st.caption("© 2026 Medical AI Systems")
