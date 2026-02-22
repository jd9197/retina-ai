"""
Error handling and recovery utilities
Provides robust error handling for model loading and inference
"""

import streamlit as st
from utils.logging_config import logger

class ModelLoadError(Exception):
    """Custom exception for model loading failures"""
    pass

class InferenceError(Exception):
    """Custom exception for inference failures"""
    pass

def handle_model_error(error: Exception, context: str = "Model Operation") -> None:
    """
    Handle model-related errors gracefully
    
    Args:
        error: The exception that occurred
        context: Context about where the error occurred
    """
    logger.error(f"{context} Error: {str(error)}")
    st.error(f"⚠️ {context} Failed: {str(error)}")
    st.info("💡 Troubleshooting: Check model file exists, sufficient memory, and proper format")

def handle_inference_error(error: Exception) -> None:
    """
    Handle inference-related errors
    
    Args:
        error: The exception that occurred
    """
    logger.error(f"Inference Error: {str(error)}")
    st.error(f"❌ Prediction Failed: {str(error)}")
    st.warning("Please try with another image or check system logs")

def validate_image_file(uploaded_file) -> bool:
    """
    Validate uploaded image file
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        True if valid, False otherwise
    """
    if uploaded_file is None:
        return False
    
    # Check file size (max 50MB)
    max_size = 50 * 1024 * 1024
    if uploaded_file.size > max_size:
        st.error(f"File too large: {uploaded_file.size / (1024*1024):.2f}MB (max: 50MB)")
        return False
    
    # Check file type
    allowed_types = {"image/jpeg", "image/png", "image/jpg", "image/gif", "image/webp"}
    if uploaded_file.type not in allowed_types:
        st.error(f"Invalid file type: {uploaded_file.type}. Allowed: JPEG, PNG, GIF, WebP")
        return False
    
    logger.info(f"Valid image file: {uploaded_file.name} ({uploaded_file.size} bytes)")
    return True

def log_prediction(image_name: str, prediction: dict, confidence: float) -> None:
    """
    Log prediction event for audit trail
    
    Args:
        image_name: Name of the image file
        prediction: Prediction result
        confidence: Confidence score
    """
    logger.info(
        f"Prediction | Image: {image_name} | Result: {prediction} | "
        f"Confidence: {confidence:.4f}"
    )

def get_logs_path():
    """Get path to logs directory"""
    from pathlib import Path
    return Path(__file__).parent.parent / "logs"
