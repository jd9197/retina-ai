"""
Production FastAPI Backend for Retinitis Pigmentosa Classification
Implements:
- Real-time inference API
- Batch prediction support
- Medical report generation
- Grad-CAM explainability
- Database logging
- Error handling & validation
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import logging
import json
import sqlite3
import tempfile
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import io
import uuid

from app.model_loader import ModelLoader
from app.inference import InferenceEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize application
app = FastAPI(
    title="🏥 Retina AI Diagnostic API",
    description="Medical-grade Deep Learning API for Retinitis Pigmentosa Classification",
    version="1.0.0",
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded once on startup)
_model = None
_inference_engine = None
_device = None
_db_path = Path("./data/predictions.db")


class ModelSingleton:
    """Singleton pattern for model loading - ensures model loads only once."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize model on first access."""
        global _model, _inference_engine, _device

        try:
            logger.info("Initializing model...")

            # Load model
            model_loader = ModelLoader(device="auto")
            _device = model_loader.device
            logger.info(f"Device: {_device}")

            model_path = Path("models/VIT_rp.pth")
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}. Current directory: {Path.cwd()}")

            logger.info(f"Loading model from {model_path}...")
            
            # Load ViT model (98% accuracy - best performing model)
            try:
                model, info = model_loader.load_model("vit", str(model_path))
                logger.info(f"✓ Model architecture loaded successfully (ViT - Production Model)")
            except Exception as model_load_error:
                logger.error(f"Failed to load model architecture: {model_load_error}", exc_info=True)
                raise RuntimeError(f"Model loading failed: {str(model_load_error)}")

            _model = model
            _inference_engine = InferenceEngine(model, _device, "best_model")

            logger.info("✓ Model loaded successfully")
            logger.info(f"  Device: {_device}")
            logger.info(f"  Parameters: {info['total_params']:,}")
            logger.info(f"  Model Size: {info.get('model_size_mb', 0)}MB")

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}", exc_info=True)
            raise

    @property
    def model(self):
        return _model

    @property
    def inference_engine(self):
        return _inference_engine

    @property
    def device(self):
        return _device


@app.on_event("startup")
async def startup_event():
    """Initialize on application startup."""
    logger.info("🚀 Starting Retina AI API...")

    # Load model
    ModelSingleton()

    # Initialize database
    init_database()

    logger.info("✅ API ready for requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Shutting down Retina AI API...")


def init_database():
    """Initialize SQLite database for prediction logging."""
    _db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(_db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            predicted_class TEXT,
            confidence REAL,
            all_probabilities TEXT,
            model_name TEXT,
            is_rp INTEGER,
            patient_id TEXT,
            prediction_time_ms REAL
        )
    """)

    conn.commit()
    conn.close()


def log_prediction(prediction: Dict, patient_id: Optional[str] = None):
    """Log prediction to database."""
    try:
        prediction_id = str(uuid.uuid4())

        conn = sqlite3.connect(_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO predictions
            (id, timestamp, predicted_class, confidence, all_probabilities,
             model_name, is_rp, patient_id, prediction_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prediction_id,
            datetime.now().isoformat(),
            prediction.get("predicted_class"),
            prediction.get("confidence"),
            json.dumps(prediction.get("all_probabilities", {})),
            prediction.get("model_name"),
            int(prediction.get("is_rp", False)),
            patient_id,
            prediction.get("inference_time_ms"),
        ))

        conn.commit()
        conn.close()

        return prediction_id

    except Exception as e:
        logger.error(f"Error logging prediction: {e}")
        return None


@app.get("/")
async def root():
    """API information endpoint."""
    return {
        "name": "🏥 Retina AI Diagnostic API",
        "version": "1.0.0",
        "description": "Medical-grade Deep Learning API for Retinitis Pigmentosa Classification",
        "endpoints": {
            "POST /predict": "Single image prediction",
            "POST /predict-batch": "Batch prediction",
            "POST /explain": "Grad-CAM explainability",
            "GET /health": "Health check",
            "GET /stats": "API statistics",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        model = ModelSingleton().model
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "device": str(_device),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
    doctor_id: Optional[str] = Form(None),
):
    """
    Single image prediction endpoint.

    Request:
        - file: Image file (PNG, JPG)
        - patient_id: Optional patient identifier
        - doctor_id: Optional doctor identifier

    Response:
        - predicted_class: "Normal" or "Retinitis Pigmentosa"
        - confidence: Confidence score (0-1)
        - all_probabilities: Full probability distribution
        - is_rp: Binary flag for RP diagnosis
        - inference_time_ms: Model inference time
    """
    temp_path = None
    try:
        # Validate file
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Only PNG and JPG images are supported")

        logger.info(f"Processing image: {file.filename}, size: {len(await file.read())} bytes")
        await file.seek(0)  # Reset file pointer after reading
        
        # Save temp file using Python tempfile (cross-platform)
        temp_fd, temp_path = tempfile.mkstemp(suffix='.png', prefix='retina_predict_')
        try:
            content = await file.read()
            with os.fdopen(temp_fd, 'wb') as f:
                f.write(content)

            logger.info(f"Temp file created: {temp_path}")

            # Inference
            try:
                inference_engine = ModelSingleton().inference_engine
                if inference_engine is None:
                    raise RuntimeError("Inference engine not initialized")
                
                prediction = inference_engine.predict(temp_path)
                logger.info(f"Prediction successful: {prediction['predicted_class']} ({prediction['confidence']:.2%})")
            except Exception as inference_error:
                logger.error(f"Inference failed: {inference_error}", exc_info=True)
                raise RuntimeError(f"Model inference failed: {str(inference_error)}")

            # Log to database
            prediction_id = log_prediction(prediction, patient_id)
            prediction["prediction_id"] = prediction_id
            prediction["patient_id"] = patient_id
            prediction["doctor_id"] = doctor_id

            return prediction
            
        finally:
            # Robust cleanup
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    logger.debug(f"Cleaned up temp file: {temp_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    patient_id: Optional[str] = Form(None),
):
    """
    Batch prediction endpoint.

    Request:
        - files: List of image files
        - patient_id: Optional patient identifier

    Response:
        - predictions: List of prediction results
        - batch_stats: Summary statistics
    """
    temp_paths = []
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        if len(files) > 100:
            raise HTTPException(status_code=400, detail="Maximum 100 images per batch")

        # Save temp files
        try:
            for file in files:
                if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                temp_fd, temp_path = tempfile.mkstemp(suffix='.png', prefix='retina_batch_')
                content = await file.read()
                with os.fdopen(temp_fd, 'wb') as f:
                    f.write(content)
                temp_paths.append(temp_path)

            # Batch inference
            inference_engine = ModelSingleton().inference_engine
            predictions = inference_engine.predict_batch(temp_paths)

            # Log predictions
            for pred in predictions:
                log_prediction(pred, patient_id)

            # Compute batch statistics
            rp_count = sum(1 for p in predictions if p["is_rp"])
            normal_count = len(predictions) - rp_count
            avg_confidence = sum(p["confidence"] for p in predictions) / len(predictions) if predictions else 0

            batch_stats = {
                "total_images": len(predictions),
                "rp_cases": rp_count,
                "normal_cases": normal_count,
                "avg_confidence": round(avg_confidence, 4),
                "processing_time_s": sum(p.get("inference_time_ms", 0) for p in predictions) / 1000,
            }

            return {
                "predictions": predictions,
                "batch_stats": batch_stats,
                "patient_id": patient_id,
            }

        finally:
            # Robust cleanup of all temp files
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to cleanup temp file {temp_path}: {cleanup_error}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/explain")
async def explain_prediction(file: UploadFile = File(...)):
    """
    Generate Grad-CAM explanation for prediction.

    Request:
        - file: Image file

    Response:
        - prediction: Initial prediction
        - heatmap_base64: Base64-encoded heatmap
    """
    temp_path = None
    try:
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Only PNG and JPG images are supported")

        # Save temp file using cross-platform tempfile module
        temp_fd, temp_path_str = tempfile.mkstemp(suffix='.png', prefix='retina_explain_')
        try:
            content = await file.read()
            with os.fdopen(temp_fd, 'wb') as f:
                f.write(content)

            # Get prediction and explanation
            inference_engine = ModelSingleton().inference_engine
            prediction = inference_engine.predict(temp_path_str)

            original_image, heatmap = inference_engine.explain_prediction(temp_path_str)
            heatmap_image = inference_engine.overlay_heatmap(original_image, heatmap)

            # Convert to base64
            import base64
            buffered = io.BytesIO()
            heatmap_image.save(buffered, format="PNG")
            heatmap_base64 = base64.b64encode(buffered.getvalue()).decode()

            return {
                "prediction": prediction,
                "heatmap_image": heatmap_base64,
            }

        finally:
            # Robust cleanup
            if temp_path_str and os.path.exists(temp_path_str):
                try:
                    os.unlink(temp_path_str)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temp file {temp_path_str}: {cleanup_error}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get API usage statistics."""
    try:
        conn = sqlite3.connect(_db_path)
        cursor = conn.cursor()

        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]

        # RP vs Normal
        cursor.execute("SELECT is_rp, COUNT(*) FROM predictions GROUP BY is_rp")
        results = cursor.fetchall()
        rp_count = next((r[1] for r in results if r[0] == 1), 0)
        normal_count = next((r[1] for r in results if r[0] == 0), 0)

        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM predictions")
        avg_confidence = cursor.fetchone()[0] or 0

        conn.close()

        return {
            "total_predictions": total_predictions,
            "rp_cases": rp_count,
            "normal_cases": normal_count,
            "avg_confidence": round(avg_confidence, 4),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
