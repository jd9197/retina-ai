"""
Inference Engine for RP Classification
Handles preprocessing, batch inference, and prediction generation.
Includes Grad-CAM explainability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, List, Any
import cv2
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Image preprocessing pipeline for medical imaging."""

    # ImageNet statistics (standard for pretrained models)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    IMAGE_SIZE = 224

    def __init__(self):
        """Initialize preprocessing pipeline."""
        self.transform = transforms.Compose([
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

    def preprocess(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed tensor of shape (1, 3, 224, 224)
        """
        try:
            image = Image.open(image_path).convert("RGB")
            tensor = self.transform(image)
            return tensor.unsqueeze(0)  # Add batch dimension
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            raise

    def preprocess_batch(self, image_paths: List[str]) -> Tuple[torch.Tensor, List[str]]:
        """
        Preprocess multiple images into a batch.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Tuple of (batch_tensor, valid_paths)
        """
        tensors = []
        valid_paths = []

        for path in image_paths:
            try:
                tensor = self.preprocess(path)
                tensors.append(tensor.squeeze(0))  # Remove batch dim for stacking
                valid_paths.append(path)
            except Exception as e:
                logger.warning(f"Skipped {path}: {e}")

        if not tensors:
            raise ValueError("No valid images in batch")

        batch = torch.stack(tensors, dim=0)
        return batch, valid_paths


class GradCAMExplainer:
    """Grad-CAM for medical image explainability."""

    def __init__(self, model: nn.Module, target_layer: str = None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Name of layer to generate CAM from (auto-selected if None)
        """
        self.model = model
        self.target_layer = target_layer or self._find_target_layer()
        self.activation = None
        self.gradient = None
        self._register_hooks()

    def _find_target_layer(self) -> str:
        """Auto-find a suitable target layer."""
        # Get the last conv layer name
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Conv2d):
                return name
        return ""

    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activation = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0].detach()

        # Find and hook target layer
        target_module = None
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                target_module = module
                break

        if target_module:
            target_module.register_forward_hook(forward_hook)
            target_module.register_backward_hook(backward_hook)

    def generate_cam(self, image_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            image_tensor: Input image tensor (1, 3, H, W)
            class_idx: Target class index
            
        Returns:
            CAM heatmap of shape (H, W)
        """
        image_tensor = image_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(image_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        score = output[:, class_idx].sum()
        score.backward()

        # Compute CAM
        activation = self.activation.squeeze(0)  # (C, H, W)
        gradient = self.gradient.squeeze(0)  # (C, H, W)

        weights = gradient.mean(dim=(1, 2))  # (C,)
        cam = activation * weights.unsqueeze(1).unsqueeze(2)
        cam = cam.sum(dim=0).detach().cpu().numpy()

        # Normalize to 0-1
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)

        return cam


class InferenceEngine:
    """
    Medical inference engine with model management and prediction generation.
    """

    # Model's class_to_idx: {'Normal': 0, 'Retinitis': 1}
    # So: Index 0 = Normal, Index 1 = Retinitis Pigmentosa
    CLASS_NAMES = ["Normal", "Retinitis Pigmentosa"]

    def __init__(self, model: nn.Module, device: torch.device, model_name: str = ""):
        """
        Initialize inference engine.
        
        Args:
            model: Loaded PyTorch model
            device: Computation device
            model_name: Name of model for logging
        """
        self.model = model
        self.device = device
        self.model_name = model_name
        self.preprocessor = ImagePreprocessor()

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Single image prediction with confidence and timing.
        
        Args:
            image_path: Path to fundus image
            
        Returns:
            Dictionary with prediction results
        """
        import time

        start_time = time.time()

        # Preprocess
        image_tensor = self.preprocessor.preprocess(image_path).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)

        # Parse results
        pred_class = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, pred_class].item()
        inference_time = time.time() - start_time

        # Debug logging
        logger.info(f"Image: {image_path}")
        logger.info(f"Raw logits: {logits.cpu().numpy()}")
        logger.info(f"Probabilities: Retinitis Pigmentosa={probabilities[0, 0].item():.4f}, Normal={probabilities[0, 1].item():.4f}")
        logger.info(f"Predicted class: {pred_class}, Predicted class name: {self.CLASS_NAMES[pred_class]}")

        # Get all class probabilities
        class_probs = {}
        for i, class_name in enumerate(self.CLASS_NAMES):
            class_probs[class_name] = float(probabilities[0, i].item())

        return {
            "image_path": str(image_path),
            "predicted_class": self.CLASS_NAMES[pred_class],
            "predicted_class_idx": pred_class,
            "confidence": confidence,
            "all_probabilities": class_probs,
            "is_rp": pred_class == 1,  # RP is class 1 in this model
            "inference_time_ms": round(inference_time * 1000, 2),
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
        }

    def predict_batch(self, image_paths: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Batch inference for multiple images.
        
        Args:
            image_paths: List of image paths
            batch_size: Number of images per batch
            
        Returns:
            List of prediction dictionaries
        """
        all_predictions = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]

            try:
                batch_tensor, valid_paths = self.preprocessor.preprocess_batch(batch_paths)
                batch_tensor = batch_tensor.to(self.device)

                with torch.no_grad():
                    logits = self.model(batch_tensor)
                    probabilities = F.softmax(logits, dim=1)

                # Process each prediction
                for j, path in enumerate(valid_paths):
                    pred_class = probabilities[j].argmax().item()
                    confidence = probabilities[j, pred_class].item()

                    class_probs = {}
                    for k, class_name in enumerate(self.CLASS_NAMES):
                        class_probs[class_name] = float(probabilities[j, k].item())

                    prediction = {
                        "image_path": str(path),
                        "predicted_class": self.CLASS_NAMES[pred_class],
                        "predicted_class_idx": pred_class,
                        "confidence": confidence,
                        "all_probabilities": class_probs,
                        "is_rp": pred_class == 1,  # RP is class 1 in this model
                        "model_name": self.model_name,
                    }
                    all_predictions.append(prediction)

            except Exception as e:
                logger.error(f"Error in batch inference: {e}")

        return all_predictions

    def explain_prediction(self, image_path: str) -> Tuple[Image.Image, np.ndarray]:
        """
        Generate Grad-CAM explanation for a prediction.
        
        Args:
            image_path: Path to image
            
        Returns:
            Tuple of (original_image, cam_heatmap)
        """
        # Load original image
        original_image = Image.open(image_path).convert("RGB")

        # Preprocess
        image_tensor = self.preprocessor.preprocess(image_path).to(self.device)

        # Generate CAM
        explainer = GradCAMExplainer(self.model)
        cam = explainer.generate_cam(image_tensor)

        # Resize CAM to original image size
        cam_resized = cv2.resize(cam, (original_image.width, original_image.height))

        return original_image, cam_resized

    def overlay_heatmap(self, image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
        """
        Overlay Grad-CAM heatmap on original image.
        
        Args:
            image: Original PIL image
            heatmap: CAM heatmap
            alpha: Transparency of overlay
            
        Returns:
            Image with heatmap overlay
        """
        # Convert image to numpy
        image_np = np.array(image)

        # Normalize heatmap to 0-255
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap_colored, alpha, 0)

        return Image.fromarray(overlay)
