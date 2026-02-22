"""
Model Loader for Retinitis Pigmentosa Classification
Handles loading of 5 different model architectures
Device-aware and optimized for GPU/CPU inference
Includes memory optimization for resource-constrained devices (Raspberry Pi, etc)
"""

import torch
import torch.nn as nn
from torchvision import models
import timm
from pathlib import Path
from typing import Tuple, Dict, Any
import logging
import psutil
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Centralized model loader for all RP classification models.
    Supports: EfficientNet, MobileNet, ResNet50, Swin Transformer, ViT
    """

    def __init__(self, device: str = None):
        """
        Initialize model loader with device selection.
        Args:
            device: 'cuda', 'cpu', or 'auto'. If auto, uses GPU if available.
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        logger.info(f"Using device: {self.device}")
        self._log_system_info()

        # Model info
        self.models_info = {
            "efficientnet": {
                "name": "EfficientNet-B0",
                "source": "torchvision",
                "num_classes": 2,
            },
            "mobilenet": {
                "name": "MobileNetV2",
                "source": "torchvision",
                "num_classes": 2,
            },
            "resnet50": {
                "name": "ResNet50",
                "source": "torchvision",
                "num_classes": 2,
            },
            "swin": {
                "name": "Swin Transformer",
                "source": "timm",
                "num_classes": 2,
            },
            "vit": {
                "name": "Vision Transformer",
                "source": "timm",
                "num_classes": 2,
            },
        }

    def _log_system_info(self):
        """Log system information for debugging."""
        try:
            # CPU info
            cpu_count = os.cpu_count()
            logger.info(f"CPU Cores: {cpu_count}")
            
            # Memory info
            mem = psutil.virtual_memory()
            logger.info(f"RAM Available: {mem.available / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB")
            
            # GPU info if available
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        except Exception as e:
            logger.warning(f"Could not log system info: {e}")

    def _check_memory(self, required_mb: float = 500):
        """
        Check if sufficient memory is available.
        
        Args:
            required_mb: Required memory in MB
            
        Returns:
            tuple: (has_sufficient_memory: bool, available_mb: float)
        """
        try:
            mem = psutil.virtual_memory()
            available_mb = mem.available / (1024**2)
            
            if available_mb < required_mb:
                logger.warning(f"Low memory: {available_mb:.0f}MB available, {required_mb:.0f}MB required")
                return False, available_mb
            return True, available_mb
        except Exception as e:
            logger.warning(f"Could not check memory: {e}")
            return True, -1  # Assume sufficient if we can't check

    def _create_efficientnet(self, num_classes: int = 2) -> nn.Module:
        """Create EfficientNet-B0 architecture."""
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(1280, num_classes)
        return model

    def _create_mobilenet(self, num_classes: int = 2) -> nn.Module:
        """Create MobileNetV2 architecture."""
        model = models.mobilenet_v2(weights=None)
        model.classifier[-1] = nn.Linear(1280, num_classes)
        return model

    def _create_resnet50(self, num_classes: int = 2) -> nn.Module:
        """Create standard ResNet50 architecture."""
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(2048, num_classes)
        return model

    def _create_swin(self, num_classes: int = 2) -> nn.Module:
        """Create Swin Transformer architecture."""
        model = timm.create_model("swin_tiny_patch4_window7_224", num_classes=num_classes, pretrained=False)
        return model

    def _create_vit(self, num_classes: int = 2) -> nn.Module:
        """Create Vision Transformer architecture."""
        model = timm.create_model("vit_base_patch16_224", num_classes=num_classes, pretrained=False)
        return model

    def load_model(self, model_name: str, checkpoint_path: str) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load a model from checkpoint.
        
        Args:
            model_name: Name of model ('efficientnet', 'mobilenet', 'resnet50', 'swin', 'vit')
            checkpoint_path: Path to .pth checkpoint file
            
        Returns:
            Tuple of (model, model_info)
        """
        model_name = model_name.lower()

        if model_name not in self.models_info:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.models_info.keys())}")

        logger.info(f"Loading {model_name}...")

        # Check memory before loading
        has_mem, available_mb = self._check_memory(required_mb=500)
        if not has_mem:
            logger.warning(f"Low memory condition detected: {available_mb:.0f}MB available")

        # Create architecture
        try:
            if model_name == "efficientnet":
                model = self._create_efficientnet()
            elif model_name == "mobilenet":
                model = self._create_mobilenet()
            elif model_name == "resnet50":
                model = self._create_resnet50()
            elif model_name == "swin":
                model = self._create_swin()
            elif model_name == "vit":
                model = self._create_vit()
            
            logger.info(f"{model_name} architecture created successfully")
        except Exception as arch_error:
            logger.error(f"Failed to create {model_name} architecture: {arch_error}", exc_info=True)
            raise RuntimeError(f"Model architecture creation failed: {str(arch_error)}")

        # Load weights
        try:
            checkpoint_path_obj = Path(checkpoint_path)
            if not checkpoint_path_obj.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
            
            logger.info(f"Loading weights from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            logger.info(f"Checkpoint loaded successfully")
            
            # Handle checkpoint format - extract model_state_dict if present
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                    logger.info("Extracted model_state_dict from checkpoint")
                else:
                    state_dict = checkpoint
            else:
                state_dict = {"state_dict": checkpoint}
            
            # Try loading as-is first
            try:
                model.load_state_dict(state_dict, strict=True)
                logger.info("Weights loaded with exact match")
            except RuntimeError as e:
                logger.warning(f"Exact weight loading failed, attempting with strict=False: {str(e)[:200]}")
                try:
                    # Try non-strict loading which allows for shape mismatches
                    model.load_state_dict(state_dict, strict=False)
                    logger.info("Weights loaded in non-strict mode")
                except RuntimeError as e2:
                    logger.error(f"Non-strict loading also failed: {e2}")
                    raise
            
            logger.info(f"Successfully loaded {model_name} from {checkpoint_path}")
        except FileNotFoundError as fnf_error:
            logger.error(f"Checkpoint file error: {fnf_error}")
            raise
        except Exception as e:
            logger.error(f"Error loading {model_name}: {e}", exc_info=True)
            raise RuntimeError(f"Model weights loading failed: {str(e)}")

        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()

        # Collect model info
        model_info = self.models_info[model_name].copy()
        model_info["device"] = str(self.device)
        model_info["total_params"] = sum(p.numel() for p in model.parameters())
        model_info["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Calculate model size in MB
        model_path = Path(checkpoint_path)
        if model_path.exists():
            model_info["model_size_mb"] = round(model_path.stat().st_size / (1024 * 1024), 2)
        else:
            model_info["model_size_mb"] = 0

        return model, model_info

    def load_all_models(self, model_dir: str) -> Dict[str, Tuple[nn.Module, Dict[str, Any]]]:
        """
        Load all 5 models from directory.
        
        Args:
            model_dir: Directory containing all .pth files
            
        Returns:
            Dictionary of {model_name: (model, info)}
        """
        model_dir = Path(model_dir)
        models_dict = {}

        model_mapping = {
            "efficientnet_rp_model.pth": "efficientnet",
            "mobilenet.pth": "mobilenet",
            "resnet50_retinitis_classifier.pth": "resnet50",
            "swin_retina_rp_deploy.pth": "swin",
            "VIT_rp.pth": "vit",
        }

        for file_name, model_name in model_mapping.items():
            model_path = model_dir / file_name
            if model_path.exists():
                try:
                    model, info = self.load_model(model_name, str(model_path))
                    models_dict[model_name] = (model, info)
                    logger.info(f"Loaded {model_name} - Params: {info['total_params']}, Size: {info['model_size_mb']}MB")
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
            else:
                logger.warning(f"Model file not found: {model_path}")

        if not models_dict:
            raise RuntimeError("No models could be loaded. Check model directory and file names.")

        logger.info(f"Successfully loaded {len(models_dict)} models")
        return models_dict

    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get information about available compute devices."""
        return {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "pytorch_version": torch.__version__,
        }
