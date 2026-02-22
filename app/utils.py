"""
Utility functions for Retina AI system
"""

import logging
from pathlib import Path
from typing import Dict, Any
import json

logger = logging.getLogger(__name__)


def setup_logging(log_file: str = "retina_ai.log") -> None:
    """
    Configure application logging.
    
    Args:
        log_file: Path to log file
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def ensure_directories(paths: list) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        paths: List of directory paths
    """
    for path in paths:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory: {p}")


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path to output file
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON to {file_path}")


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded JSON from {file_path}")
    return data


def format_bytes(bytes_count: int) -> str:
    """
    Format byte count to human-readable format.
    
    Args:
        bytes_count: Number of bytes
        
    Returns:
        Formatted string (e.g., "102.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f} TB"


def format_time_ms(milliseconds: float) -> str:
    """
    Format milliseconds to readable format.
    
    Args:
        milliseconds: Time in milliseconds
        
    Returns:
        Formatted string
    """
    if milliseconds < 1000:
        return f"{milliseconds:.2f} ms"
    else:
        return f"{milliseconds/1000:.2f} s"


class MetricsAggregator:
    """Aggregate and compute statistics from metrics."""

    def __init__(self):
        self.metrics = []

    def add(self, metric: Dict[str, Any]) -> None:
        """Add a metric."""
        self.metrics.append(metric)

    def average(self, key: str) -> float:
        """Get average of metric key."""
        values = [m[key] for m in self.metrics if key in m]
        return sum(values) / len(values) if values else 0.0

    def max(self, key: str) -> float:
        """Get maximum of metric key."""
        values = [m[key] for m in self.metrics if key in m]
        return max(values) if values else 0.0

    def min(self, key: str) -> float:
        """Get minimum of metric key."""
        values = [m[key] for m in self.metrics if key in m]
        return min(values) if values else 0.0

    def summary(self) -> Dict[str, Any]:
        """Get aggregated summary."""
        return {
            "count": len(self.metrics),
            "avg_accuracy": self.average("accuracy"),
            "avg_f1": self.average("f1_score"),
            "avg_auc": self.average("auc_score"),
            "best_accuracy": self.max("accuracy"),
            "best_f1": self.max("f1_score"),
            "best_auc": self.max("auc_score"),
        }
