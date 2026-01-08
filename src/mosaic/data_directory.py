"""Data directory management for model files.

This module provides a centralized function to locate the directory containing
model data files, checking environment variables and falling back to local paths.
"""

import os
from pathlib import Path


# Global variable to store the model data directory path
_MODEL_DATA_DIR = None


def get_data_directory():
    """Get the directory containing model data files.

    Returns the HuggingFace cache directory path for the model repository,
    or falls back to local 'data/' directory if not yet downloaded.

    Returns:
        Path: Path to the model data directory
    """
    global _MODEL_DATA_DIR
    if _MODEL_DATA_DIR is not None:
        return _MODEL_DATA_DIR

    # Check if environment variable is set
    if "MOSAIC_DATA_DIR" in os.environ:
        _MODEL_DATA_DIR = Path(os.environ["MOSAIC_DATA_DIR"])
        return _MODEL_DATA_DIR

    # Check if local data/ directory exists (for development/backward compat)
    local_data = Path("data")
    if local_data.exists() and (local_data / "paladin_model_map.csv").exists():
        _MODEL_DATA_DIR = local_data
        return _MODEL_DATA_DIR

    # Fall back to repo root data/ directory
    _MODEL_DATA_DIR = local_data
    return _MODEL_DATA_DIR


def set_data_directory(path):
    """Set the data directory path.

    Args:
        path: Path to the data directory
    """
    global _MODEL_DATA_DIR
    _MODEL_DATA_DIR = Path(path)
    os.environ["MOSAIC_DATA_DIR"] = str(_MODEL_DATA_DIR)
