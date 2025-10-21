"""Inference module for Mosaic.

This module provides the inference interfaces for:
- Aeon: Cancer subtype prediction from WSI features
- Paladin: Biomarker prediction from WSI features
"""

from .aeon import run as run_aeon
from .paladin import run as run_paladin
