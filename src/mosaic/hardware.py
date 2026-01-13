"""Hardware detection and configuration for GPU-specific optimizations.

This module centralizes all hardware detection logic to provide consistent
GPU-specific settings across the Mosaic application.
"""

import os
import torch
from loguru import logger

# Detect HuggingFace Spaces ZeroGPU environment
try:
    import spaces

    HAS_SPACES = True
    # Check if we're actually running on ZeroGPU
    # HF Spaces sets SPACES_ZERO_GPU=1 when using ZeroGPU
    IS_ZEROGPU = os.environ.get("SPACES_ZERO_GPU") == "1"
except ImportError:
    HAS_SPACES = False
    IS_ZEROGPU = False

    # Create a no-op decorator if spaces is not available
    class spaces:
        @staticmethod
        def GPU(fn=None, duration=None):
            if fn is None:
                return lambda f: f
            return fn


# Detect GPU hardware type
IS_T4_GPU = False
GPU_NAME = "Unknown"

if not IS_ZEROGPU and torch.cuda.is_available():
    try:
        GPU_NAME = torch.cuda.get_device_name(0)
        IS_T4_GPU = "T4" in GPU_NAME
    except Exception:
        pass


# Set optimal parameters based on hardware
if IS_ZEROGPU:
    DEFAULT_BATCH_SIZE = 128
    DEFAULT_NUM_WORKERS = 0
    DEFAULT_CONCURRENCY_LIMIT = 8  # ZeroGPU manages its own queue
    GPU_TYPE = "ZeroGPU (H100)"
elif IS_T4_GPU:
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_NUM_WORKERS = 4
    DEFAULT_CONCURRENCY_LIMIT = 1  # T4 can only handle one analysis at a time (16GB memory)
    GPU_TYPE = f"T4 ({GPU_NAME})"
else:
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_NUM_WORKERS = 8
    DEFAULT_CONCURRENCY_LIMIT = 8  # High-memory GPUs can handle multiple analyses
    GPU_TYPE = f"Standard GPU ({GPU_NAME})"


# Log hardware detection at module load
logger.info(
    f"Hardware: {GPU_TYPE} | "
    f"batch_size={DEFAULT_BATCH_SIZE}, "
    f"num_workers={DEFAULT_NUM_WORKERS}, "
    f"concurrency_limit={DEFAULT_CONCURRENCY_LIMIT}"
)


# Export commonly used symbols
__all__ = [
    "spaces",
    "HAS_SPACES",
    "IS_ZEROGPU",
    "IS_T4_GPU",
    "GPU_NAME",
    "GPU_TYPE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_NUM_WORKERS",
    "DEFAULT_CONCURRENCY_LIMIT",
]
