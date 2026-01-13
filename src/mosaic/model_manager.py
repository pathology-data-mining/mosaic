"""Model management module for batch processing optimization.

This module provides model loading and caching infrastructure to support
efficient batch processing of multiple slides by loading models once instead
of reloading for each slide.
"""

import gc
import pickle
from pathlib import Path
from typing import Dict, Optional
import torch
from loguru import logger

from mosaic.data_directory import get_data_directory
from mosaic.hardware import IS_T4_GPU, GPU_NAME
from mussel.models import ModelType, get_model_factory


class ModelCache:
    """Container for pre-loaded models with T4-aware memory management.

    This class manages loading and caching of all models used in the slide
    analysis pipeline. It implements adaptive memory management that adjusts
    behavior based on GPU type (T4 vs A100) to avoid out-of-memory errors.

    Attributes:
        ctranspath_model: Pre-loaded CTransPath feature extraction model
        optimus_model: Pre-loaded Optimus feature extraction model
        marker_classifier: Pre-loaded marker classifier model
        aeon_model: Pre-loaded Aeon cancer subtype prediction model
        paladin_models: Dict mapping (cancer_subtype, target) -> model
        is_t4_gpu: Whether running on a T4 GPU (16GB memory)
        aggressive_memory_mgmt: If True, aggressively free Paladin models after use
        device: torch.device for GPU/CPU placement
    """

    def __init__(
        self,
        ctranspath_model=None,
        optimus_model=None,
        marker_classifier=None,
        aeon_model=None,
        is_t4_gpu=False,
        aggressive_memory_mgmt=False,
        device=None,
    ):
        self.ctranspath_model = ctranspath_model
        self.optimus_model = optimus_model
        self.marker_classifier = marker_classifier
        self.aeon_model = aeon_model
        self.paladin_models: Dict[tuple, torch.nn.Module] = {}
        self.is_t4_gpu = is_t4_gpu
        self.aggressive_memory_mgmt = aggressive_memory_mgmt
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def cleanup_paladin(self):
        """Aggressively free all Paladin models from memory.

        Used on T4 GPUs to free memory between inferences.
        """
        if self.paladin_models:
            logger.debug(f"Cleaning up {len(self.paladin_models)} Paladin models")
            for key in list(self.paladin_models.keys()):
                del self.paladin_models[key]
            self.paladin_models.clear()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def cleanup(self):
        """Release all models and free GPU memory.

        Called at the end of batch processing to ensure clean shutdown.
        """
        logger.info("Cleaning up all models from memory")

        # Clean up Paladin models
        self.cleanup_paladin()

        # Clean up core models
        if self.ctranspath_model is not None:
            del self.ctranspath_model
            self.ctranspath_model = None
        if self.optimus_model is not None:
            del self.optimus_model
            self.optimus_model = None
        if self.marker_classifier is not None:
            del self.marker_classifier
            self.marker_classifier = None
        if self.aeon_model is not None:
            del self.aeon_model
            self.aeon_model = None

        # Force garbage collection and GPU cache clearing
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            mem_allocated = torch.cuda.memory_allocated() / (1024**3)
            logger.info(f"GPU memory after cleanup: {mem_allocated:.2f} GB")


def load_all_models(
    use_gpu=True,
    aggressive_memory_mgmt: Optional[bool] = None,
) -> ModelCache:
    """Load core models once for batch processing.

    Loads CTransPath, Optimus, Marker Classifier, and Aeon models into memory.
    Paladin models are loaded on-demand via load_paladin_model_for_inference().

    Args:
        use_gpu: If True, load models to GPU. If False, use CPU.
        aggressive_memory_mgmt: Memory management strategy:
            - None: Auto-detect based on GPU type (T4 = True, A100 = False)
            - True: T4-style aggressive cleanup (load/delete Paladin models)
            - False: A100-style caching (keep Paladin models loaded)

    Returns:
        ModelCache instance with all core models loaded

    Raises:
        FileNotFoundError: If model files are not found in data/ directory
        RuntimeError: If CUDA is requested but not available
    """
    logger.info("=" * 80)
    logger.info("BATCH PROCESSING: Loading models (this happens ONCE per batch)")
    logger.info("=" * 80)

    # Use centralized GPU detection
    device = torch.device("cpu")

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU detected: {GPU_NAME}")
        logger.info(f"GPU total memory: {gpu_memory_total:.2f} GB")

        # Log initial GPU memory
        mem_before = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"GPU memory before loading models: {mem_before:.2f} GB")

        # Auto-detect memory management strategy based on centralized hardware detection
        if aggressive_memory_mgmt is None:
            aggressive_memory_mgmt = IS_T4_GPU
            strategy = "AGGRESSIVE (T4)" if IS_T4_GPU else "CACHING (High-Memory GPU)"
            logger.info(f"Memory management strategy: {strategy}")
            if IS_T4_GPU:
                logger.info("  → Paladin models will be loaded and freed per slide")
            else:
                logger.info(
                    "  → Paladin models will be cached and reused across slides"
                )
    elif use_gpu and not torch.cuda.is_available():
        logger.warning("GPU requested but CUDA not available, falling back to CPU")
        use_gpu = False

    if aggressive_memory_mgmt is None:
        aggressive_memory_mgmt = False

    # Get model data directory (HF cache or local data/)
    data_dir = get_data_directory()
    logger.info(f"Using model data directory: {data_dir}")

    # Load CTransPath model
    logger.info("Loading CTransPath model...")
    ctranspath_path = data_dir / "ctranspath.pth"
    if not ctranspath_path.exists():
        raise FileNotFoundError(f"CTransPath model not found at {ctranspath_path}")

    ctranspath_factory = get_model_factory(ModelType.CTRANSPATH)
    ctranspath_model = ctranspath_factory.get_model(
        str(ctranspath_path), use_gpu=use_gpu, gpu_device_id=0 if use_gpu else None
    )
    logger.info("✓ CTransPath model loaded")

    if use_gpu and torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"  GPU memory: {mem:.2f} GB")

    # Load Optimus model from Hugging Face Hub
    logger.info("Loading Optimus model from bioptimus/H-optimus-0...")
    optimus_factory = get_model_factory(ModelType.OPTIMUS)
    optimus_model = optimus_factory.get_model(
        model_path="hf-hub:bioptimus/H-optimus-0",
        use_gpu=use_gpu,
        gpu_device_id=0 if use_gpu else None,
    )
    logger.info("✓ Optimus model loaded")

    if use_gpu and torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"  GPU memory: {mem:.2f} GB")

    # Load Marker Classifier
    logger.info("Loading Marker Classifier...")
    marker_classifier_path = data_dir / "marker_classifier.pkl"
    if not marker_classifier_path.exists():
        raise FileNotFoundError(
            f"Marker classifier not found at {marker_classifier_path}"
        )

    with open(marker_classifier_path, "rb") as f:
        marker_classifier = pickle.load(f)  # nosec
    logger.info("✓ Marker Classifier loaded")

    if use_gpu and torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"  GPU memory: {mem:.2f} GB")

    # Load Aeon model
    logger.info("Loading Aeon model...")
    aeon_path = data_dir / "aeon_model.pkl"
    if not aeon_path.exists():
        raise FileNotFoundError(f"Aeon model not found at {aeon_path}")

    with open(aeon_path, "rb") as f:
        aeon_model = pickle.load(f)  # nosec
    aeon_model.to(device)
    aeon_model.eval()
    logger.info("✓ Aeon model loaded")

    if use_gpu and torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"  GPU memory: {mem:.2f} GB")

    # Log final memory usage
    logger.info("-" * 80)
    if use_gpu and torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / (1024**3)
        logger.info(f"✓ All core models loaded to GPU")
        logger.info(f"  Total GPU memory used: {mem_allocated:.2f} GB")
        logger.info(f"  These models will be REUSED for all slides in this batch")
    else:
        logger.info("✓ All core models loaded to CPU")
        logger.info("  These models will be REUSED for all slides in this batch")
    logger.info("-" * 80)

    # Create ModelCache
    cache = ModelCache(
        ctranspath_model=ctranspath_model,
        optimus_model=optimus_model,
        marker_classifier=marker_classifier,
        aeon_model=aeon_model,
        is_t4_gpu=IS_T4_GPU,
        aggressive_memory_mgmt=aggressive_memory_mgmt,
        device=device,
    )

    return cache


def load_paladin_model_for_inference(
    cache: ModelCache,
    model_path: Path,
) -> torch.nn.Module:
    """Load a single Paladin model for inference, downloading on-demand if needed.

    Implements adaptive loading strategy:
    - T4 GPU (aggressive mode): Load model fresh, caller must delete after use
    - A100 GPU (caching mode): Check cache, load if needed, return cached model

    If the model file doesn't exist locally, downloads it from HuggingFace Hub.

    Args:
        cache: ModelCache instance managing loaded models
        model_path: Path to the Paladin model file

    Returns:
        Loaded Paladin model ready for inference

    Note:
        On T4 GPUs, caller MUST delete the model and call torch.cuda.empty_cache()
        after inference to avoid OOM errors.
    """
    from huggingface_hub import hf_hub_download

    model_key = str(model_path)

    # Check cache first (only used in non-aggressive mode)
    if not cache.aggressive_memory_mgmt and model_key in cache.paladin_models:
        logger.info(f"  ✓ Using CACHED Paladin model: {model_path.name} (no disk I/O!)")
        return cache.paladin_models[model_key]

    # Download model from HF Hub if it doesn't exist locally
    if not model_path.exists():
        logger.info(
            f"  ⬇ Downloading Paladin model from HuggingFace Hub: {model_path.name}"
        )
        # Extract the relative path from the data directory
        data_dir = get_data_directory()
        relative_path = model_path.relative_to(data_dir)

        downloaded_path = hf_hub_download(
            repo_id="PDM-Group/paladin-aeon-models",
            filename=str(relative_path),
            cache_dir=data_dir.parent.parent,  # Use HF cache directory
        )
        model_path = Path(downloaded_path)
        logger.info(f"  ✓ Downloaded to: {model_path}")

    # Load model from disk
    if cache.aggressive_memory_mgmt:
        logger.info(
            f"  → Loading Paladin model: {model_path.name} (will free after use)"
        )
    else:
        logger.info(
            f"  → Loading Paladin model: {model_path.name} (will cache for reuse)"
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)  # nosec

    model.to(cache.device)
    model.eval()

    # Cache if not in aggressive mode
    if not cache.aggressive_memory_mgmt:
        cache.paladin_models[model_key] = model
        logger.info(f"  ✓ Cached Paladin model for future reuse")

    return model
