"""Core slide analysis module for Mosaic.

This module provides the main slide analysis pipeline that integrates tissue segmentation,
feature extraction, and model inference for cancer subtype and biomarker prediction.
"""

# Import spaces first before any CUDA-related imports
import os

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

# Detect T4 hardware by checking actual GPU
import torch
IS_T4_GPU = False
GPU_NAME = "Unknown"
if not IS_ZEROGPU and torch.cuda.is_available():
    try:
        GPU_NAME = torch.cuda.get_device_name(0)
        IS_T4_GPU = "T4" in GPU_NAME
    except:
        pass

# Set optimal parameters based on hardware
if IS_ZEROGPU:
    DEFAULT_BATCH_SIZE = 128
    DEFAULT_NUM_WORKERS = 0
    GPU_TYPE = "ZeroGPU (H100)"
elif IS_T4_GPU:
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_NUM_WORKERS = 4
    GPU_TYPE = f"T4 ({GPU_NAME})"
else:
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_NUM_WORKERS = 8
    GPU_TYPE = f"Standard GPU ({GPU_NAME})"

import pickle
import pandas as pd
import gradio as gr
from pathlib import Path
from mussel.models import ModelType
from mussel.utils import get_features, segment_tissue, filter_features
from mussel.utils.segment import draw_slide_mask
from mussel.cli.tessellate import BiopsySegConfig, ResectionSegConfig, TcgaSegConfig
from loguru import logger
from mosaic.inference import run_aeon, run_paladin
from mosaic.data_directory import get_data_directory

# Log hardware detection at module load
logger.info(f"Hardware: {GPU_TYPE} | batch_size={DEFAULT_BATCH_SIZE}, num_workers={DEFAULT_NUM_WORKERS}")


def _extract_ctranspath_features(coords, slide_path, attrs, num_workers):
    """Extract CTransPath features on GPU.
    
    Args:
        coords: Tissue tile coordinates
        slide_path: Path to the whole slide image file
        attrs: Slide attributes
        num_workers: Number of worker processes
        
    Returns:
        tuple: (ctranspath_features, coords)
    """
    if IS_ZEROGPU:
        num_workers = 0
        batch_size = 128
        logger.info(f"Running CTransPath on ZeroGPU: processing {len(coords)} tiles")
    elif IS_T4_GPU:
        num_workers = DEFAULT_NUM_WORKERS
        batch_size = DEFAULT_BATCH_SIZE
        logger.info(f"Running CTransPath on T4: processing {len(coords)} tiles with batch_size={batch_size}")
    else:
        num_workers = max(num_workers, 8)
        batch_size = 64
        logger.info(f"Running CTransPath with {num_workers} workers")
    
    start_time = pd.Timestamp.now()

    data_dir = get_data_directory()
    ctranspath_features, _ = get_features(
        coords,
        slide_path,
        attrs,
        model_type=ModelType.CTRANSPATH,
        model_path=str(data_dir / "ctranspath.pth"),
        num_workers=num_workers,
        batch_size=batch_size,
        use_gpu=True,
    )
    
    end_time = pd.Timestamp.now()
    logger.info(f"CTransPath extraction took {end_time - start_time}")
    
    return ctranspath_features, coords


def _extract_optimus_features(filtered_coords, slide_path, attrs, num_workers):
    """Extract Optimus features on GPU.
    
    Args:
        filtered_coords: Filtered tissue tile coordinates
        slide_path: Path to the whole slide image file
        attrs: Slide attributes
        num_workers: Number of worker processes
        
    Returns:
        Optimus features
    """
    if IS_ZEROGPU:
        num_workers = 0
        batch_size = 128
        logger.info(f"Running Optimus on ZeroGPU: processing {len(filtered_coords)} tiles")
    elif IS_T4_GPU:
        num_workers = DEFAULT_NUM_WORKERS
        batch_size = DEFAULT_BATCH_SIZE
        logger.info(f"Running Optimus on T4: processing {len(filtered_coords)} tiles with batch_size={batch_size}")
    else:
        num_workers = max(num_workers, 8)
        batch_size = 64
        logger.info(f"Running Optimus with {num_workers} workers")
    
    start_time = pd.Timestamp.now()

    data_dir = get_data_directory()
    features, _ = get_features(
        filtered_coords,
        slide_path,
        attrs,
        model_type=ModelType.OPTIMUS,
        model_path=str(data_dir / "optimus.pkl"),
        num_workers=num_workers,
        batch_size=batch_size,
        use_gpu=True,
    )
    
    end_time = pd.Timestamp.now()
    logger.info(f"Optimus extraction took {end_time - start_time}")
    
    return features


def _run_aeon_inference(features, site_type, num_workers, sex=None, tissue_site_idx=None):
    """Run Aeon cancer subtype inference on GPU.
    
    Args:
        features: Optimus features
        site_type: Site type ("Primary" or "Metastatic")
        num_workers: Number of worker processes
        sex: Patient sex (0=Male, 1=Female), optional
        tissue_site_idx: Tissue site index (0-56), optional
        
    Returns:
        Aeon results DataFrame
    """
    if IS_ZEROGPU:
        num_workers = 0
        logger.info("Running Aeon on ZeroGPU: setting num_workers=0")
    elif IS_T4_GPU:
        num_workers = DEFAULT_NUM_WORKERS
        logger.info(f"Running Aeon on T4 with num_workers={num_workers}")
    else:
        num_workers = max(num_workers, 8)
        logger.info(f"Running Aeon with num_workers={num_workers}")
    
    start_time = pd.Timestamp.now()
    logger.info("Running Aeon for cancer subtype inference")
    data_dir = get_data_directory()
    aeon_results, _ = run_aeon(
        features=features,
        model_path=str(data_dir / "aeon_model.pkl"),
        metastatic=(site_type == "Metastatic"),
        batch_size=8,
        num_workers=num_workers,
        sex=sex,
        tissue_site_idx=tissue_site_idx,
        use_cpu=False,
    )
    end_time = pd.Timestamp.now()
    
    # Log memory stats if CUDA is available
    if torch.cuda.is_available():
        try:
            max_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)
            logger.info(
                f"Aeon inference took {end_time - start_time} and used {max_gpu_memory:.2f} GB GPU memory"
            )
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            logger.info(f"Aeon inference took {end_time - start_time}")
    else:
        logger.info(f"Aeon inference took {end_time - start_time}")
    
    return aeon_results


def _run_paladin_inference(features, aeon_results, site_type, num_workers):
    """Run Paladin biomarker inference on GPU.
    
    Args:
        features: Optimus features
        aeon_results: Aeon results DataFrame
        site_type: Site type ("Primary" or "Metastatic")
        num_workers: Number of worker processes
        
    Returns:
        Paladin results DataFrame
    """
    if IS_ZEROGPU:
        num_workers = 0
        logger.info("Running Paladin on ZeroGPU: setting num_workers=0")
    elif IS_T4_GPU:
        num_workers = DEFAULT_NUM_WORKERS
        logger.info(f"Running Paladin on T4 with num_workers={num_workers}")
    else:
        num_workers = max(num_workers, 8)
        logger.info(f"Running Paladin with num_workers={num_workers}")
    
    start_time = pd.Timestamp.now()
    logger.info("Running Paladin for biomarker inference")
    data_dir = get_data_directory()
    paladin_results = run_paladin(
        features=features,
        model_map_path=str(data_dir / "paladin_model_map.csv"),
        aeon_results=aeon_results,
        metastatic=(site_type == "Metastatic"),
        batch_size=8,
        num_workers=num_workers,
        use_cpu=False,
    )
    end_time = pd.Timestamp.now()
    
    # Log memory stats if CUDA is available
    if torch.cuda.is_available():
        try:
            max_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)
            logger.info(
                f"Paladin inference took {end_time - start_time} and used {max_gpu_memory:.2f} GB GPU memory"
            )
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            logger.info(f"Paladin inference took {end_time - start_time}")
    else:
        logger.info(f"Paladin inference took {end_time - start_time}")
    
    return paladin_results


@spaces.GPU(duration=60)
def _run_inference_pipeline_free(
    coords,
    slide_path,
    attrs,
    site_type,
    sex,
    tissue_site_idx,
    cancer_subtype,
    cancer_subtype_name_map,
    num_workers,
    progress,
):
    """Run inference pipeline with 60s GPU limit (for free users)."""
    return _run_inference_pipeline_impl(
        coords, slide_path, attrs, site_type, sex, tissue_site_idx,
        cancer_subtype, cancer_subtype_name_map, num_workers, progress
    )


@spaces.GPU(duration=300)
def _run_inference_pipeline_pro(
    coords,
    slide_path,
    attrs,
    site_type,
    sex,
    tissue_site_idx,
    cancer_subtype,
    cancer_subtype_name_map,
    num_workers,
    progress,
):
    """Run inference pipeline with 300s GPU limit (for PRO users)."""
    return _run_inference_pipeline_impl(
        coords, slide_path, attrs, site_type, sex, tissue_site_idx,
        cancer_subtype, cancer_subtype_name_map, num_workers, progress
    )


def _run_inference_pipeline_impl(
    coords,
    slide_path,
    attrs,
    site_type,
    sex,
    tissue_site_idx,
    cancer_subtype,
    cancer_subtype_name_map,
    num_workers,
    progress,
):
    """Run complete inference pipeline with separate GPU calls.

    This function orchestrates the GPU operations by calling separate functions
    for each GPU-intensive task, allowing HF Spaces to allocate GPU resources
    independently for each operation.

    Args:
        coords: Tissue tile coordinates
        slide_path: Path to the whole slide image file
        attrs: Slide attributes
        site_type: Site type, either "Primary" or "Metastatic"
        cancer_subtype: Cancer subtype (OncoTree code or "Unknown" for inference)
        cancer_subtype_name_map: Dictionary mapping cancer subtype names to codes
        num_workers: Number of worker processes for feature extraction
        progress: Gradio progress tracker for UI updates

    Returns:
        tuple: (aeon_results, paladin_results)
            - aeon_results: DataFrame with cancer subtype predictions and confidence scores
            - paladin_results: DataFrame with biomarker predictions
    """
    # Step 2: Extract CTransPath features
    progress(0.3, desc="Extracting CTransPath features")
    ctranspath_features, coords = _extract_ctranspath_features(
        coords, slide_path, attrs, num_workers
    )

    # Step 3: Filter features using marker classifier (CPU operation)
    start_time = pd.Timestamp.now()
    data_dir = get_data_directory()
    marker_classifier = pickle.load(open(data_dir / "marker_classifier.pkl", "rb"))
    progress(0.35, desc="Filtering features with marker classifier")
    logger.info("Filtering features with marker classifier")
    _, filtered_coords = filter_features(
        ctranspath_features,
        coords,
        marker_classifier,
        threshold=0.25,
    )
    end_time = pd.Timestamp.now()
    logger.info(f"Feature filtering took {end_time - start_time}")
    logger.info(
        f"Filtered from {len(coords)} to {len(filtered_coords)} tiles using marker classifier"
    )

    # Step 4: Extract Optimus features
    progress(0.4, desc="Extracting Optimus features")
    features = _extract_optimus_features(filtered_coords, slide_path, attrs, num_workers)

    # Step 5: Run Aeon to predict histology if not supplied
    if cancer_subtype == "Unknown":
        progress(0.9, desc="Running Aeon for cancer subtype inference")
        aeon_results = _run_aeon_inference(features, site_type, num_workers, sex, tissue_site_idx)
    else:
        cancer_subtype_code = cancer_subtype_name_map.get(cancer_subtype)
        aeon_results = pd.DataFrame(
            {
                "Cancer Subtype": [cancer_subtype_code],
                "Confidence": [1.0],
            }
        )
        logger.info(f"Using user-supplied cancer subtype: {cancer_subtype}")

    # Step 6: Run Paladin to predict biomarkers
    if len(aeon_results) == 0:
        logger.warning("No Aeon results, skipping Paladin inference")
        return None, None
    
    progress(0.95, desc="Running Paladin for biomarker inference")
    paladin_results = _run_paladin_inference(features, aeon_results, site_type, num_workers)

    aeon_results.set_index("Cancer Subtype", inplace=True)

    return aeon_results, paladin_results


# ============================================================================
# Batch-Optimized Pipeline Functions (use pre-loaded models)
# ============================================================================


def _run_aeon_inference_with_model(
    features, model, device, site_type, num_workers, sex_idx=None, tissue_site_idx=None
):
    """Run Aeon inference using pre-loaded model (for batch processing).

    Args:
        features: CTransPath features
        model: Pre-loaded Aeon model
        device: torch.device for GPU/CPU placement
        site_type: "Primary" or "Metastatic"
        num_workers: Number of workers for data loading
        sex_idx: Encoded sex index (0=Male, 1=Female), optional
        tissue_site_idx: Encoded tissue site index (0-56), optional

    Returns:
        DataFrame with cancer subtype predictions and confidence scores
    """
    from mosaic.inference import aeon

    metastatic = site_type == "Metastatic"

    # Use appropriate batch size based on GPU type
    if IS_T4_GPU:
        batch_size = 4
        logger.info(f"Running Aeon on T4 with num_workers={num_workers}")
    else:
        batch_size = 8
        logger.info(f"Running Aeon with num_workers={num_workers}")

    start_time = pd.Timestamp.now()
    aeon_results, _ = aeon.run_with_model(
        features=features,
        model=model,
        device=device,
        metastatic=metastatic,
        batch_size=batch_size,
        num_workers=num_workers,
        sex=sex_idx,
        tissue_site_idx=tissue_site_idx,
    )
    end_time = pd.Timestamp.now()

    if torch.cuda.is_available():
        max_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)
        logger.info(
            f"Aeon inference took {end_time - start_time} and used {max_gpu_memory:.2f} GB GPU memory"
        )

    return aeon_results


def _run_paladin_inference_with_models(
    features, aeon_results, site_type, model_cache, num_workers
):
    """Run Paladin inference using pre-loaded models from cache (for batch processing).

    Args:
        features: Optimus features
        aeon_results: DataFrame with Aeon predictions
        site_type: "Primary" or "Metastatic"
        model_cache: ModelCache instance with pre-loaded models
        num_workers: Number of workers for data loading

    Returns:
        DataFrame with biomarker predictions (Cancer Subtype, Biomarker, Score)
    """
    from mosaic.inference import paladin

    metastatic = site_type == "Metastatic"
    data_dir = get_data_directory()
    model_map_path = str(data_dir / "paladin_model_map.csv")

    # Use appropriate batch size based on GPU type
    if IS_T4_GPU:
        batch_size = 4
        logger.info(f"Running Paladin on T4 with num_workers={num_workers}")
    else:
        batch_size = 8
        logger.info(f"Running Paladin with num_workers={num_workers}")

    start_time = pd.Timestamp.now()
    paladin_results = paladin.run_with_models(
        features=features,
        aeon_results=aeon_results,
        model_cache=model_cache,
        model_map_path=model_map_path,
        metastatic=metastatic,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    end_time = pd.Timestamp.now()

    if torch.cuda.is_available():
        max_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)
        logger.info(
            f"Paladin inference took {end_time - start_time} and used {max_gpu_memory:.2f} GB GPU memory"
        )

    return paladin_results


def _run_inference_pipeline_with_models(
    coords,
    slide_path,
    attrs,
    site_type,
    sex_idx,
    tissue_site_idx,
    cancer_subtype,
    cancer_subtype_name_map,
    model_cache,
    num_workers,
    progress,
):
    """Run complete inference pipeline using pre-loaded models (for batch processing).

    This function is optimized for batch processing where models are loaded once
    and reused across multiple slides instead of being reloaded each time.

    Args:
        coords: Tile coordinates from tissue segmentation
        slide_path: Path to the slide file
        attrs: Attributes dictionary from tissue segmentation
        site_type: "Primary" or "Metastatic"
        sex_idx: Encoded sex index
        tissue_site_idx: Encoded tissue site index
        cancer_subtype: Known cancer subtype (or "Unknown")
        cancer_subtype_name_map: Dict mapping display names to OncoTree codes
        model_cache: ModelCache instance with pre-loaded models
        num_workers: Number of workers for data loading
        progress: Gradio progress tracker

    Returns:
        Tuple of (aeon_results, paladin_results)
    """
    # Step 1: Extract CTransPath features (still uses mussel's get_features)
    # Note: Feature extraction optimization can be added later if needed
    progress(0.3, desc="Extracting CTransPath features")
    ctranspath_features, coords = _extract_ctranspath_features(
        coords, slide_path, attrs, num_workers
    )

    # Step 2: Filter features using pre-loaded marker classifier
    start_time = pd.Timestamp.now()
    progress(0.35, desc="Filtering features with marker classifier")
    logger.info("Filtering features with PRE-LOADED marker classifier")
    _, filtered_coords = filter_features(
        ctranspath_features,
        coords,
        model_cache.marker_classifier,  # Use pre-loaded classifier
        threshold=0.25,
    )
    end_time = pd.Timestamp.now()
    logger.info(f"Feature filtering took {end_time - start_time}")
    logger.info(
        f"Filtered from {len(coords)} to {len(filtered_coords)} tiles using marker classifier"
    )

    # Step 3: Extract Optimus features (still uses mussel's get_features)
    progress(0.5, desc="Extracting Optimus features")
    features = _extract_optimus_features(filtered_coords, slide_path, attrs, num_workers)

    # Step 4: Run Aeon inference with pre-loaded model (if cancer subtype unknown)
    aeon_results = None
    progress(0.7, desc="Running Aeon for cancer subtype inference")

    # Check if cancer subtype is unknown
    if cancer_subtype in ["Unknown", None]:
        logger.info("Running Aeon inference with PRE-LOADED model (cancer subtype unknown)")
        aeon_results = _run_aeon_inference_with_model(
            features,
            model_cache.aeon_model,  # Use pre-loaded Aeon model
            model_cache.device,
            site_type,
            num_workers,
            sex_idx,
            tissue_site_idx,
        )
    else:
        # Cancer subtype is known, create synthetic Aeon results
        logger.info(f"Using known cancer subtype: {cancer_subtype}")
        oncotree_code = cancer_subtype_name_map.get(cancer_subtype, cancer_subtype)
        aeon_results = pd.DataFrame(
            [(oncotree_code, 1.0)], columns=["Cancer Subtype", "Confidence"]
        )

    # Step 5: Run Paladin inference with pre-loaded models
    progress(0.95, desc="Running Paladin for biomarker inference")
    paladin_results = _run_paladin_inference_with_models(
        features, aeon_results, site_type, model_cache, num_workers
    )

    aeon_results.set_index("Cancer Subtype", inplace=True)

    return aeon_results, paladin_results


def analyze_slide_with_models(
    slide_path,
    seg_config,
    site_type,
    sex,
    tissue_site,
    cancer_subtype,
    cancer_subtype_name_map,
    model_cache,
    ihc_subtype="",
    num_workers=4,
    progress=None,
):
    """Analyze a slide using pre-loaded models (batch-optimized version).

    This function is optimized for batch processing where models are loaded once
    in a ModelCache and reused across multiple slides.

    Args:
        slide_path: Path to the slide file
        seg_config: Segmentation configuration ("Biopsy", "Resection", or "TCGA")
        site_type: "Primary" or "Metastatic"
        sex: Patient sex ("Unknown", "Male", "Female")
        tissue_site: Tissue site name
        cancer_subtype: Known cancer subtype or "Unknown"
        cancer_subtype_name_map: Dict mapping display names to OncoTree codes
        model_cache: ModelCache instance with pre-loaded models
        ihc_subtype: IHC subtype for breast cancer (optional)
        num_workers: Number of workers for data loading
        progress: Gradio progress tracker

    Returns:
        Tuple of (slide_mask, aeon_results, paladin_results)
    """
    from mosaic.inference.data import encode_sex, encode_tissue_site

    if progress is None:
        progress = lambda frac, desc: None  # No-op progress function

    # Encode sex and tissue site
    sex_idx = encode_sex(sex) if sex else None
    tissue_site_idx = encode_tissue_site(tissue_site) if tissue_site else None

    # Step 1: Convert seg_config string to config object
    if isinstance(seg_config, str):
        if seg_config == "Biopsy":
            seg_config = BiopsySegConfig()
        elif seg_config == "Resection":
            seg_config = ResectionSegConfig()
        elif seg_config == "TCGA":
            seg_config = TcgaSegConfig()
        else:
            raise ValueError(f"Unknown segmentation configuration: {seg_config}")

    # Step 2: Tissue segmentation (CPU operation, not affected by model caching)
    progress(0.0, desc="Segmenting tissue")
    logger.info(f"Segmenting tissue for slide: {slide_path}")
    start_time = pd.Timestamp.now()

    if values := segment_tissue(
        slide_path=slide_path,
        patch_size=224,
        mpp=0.5,
        seg_level=-1,
        segment_threshold=seg_config.segment_threshold,
        median_blur_ksize=seg_config.median_blur_ksize,
        morphology_ex_kernel=seg_config.morphology_ex_kernel,
        tissue_area_threshold=seg_config.tissue_area_threshold,
        hole_area_threshold=seg_config.hole_area_threshold,
        max_num_holes=seg_config.max_num_holes,
    ):
        polygon, _, coords, attrs = values
    else:
        logger.warning("No tissue detected in slide")
        return None, None, None

    end_time = pd.Timestamp.now()
    logger.info(f"Tissue segmentation took {end_time - start_time}")
    logger.info(f"Found {len(coords)} tissue tiles")

    if len(coords) == 0:
        logger.warning("No tissue tiles found in slide")
        return None, None, None

    # Step 2: Create slide mask visualization (CPU operation)
    progress(0.2, desc="Creating slide mask")
    logger.info("Drawing slide mask")
    slide_mask = draw_slide_mask(
        slide_path, polygon, outline="black", fill=(255, 0, 0, 80), vis_level=-1
    )
    logger.info("Slide mask drawn")

    # Step 3: Run inference pipeline with pre-loaded models
    aeon_results, paladin_results = _run_inference_pipeline_with_models(
        coords,
        slide_path,
        attrs,
        site_type,
        sex_idx,
        tissue_site_idx,
        cancer_subtype,
        cancer_subtype_name_map,
        model_cache,
        num_workers,
        progress,
    )

    progress(1.0, desc="Analysis complete")

    return slide_mask, aeon_results, paladin_results


def analyze_slide(
    slide_path,
    seg_config,
    site_type,
    sex,
    tissue_site,
    cancer_subtype,
    cancer_subtype_name_map,
    ihc_subtype="",
    num_workers=4,
    progress=gr.Progress(track_tqdm=True),
    request: gr.Request = None,
):
    """Analyze a whole slide image for cancer subtype and biomarker prediction.

    This function performs a complete analysis pipeline including:
    1. Tissue segmentation (CPU-only, no GPU required)
    2. GPU-intensive feature extraction and model inference

    The GPU-intensive operations are handled by a separate function decorated
    with @spaces.GPU to efficiently manage GPU resources on Hugging Face Spaces.
    Tissue segmentation runs on CPU and is not included in the GPU allocation.

    Args:
        slide_path: Path to the whole slide image file
        seg_config: Segmentation configuration, one of "Biopsy", "Resection", or "TCGA"
        site_type: Site type, either "Primary" or "Metastatic"
        cancer_subtype: Cancer subtype (OncoTree code or "Unknown" for inference)
        cancer_subtype_name_map: Dictionary mapping cancer subtype names to codes
        ihc_subtype: IHC subtype for breast cancer (optional)
        num_workers: Number of worker processes for feature extraction
        progress: Gradio progress tracker for UI updates

    Returns:
        tuple: (slide_mask, aeon_results, paladin_results)
            - slide_mask: PIL Image of tissue segmentation visualization
            - aeon_results: DataFrame with cancer subtype predictions and confidence scores
            - paladin_results: DataFrame with biomarker predictions

    Raises:
        gr.Error: If no slide is provided
        gr.Warning: If no tissue is detected in the slide
        ValueError: If an unknown segmentation configuration is provided
    """
    if slide_path is None:
        raise gr.Error("Please upload a slide.")

    # Step 1: Segment tissue (CPU-only, not GPU-intensive)
    start_time = pd.Timestamp.now()

    if seg_config == "Biopsy":
        seg_config = BiopsySegConfig()
    elif seg_config == "Resection":
        seg_config = ResectionSegConfig()
    elif seg_config == "TCGA":
        seg_config = TcgaSegConfig()
    else:
        raise ValueError(f"Unknown segmentation configuration: {seg_config}")

    progress(0.0, desc="Segmenting tissue")
    logger.info(f"Segmenting tissue for slide: {slide_path}")
    if values := segment_tissue(
        slide_path=slide_path,
        patch_size=224,
        mpp=0.5,
        seg_level=-1,
        segment_threshold=seg_config.segment_threshold,
        median_blur_ksize=seg_config.median_blur_ksize,
        morphology_ex_kernel=seg_config.morphology_ex_kernel,
        tissue_area_threshold=seg_config.tissue_area_threshold,
        hole_area_threshold=seg_config.hole_area_threshold,
        max_num_holes=seg_config.max_num_holes,
    ):
        polygon, _, coords, attrs = values
    else:
        gr.Warning(f"No tissue detected in slide: {slide_path}")
        return None, None, None
    end_time = pd.Timestamp.now()
    logger.info(f"Tissue segmentation took {end_time - start_time}")
    logger.info(f"Found {len(coords)} tissue tiles")
    progress(0.2, desc="Tissue segmented")

    # Draw slide mask for visualization
    logger.info("Drawing slide mask")
    progress(0.25, desc="Drawing slide mask")
    slide_mask = draw_slide_mask(
        slide_path, polygon, outline="black", fill=(255, 0, 0, 80), vis_level=-1
    )
    logger.info("Slide mask drawn")

    # Step 2-6: Run inference pipeline with GPU
    # Check if user is logged in for longer GPU duration
    is_logged_in = False
    username = "anonymous"
    if request is not None:
        try:
            # Check if user is logged in via JWT token in referer
            # HF Spaces doesn't populate request.username but includes JWT in URL
            if hasattr(request, 'headers'):
                referer = request.headers.get('referer', '')
                if '__sign=' in referer:
                    # Extract and decode JWT token
                    import re
                    import json
                    import base64
                    
                    match = re.search(r'__sign=([^&]+)', referer)
                    if match:
                        token = match.group(1)
                        try:
                            # JWT format: header.payload.signature
                            # We only need the payload (middle part)
                            parts = token.split('.')
                            if len(parts) == 3:
                                # Decode base64 payload (add padding if needed)
                                payload = parts[1]
                                payload += '=' * (4 - len(payload) % 4)
                                decoded = base64.urlsafe_b64decode(payload)
                                token_data = json.loads(decoded)
                                
                                # Check if user is in token
                                if 'onBehalfOf' in token_data and 'user' in token_data['onBehalfOf']:
                                    username = token_data['onBehalfOf']['user']
                                    is_logged_in = True
                                    logger.info(f"Found user in JWT token: {username}")
                        except Exception as e:
                            logger.warning(f"Failed to decode JWT: {e}")
            
            if IS_ZEROGPU:
                logger.info(f"User: {username} | Logged in: {is_logged_in}")
        except Exception as e:
            logger.warning(f"Failed to detect user: {e}")
            import traceback
            logger.warning(traceback.format_exc())

    # Convert sex and tissue_site to indices for Aeon model
    from mosaic.inference.data import encode_sex, encode_tissue_site

    sex_idx = None
    if sex is not None:
        sex_idx = encode_sex(sex)

    tissue_site_idx = None
    if tissue_site is not None:
        tissue_site_idx = encode_tissue_site(tissue_site)

    if is_logged_in:
        if IS_ZEROGPU:
            logger.info("Using 300s GPU allocation (logged-in user)")
        aeon_results, paladin_results = _run_inference_pipeline_pro(
            coords,
            slide_path,
            attrs,
            site_type,
            sex_idx,
            tissue_site_idx,
            cancer_subtype,
            cancer_subtype_name_map,
            num_workers,
            progress,
        )
    else:
        if IS_ZEROGPU:
            logger.info("Using 60s GPU allocation (anonymous user)")
        aeon_results, paladin_results = _run_inference_pipeline_free(
            coords,
            slide_path,
            attrs,
            site_type,
            sex_idx,
            tissue_site_idx,
            cancer_subtype,
            cancer_subtype_name_map,
            num_workers,
            progress,
        )

    return slide_mask, aeon_results, paladin_results
