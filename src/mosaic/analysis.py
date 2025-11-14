"""Core slide analysis module for Mosaic.

This module provides the main slide analysis pipeline that integrates tissue segmentation,
feature extraction, and model inference for cancer subtype and biomarker prediction.
"""

# Import spaces first before any CUDA-related imports
try:
    import spaces

    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

    # Create a no-op decorator if spaces is not available
    class spaces:
        @staticmethod
        def GPU(fn):
            return fn

import pickle
import torch
import pandas as pd
import gradio as gr
from pathlib import Path
from mussel.models import ModelType
from mussel.utils import get_features, segment_tissue, filter_features
from mussel.utils.segment import draw_slide_mask
from mussel.cli.tessellate import BiopsySegConfig, ResectionSegConfig, TcgaSegConfig
from loguru import logger


from mosaic.inference import run_aeon, run_paladin


@spaces.GPU(duration=120)
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
    if HAS_SPACES:
        num_workers = 0
        logger.info("Running CTransPath on Zero GPU: setting num_workers=0")
    
    start_time = pd.Timestamp.now()
    logger.info("Extracting CTransPath features")
    ctranspath_features, _ = get_features(
        coords,
        slide_path,
        attrs,
        model_type=ModelType.CTRANSPATH,
        model_path="data/ctranspath.pth",
        num_workers=num_workers,
        batch_size=64,
        use_gpu=True,
    )
    end_time = pd.Timestamp.now()
    max_gpu_memory = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0
    )
    logger.info(
        f"CTransPath extraction took {end_time - start_time} and used {max_gpu_memory:.2f} GB GPU memory"
    )
    torch.cuda.reset_peak_memory_stats()
    
    return ctranspath_features, coords


@spaces.GPU(duration=600)
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
    if HAS_SPACES:
        num_workers = 0
        logger.info("Running Optimus on Zero GPU: setting num_workers=0")
    
    start_time = pd.Timestamp.now()
    logger.info("Extracting Optimus features")
    features, _ = get_features(
        filtered_coords,
        slide_path,
        attrs,
        model_type=ModelType.OPTIMUS,
        model_path="data/optimus.pkl",
        num_workers=num_workers,
        batch_size=64,
        use_gpu=True,
    )
    end_time = pd.Timestamp.now()
    max_gpu_memory = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0
    )
    logger.info(
        f"Optimus extraction took {end_time - start_time} and used {max_gpu_memory:.2f} GB GPU memory"
    )
    torch.cuda.reset_peak_memory_stats()
    
    return features


@spaces.GPU(duration=60)
def _run_aeon_inference(features, site_type, num_workers):
    """Run Aeon cancer subtype inference on GPU.
    
    Args:
        features: Optimus features
        site_type: Site type ("Primary" or "Metastatic")
        num_workers: Number of worker processes
        
    Returns:
        Aeon results DataFrame
    """
    if HAS_SPACES:
        num_workers = 0
        logger.info("Running Aeon on Zero GPU: setting num_workers=0")
    
    start_time = pd.Timestamp.now()
    logger.info("Running Aeon for cancer subtype inference")
    aeon_results, _ = run_aeon(
        features=features,
        model_path="data/aeon_model.pkl",
        metastatic=(site_type == "Metastatic"),
        batch_size=8,
        num_workers=num_workers,
        use_cpu=False,
    )
    end_time = pd.Timestamp.now()
    max_gpu_memory = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0
    )
    logger.info(
        f"Aeon inference took {end_time - start_time} and used {max_gpu_memory:.2f} GB GPU memory"
    )
    torch.cuda.reset_peak_memory_stats()
    
    return aeon_results


@spaces.GPU(duration=60)
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
    if HAS_SPACES:
        num_workers = 0
        logger.info("Running Paladin on Zero GPU: setting num_workers=0")
    
    start_time = pd.Timestamp.now()
    logger.info("Running Paladin for biomarker inference")
    paladin_results = run_paladin(
        features=features,
        model_map_path="data/paladin_model_map.csv",
        aeon_results=aeon_results,
        metastatic=(site_type == "Metastatic"),
        batch_size=8,
        num_workers=num_workers,
        use_cpu=False,
    )
    end_time = pd.Timestamp.now()
    max_gpu_memory = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0
    )
    logger.info(
        f"Paladin inference took {end_time - start_time} and used {max_gpu_memory:.2f} GB GPU memory"
    )
    torch.cuda.reset_peak_memory_stats()
    
    return paladin_results


def _run_inference_pipeline(
    coords,
    slide_path,
    attrs,
    site_type,
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
    marker_classifier = pickle.load(open("data/marker_classifier.pkl", "rb"))
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
        aeon_results = _run_aeon_inference(features, site_type, num_workers)
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


def analyze_slide(
    slide_path,
    seg_config,
    site_type,
    cancer_subtype,
    cancer_subtype_name_map,
    ihc_subtype="",
    num_workers=4,
    progress=gr.Progress(track_tqdm=True),
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

    # Step 2-6: Run inference pipeline with separate GPU calls
    aeon_results, paladin_results = _run_inference_pipeline(
        coords,
        slide_path,
        attrs,
        site_type,
        cancer_subtype,
        cancer_subtype_name_map,
        num_workers,
        progress,
    )

    return slide_mask, aeon_results, paladin_results
