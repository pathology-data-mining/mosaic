"""Core slide analysis module for Mosaic.

This module provides the main slide analysis pipeline that integrates tissue segmentation,
feature extraction, and model inference for cancer subtype and biomarker prediction.
"""

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
    1. Tissue segmentation
    2. CTransPath feature extraction
    3. Feature filtering with marker classifier
    4. Optimus feature extraction on filtered tiles
    5. Aeon inference for cancer subtype (if not provided)
    6. Paladin inference for biomarker prediction
    
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
    # Step 1: Segment tissue
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

    # Step 2: Extract features with CTransPath
    start_time = pd.Timestamp.now()
    progress(0.3, desc="Extracting CTransPath features")
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
        f"CTransPath Feature extraction took {end_time - start_time} and used {max_gpu_memory:.2f} GB GPU memory"
    )

    torch.cuda.reset_peak_memory_stats()

    # Step 3: Filter features using marker classifier
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

    # Step 4: Extract features with Optimus on filtered coords
    start_time = pd.Timestamp.now()
    progress(0.4, desc="Extracting Optimus features")
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
        f"Optimus Feature extraction took {end_time - start_time} and used {max_gpu_memory:.2f} GB GPU memory"
    )

    torch.cuda.reset_peak_memory_stats()

    # Step 3: Run Aeon to predict histology if not supplied
    if cancer_subtype == "Unknown":
        start_time = pd.Timestamp.now()
        progress(0.9, desc="Running Aeon for cancer subtype inference")
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
    else:
        cancer_subtype_code = cancer_subtype_name_map.get(cancer_subtype)
        aeon_results = pd.DataFrame(
            {
                "Cancer Subtype": [cancer_subtype_code],
                "Confidence": [1.0],
            }
        )
        logger.info(f"Using user-supplied cancer subtype: {cancer_subtype}")

    # Step 4: Run Paladin to predict biomarkers
    if len(aeon_results) == 0:
        logger.warning("No Aeon results, skipping Paladin inference")
        return slide_mask, None, None
    start_time = pd.Timestamp.now()
    progress(0.95, desc="Running Paladin for biomarker inference")
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

    aeon_results.set_index("Cancer Subtype", inplace=True)

    return slide_mask, aeon_results, paladin_results
