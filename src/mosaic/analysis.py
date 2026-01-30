"""Core slide analysis module for Mosaic.

This module provides the main slide analysis pipeline that integrates tissue segmentation,
feature extraction, and model inference for cancer subtype and biomarker prediction.
"""

import pickle
import gc
import time as time_module
from contextlib import contextmanager

import torch
import pandas as pd
import gradio as gr
from gradio.helpers import TrackedIterable
from pathlib import Path
from mussel.models import ModelType
from mussel.utils import get_features, segment_tissue, filter_features
from mussel.utils.segment import draw_slide_mask
from mussel.cli.tessellate import BiopsySegConfig, ResectionSegConfig, TcgaSegConfig
from loguru import logger
from mosaic.inference import run_aeon, run_paladin
from mosaic.data_directory import get_data_directory

# Import centralized hardware detection
from mosaic.hardware import (
    spaces,
    IS_ZEROGPU,
    IS_T4_GPU,
    GPU_TYPE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
)


@contextmanager
def _stage_label(progress, desc, stage_progress=None):
    """Add a persistent stage label above the tqdm sub-progress bar.

    Appends a TrackedIterable to progress.iterables so it persists
    alongside tqdm updates, then removes it on exit.

    Args:
        progress: Gradio progress tracker (may be a SlideProgressWrapper).
        desc: Description text for the stage label.
        stage_progress: Raw pipeline-stage fraction (e.g. 0.3). Used to
            compute an overall job percentage that is shown in the label
            and set on the TrackedIterable's progress field.
    """
    tracked = None
    iterables = getattr(progress, "iterables", None)
    if iterables is not None:
        # Compute overall job progress from the wrapper when available
        overall = None
        if stage_progress is not None:
            try:
                slide_idx = getattr(progress, "slide_idx", None)
                total_slides = getattr(progress, "total_slides", None)
                if (
                    isinstance(slide_idx, (int, float))
                    and isinstance(total_slides, (int, float))
                    and total_slides > 0
                ):
                    slide_start = slide_idx / total_slides
                    slide_range = 1.0 / total_slides
                    overall = slide_start + stage_progress * slide_range
                else:
                    overall = stage_progress
            except (TypeError, AttributeError):
                overall = stage_progress

        label = desc
        if overall is not None:
            pct = int(overall * 100)
            label = f"[{pct}%] {desc}"

        tracked = TrackedIterable(None, None, None, label, None, None, overall)
        iterables.append(tracked)
    try:
        yield
    finally:
        if tracked is not None and iterables is not None:
            try:
                iterables.remove(tracked)
            except ValueError:
                pass


def _extract_ctranspath_features(
    coords, slide_path, attrs, num_workers, model, timings=None
):
    """Extract CTransPath features on GPU using pre-loaded model.

    Args:
        coords: Tissue tile coordinates
        slide_path: Path to the whole slide image file
        attrs: Slide attributes
        num_workers: Number of worker processes
        model: Pre-loaded CTransPath model from ModelCache
        timings: Optional dict to populate with timing data

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
        logger.info(
            f"Running CTransPath on T4: processing {len(coords)} tiles with batch_size={batch_size}"
        )
    else:
        num_workers = max(num_workers, 8)
        batch_size = 64
        logger.info(f"Running CTransPath with {num_workers} workers")

    start_time = time_module.time()

    ctranspath_features, _ = get_features(
        coords,
        slide_path,
        attrs,
        model=model,
        num_workers=num_workers,
        batch_size=batch_size,
        use_gpu=True,
    )

    duration_sec = time_module.time() - start_time
    logger.info(f"CTransPath extraction took {duration_sec:.2f}s")

    if timings is not None:
        timings["ctranspath"] = duration_sec
        timings["batch_size"] = batch_size

    return ctranspath_features, coords


def _extract_optimus_features(
    filtered_coords, slide_path, attrs, num_workers, model, timings=None
):
    """Extract Optimus features on GPU using pre-loaded model.

    Args:
        filtered_coords: Filtered tissue tile coordinates
        slide_path: Path to the whole slide image file
        attrs: Slide attributes
        num_workers: Number of worker processes
        model: Pre-loaded Optimus model from ModelCache
        timings: Optional dict to populate with timing data

    Returns:
        Optimus features
    """
    if IS_ZEROGPU:
        num_workers = 0
        batch_size = 128
        logger.info(
            f"Running Optimus on ZeroGPU: processing {len(filtered_coords)} tiles"
        )
    elif IS_T4_GPU:
        num_workers = DEFAULT_NUM_WORKERS
        batch_size = DEFAULT_BATCH_SIZE
        logger.info(
            f"Running Optimus on T4: processing {len(filtered_coords)} tiles with batch_size={batch_size}"
        )
    else:
        num_workers = max(num_workers, 8)
        batch_size = 64
        logger.info(f"Running Optimus with {num_workers} workers")

    start_time = time_module.time()

    features, _ = get_features(
        filtered_coords,
        slide_path,
        attrs,
        model=model,
        num_workers=num_workers,
        batch_size=batch_size,
        use_gpu=True,
    )

    duration_sec = time_module.time() - start_time
    logger.info(f"Optimus extraction took {duration_sec:.2f}s")

    if timings is not None:
        timings["optimus"] = duration_sec

    return features


def _run_aeon_inference(
    features, site_type, num_workers, sex=None, tissue_site_idx=None, timings=None
):
    """Run Aeon cancer subtype inference on GPU.

    Args:
        features: Optimus features
        site_type: Site type ("Primary" or "Metastatic")
        num_workers: Number of worker processes
        sex: Patient sex (0=Male, 1=Female), optional
        tissue_site_idx: Tissue site index (0-56), optional
        timings: Optional dict to populate with timing data

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

    start_time = time_module.time()
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
    duration_sec = time_module.time() - start_time

    # Log memory stats if CUDA is available
    if torch.cuda.is_available():
        try:
            max_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)
            logger.info(
                f"Aeon inference took {duration_sec:.2f}s and used {max_gpu_memory:.2f} GB GPU memory"
            )
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            logger.info(f"Aeon inference took {duration_sec:.2f}s")
    else:
        logger.info(f"Aeon inference took {duration_sec:.2f}s")

    if timings is not None:
        timings["aeon"] = duration_sec

    return aeon_results


def _run_paladin_inference(
    features, aeon_results, site_type, num_workers, timings=None
):
    """Run Paladin biomarker inference on GPU.

    Args:
        features: Optimus features
        aeon_results: Aeon results DataFrame
        site_type: Site type ("Primary" or "Metastatic")
        num_workers: Number of worker processes
        timings: Optional dict to populate with timing data

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

    start_time = time_module.time()
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
    duration_sec = time_module.time() - start_time

    # Log memory stats if CUDA is available
    if torch.cuda.is_available():
        try:
            max_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)
            logger.info(
                f"Paladin inference took {duration_sec:.2f}s and used {max_gpu_memory:.2f} GB GPU memory"
            )
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            logger.info(f"Paladin inference took {duration_sec:.2f}s")
    else:
        logger.info(f"Paladin inference took {duration_sec:.2f}s")

    if timings is not None:
        timings["paladin"] = duration_sec

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
    timings=None,
):
    """Run inference pipeline with 60s GPU limit (for free users)."""
    return _run_inference_pipeline_impl(
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
        timings=timings,
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
    timings=None,
):
    """Run inference pipeline with 300s GPU limit (for PRO users)."""
    return _run_inference_pipeline_impl(
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
        timings=timings,
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
    timings=None,
):
    """Run complete inference pipeline using model cache.

    This function loads models once and reuses them throughout the pipeline,
    orchestrating GPU operations for feature extraction and inference.

    Args:
        coords: Tissue tile coordinates
        slide_path: Path to the whole slide image file
        attrs: Slide attributes
        site_type: Site type, either "Primary" or "Metastatic"
        cancer_subtype: Cancer subtype (OncoTree code or "Unknown" for inference)
        cancer_subtype_name_map: Dictionary mapping cancer subtype names to codes
        num_workers: Number of worker processes for feature extraction
        progress: Gradio progress tracker for UI updates
        timings: Optional dict to populate with timing data

    Returns:
        tuple: (aeon_results, paladin_results)
            - aeon_results: DataFrame with cancer subtype predictions and confidence scores
            - paladin_results: DataFrame with biomarker predictions
    """
    # Load all models once for the entire pipeline
    from mosaic.model_manager import load_all_models

    progress(0.1, desc="Loading models")
    logger.info("Loading models for inference pipeline")
    model_cache = load_all_models(use_gpu=True)

    try:
        # Step 2: Extract CTransPath features using cached model
        progress(0.3, desc=f"Extracting CTransPath features ({len(coords)} tiles)")
        with _stage_label(progress, f"Extracting CTransPath features ({len(coords)} tiles)", stage_progress=0.3):
            ctranspath_features, coords = _extract_ctranspath_features(
                coords,
                slide_path,
                attrs,
                num_workers,
                model=model_cache.ctranspath_model,
                timings=timings,
            )

        # Step 3: Filter features using cached marker classifier
        progress(0.35, desc="Filtering features with marker classifier")
        logger.info("Filtering features with marker classifier")
        _, filtered_coords = filter_features(
            ctranspath_features,
            coords,
            model_cache.marker_classifier,
            threshold=0.25,
        )
        logger.info(
            f"Filtered from {len(coords)} to {len(filtered_coords)} tiles using marker classifier"
        )
        if timings is not None:
            timings["filtered_tile_count"] = len(filtered_coords)

        # Step 4: Extract Optimus features using cached model
        progress(0.4, desc=f"Extracting Optimus features ({len(filtered_coords)} tiles)")
        with _stage_label(progress, f"Extracting Optimus features ({len(filtered_coords)} tiles)", stage_progress=0.4):
            features = _extract_optimus_features(
                filtered_coords,
                slide_path,
                attrs,
                num_workers,
                model=model_cache.optimus_model,
                timings=timings,
            )

        # Step 5: Run Aeon to predict histology if not supplied
        if cancer_subtype == "Unknown":
            progress(0.9, desc="Running Aeon for cancer subtype inference")
            aeon_results = _run_aeon_inference_with_model(
                features,
                model_cache.aeon_model,
                model_cache.device,
                site_type,
                num_workers,
                sex,
                tissue_site_idx,
                timings=timings,
            )
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
        paladin_results = _run_paladin_inference_with_models(
            features, aeon_results, site_type, model_cache, num_workers, timings=timings
        )

        aeon_results.set_index("Cancer Subtype", inplace=True)

        return aeon_results, paladin_results
    finally:
        # Clean up models to free GPU memory
        logger.info("Cleaning up models after single-slide inference")
        model_cache.cleanup()

        # T4-specific: Ensure GPU operations are complete before next request
        if IS_T4_GPU and torch.cuda.is_available():
            torch.cuda.synchronize()
            logger.info("T4: GPU operations synchronized")


# ============================================================================
# Batch-Optimized Pipeline Functions (use pre-loaded models)
# ============================================================================


def _run_aeon_inference_with_model(
    features,
    model,
    device,
    site_type,
    num_workers,
    sex_idx=None,
    tissue_site_idx=None,
    timings=None,
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
        timings: Optional dict to populate with timing data

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

    start_time = time_module.time()
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
    duration_sec = time_module.time() - start_time

    if torch.cuda.is_available():
        max_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)
        logger.info(
            f"Aeon inference took {duration_sec:.2f}s and used {max_gpu_memory:.2f} GB GPU memory"
        )

    if timings is not None:
        timings["aeon"] = duration_sec

    return aeon_results


def _run_paladin_inference_with_models(
    features, aeon_results, site_type, model_cache, num_workers, timings=None
):
    """Run Paladin inference using pre-loaded models from cache (for batch processing).

    Args:
        features: Optimus features
        aeon_results: DataFrame with Aeon predictions
        site_type: "Primary" or "Metastatic"
        model_cache: ModelCache instance with pre-loaded models
        num_workers: Number of workers for data loading
        timings: Optional dict to populate with timing data

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

    start_time = time_module.time()
    paladin_results = paladin.run_with_models(
        features=features,
        aeon_results=aeon_results,
        model_cache=model_cache,
        model_map_path=model_map_path,
        metastatic=metastatic,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    duration_sec = time_module.time() - start_time

    if torch.cuda.is_available():
        max_gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)
        logger.info(
            f"Paladin inference took {duration_sec:.2f}s and used {max_gpu_memory:.2f} GB GPU memory"
        )

    if timings is not None:
        timings["paladin"] = duration_sec

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
    timings=None,
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
        timings: Optional dict to populate with timing data

    Returns:
        Tuple of (aeon_results, paladin_results)
    """
    # Step 1: Extract CTransPath features with PRE-LOADED model
    progress(0.3, desc=f"Extracting CTransPath features ({len(coords)} tiles)")
    with _stage_label(progress, f"Extracting CTransPath features ({len(coords)} tiles)", stage_progress=0.3):
        ctranspath_features, coords = _extract_ctranspath_features(
            coords,
            slide_path,
            attrs,
            num_workers,
            model=model_cache.ctranspath_model,
            timings=timings,
        )

    # Step 2: Filter features using pre-loaded marker classifier
    progress(0.35, desc="Filtering features with marker classifier")
    logger.info("Filtering features with PRE-LOADED marker classifier")
    _, filtered_coords = filter_features(
        ctranspath_features,
        coords,
        model_cache.marker_classifier,  # Use pre-loaded classifier
        threshold=0.25,
    )
    logger.info(
        f"Filtered from {len(coords)} to {len(filtered_coords)} tiles using marker classifier"
    )
    if timings is not None:
        timings["filtered_tile_count"] = len(filtered_coords)

    # Step 3: Extract Optimus features with PRE-LOADED model
    progress(0.5, desc=f"Extracting Optimus features ({len(filtered_coords)} tiles)")
    with _stage_label(progress, f"Extracting Optimus features ({len(filtered_coords)} tiles)", stage_progress=0.5):
        features = _extract_optimus_features(
            filtered_coords,
            slide_path,
            attrs,
            num_workers,
            model=model_cache.optimus_model,
            timings=timings,
        )

    # Step 4: Run Aeon inference with pre-loaded model (if cancer subtype unknown)
    aeon_results = None
    progress(0.7, desc="Running Aeon for cancer subtype inference")

    # Check if cancer subtype is unknown
    if cancer_subtype in ["Unknown", None]:
        logger.info(
            "Running Aeon inference with PRE-LOADED model (cancer subtype unknown)"
        )
        aeon_results = _run_aeon_inference_with_model(
            features,
            model_cache.aeon_model,  # Use pre-loaded Aeon model
            model_cache.device,
            site_type,
            num_workers,
            sex_idx,
            tissue_site_idx,
            timings=timings,
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
        features, aeon_results, site_type, model_cache, num_workers, timings=timings
    )

    aeon_results.set_index("Cancer Subtype", inplace=True)

    return aeon_results, paladin_results


# Removed: analyze_slide_with_models merged into analyze_slide below


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
    model_cache=None,
    analysis_id=None,
    session_hash=None,
):
    """Analyze a whole slide image for cancer subtype and biomarker prediction.

    This function works in two modes:
    1. **Single-slide mode** (model_cache=None): Loads models, analyzes one slide, cleans up
    2. **Batch mode** (model_cache provided): Uses pre-loaded models for efficiency

    Args:
        slide_path: Path to the whole slide image file
        seg_config: Segmentation configuration, one of "Biopsy", "Resection", or "TCGA"
        site_type: Site type, either "Primary" or "Metastatic"
        sex: Patient sex ("Male" or "Female") - required
        tissue_site: Tissue site name
        cancer_subtype: Cancer subtype (OncoTree code or "Unknown" for inference)
        cancer_subtype_name_map: Dictionary mapping cancer subtype names to codes
        ihc_subtype: IHC subtype for breast cancer (optional)
        num_workers: Number of worker processes for feature extraction
        progress: Gradio progress tracker for UI updates
        request: Gradio request object (for HF Spaces authentication)
        model_cache: Optional ModelCache with pre-loaded models (for batch processing)
        analysis_id: Optional telemetry analysis ID (for resource event tracking)
        session_hash: Optional session hash (for telemetry privacy)

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

    # Initialize telemetry for resource tracking
    slide_start_time = time_module.time()
    timings = {}  # Collect stage timings for telemetry

    # Step 1: Segment tissue (CPU-only, not GPU-intensive)
    seg_start_time = time_module.time()

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
    timings["segmentation"] = time_module.time() - seg_start_time
    logger.info(f"Tissue segmentation took {timings['segmentation']:.2f}s")
    logger.info(f"Found {len(coords)} tissue tiles")
    progress(0.2, desc="Tissue segmented")

    # Draw slide mask for visualization
    logger.info("Drawing slide mask")
    progress(0.25, desc="Drawing slide mask")
    slide_mask = draw_slide_mask(
        slide_path,
        polygon,
        outline="black",
        fill=(255, 0, 0, 80),
        vis_level=-1,
        max_size=2048,
    )
    logger.info("Slide mask drawn")

    # Convert sex and tissue_site to indices for Aeon model
    from mosaic.inference.data import encode_sex, encode_tissue_site

    sex_idx = None
    if sex is not None:
        sex_idx = encode_sex(sex)

    tissue_site_idx = None
    if tissue_site is not None:
        tissue_site_idx = encode_tissue_site(tissue_site)

    # Run inference pipeline - two modes based on model_cache
    if model_cache is not None:
        # Batch mode: use pre-loaded models
        logger.info("Using pre-loaded models from ModelCache (batch mode)")
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
            timings=timings,
        )
    else:
        # Single-slide mode: load models on-demand
        # Check if user is logged in for longer GPU duration (HF Spaces only)
        is_logged_in = False
        username = "anonymous"
        if request is not None:
            try:
                # Check if user is logged in via JWT token in referer
                # HF Spaces doesn't populate request.username but includes JWT in URL
                if hasattr(request, "headers"):
                    referer = request.headers.get("referer", "")
                    if "__sign=" in referer:
                        # Extract and decode JWT token
                        import re
                        import json
                        import base64

                        match = re.search(r"__sign=([^&]+)", referer)
                        if match:
                            token = match.group(1)
                            try:
                                # JWT format: header.payload.signature
                                # We only need the payload (middle part)
                                parts = token.split(".")
                                if len(parts) == 3:
                                    # Decode base64 payload (add padding if needed)
                                    payload = parts[1]
                                    payload += "=" * (4 - len(payload) % 4)
                                    decoded = base64.urlsafe_b64decode(payload)
                                    token_data = json.loads(decoded)

                                    # Check if user is in token
                                    if (
                                        "onBehalfOf" in token_data
                                        and "user" in token_data["onBehalfOf"]
                                    ):
                                        username = token_data["onBehalfOf"]["user"]
                                        is_logged_in = True
                                        logger.info(
                                            f"Found user in JWT token: {username}"
                                        )
                            except Exception as e:
                                logger.warning(f"Failed to decode JWT: {e}")

                if IS_ZEROGPU:
                    logger.info(f"User: {username} | Logged in: {is_logged_in}")
            except Exception as e:
                logger.warning(f"Failed to detect user: {e}")
                import traceback

                logger.warning(traceback.format_exc())

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
                timings=timings,
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
                timings=timings,
            )

    # Log resource event if telemetry is enabled
    if analysis_id is not None:
        from mosaic.telemetry import TelemetryTracker
        from mosaic.hardware import get_gpu_metrics

        total_duration = time_module.time() - slide_start_time
        gpu_metrics = get_gpu_metrics()

        tracker = TelemetryTracker.get_instance()
        tracker.log_resource_event(
            analysis_id=analysis_id,
            session_hash=session_hash,
            total_duration_sec=total_duration,
            segmentation_duration_sec=timings.get("segmentation"),
            ctranspath_duration_sec=timings.get("ctranspath"),
            optimus_duration_sec=timings.get("optimus"),
            aeon_duration_sec=timings.get("aeon"),
            paladin_duration_sec=timings.get("paladin"),
            tile_count=len(coords) if coords is not None else None,
            filtered_tile_count=timings.get("filtered_tile_count"),
            batch_size=timings.get("batch_size"),
            gpu_type=gpu_metrics.get("gpu_type"),
            peak_gpu_memory_gb=gpu_metrics.get("max_memory_allocated_gb"),
        )

    return slide_mask, aeon_results, paladin_results
