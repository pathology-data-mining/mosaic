"""Gradio web interface for Mosaic.

This module provides the web-based user interface for analyzing whole slide images.
It includes functionality for:
- Multi-slide upload and analysis
- TCGA slide fetching by ID (UUID or barcode)
- Settings configuration (site type, cancer subtype, IHC subtype, segmentation)
- Results visualization and export
- CSV-based batch processing
"""

import os
import tempfile
import threading
import time
import uuid

import gradio as gr
import pandas as pd
from pathlib import Path
from loguru import logger
from PIL import Image

from mosaic.ui.utils import (
    get_oncotree_code_name,
    create_user_directory,
    load_settings,
    validate_settings,
    get_tissue_sites,
    IHC_SUBTYPES,
    SEX_OPTIONS,
    SETTINGS_COLUMNS,
)
from mosaic.analysis import analyze_slide
from mosaic.model_manager import load_all_models
from mosaic.hardware import DEFAULT_CONCURRENCY_LIMIT, IS_T4_GPU, GPU_TYPE
from mosaic.tcga import (
    fetch_slide,
    get_case_metadata,
    validate_slide_id,
    GDCError,
    GDCInvalidIDError,
    GDCNotFoundError,
    GDCAccessDeniedError,
)

current_dir = Path(__file__).parent.parent


_GALLERY_MAX_SIZE = 512

# Save masks directly into Gradio's cache directory so that Gradio skips its
# hash-and-copy step in move_files_to_cache (which reads the entire file from
# disk to SHA-256 it, then copies it — expensive on network filesystems).
_GRADIO_CACHE_DIR = Path(
    os.environ.get("GRADIO_TEMP_DIR") or (Path(tempfile.gettempdir()) / "gradio")
)


def _save_mask_as_png(mask_image, name):
    """Downsample and save a PIL mask image as PNG for the browser Gallery.

    Images are downsampled to _GALLERY_MAX_SIZE to match the Gallery's rendered
    thumbnail size (3 columns ~ 250px each).

    Files are saved into Gradio's cache directory to avoid the hash-and-copy
    overhead in Gradio's postprocessing pipeline.

    Args:
        mask_image: PIL Image object to save
        name: Filename for the saved image

    Returns:
        Path to the saved PNG file, or None if mask_image is invalid
    """
    if not isinstance(mask_image, Image.Image):
        return None
    w, h = mask_image.size
    if w > _GALLERY_MAX_SIZE or h > _GALLERY_MAX_SIZE:
        scale = _GALLERY_MAX_SIZE / max(w, h)
        mask_image = mask_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    cache_subdir = _GRADIO_CACHE_DIR / "mosaic_masks"
    cache_subdir.mkdir(parents=True, exist_ok=True)
    path = cache_subdir / name
    mask_image.save(path, format="PNG")
    return str(path)


# Global variables for cancer subtypes (set by download_and_process_models)
cancer_subtype_name_map = {}
reversed_cancer_subtype_name_map = {}
cancer_subtypes = []

# Global model cache for T4 (to persist models across sequential requests)
_global_model_cache = None

# Background download thread for model initialization
_model_download_thread = None
_model_download_complete = False
_core_models_complete = False  # Track core models separately
_shutdown_requested = False  # Flag to signal graceful shutdown


def set_cancer_subtype_maps(csn_map, rcsn_map, cs):
    """Set the global cancer subtype maps."""
    global cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes
    cancer_subtype_name_map = csn_map
    reversed_cancer_subtype_name_map = rcsn_map
    cancer_subtypes = cs


def start_background_model_download():
    """Start downloading all models in a background thread.

    Downloads core models (CTransPath, Optimus, Aeon) and Paladin models asynchronously.
    This allows Gradio to start immediately and satisfy HF Spaces health checks.
    """
    global _model_download_thread, _model_download_complete, _core_models_complete

    if _model_download_thread is not None:
        logger.info("Background model download already started, skipping")
        return

    def download_worker():
        global _core_models_complete, _model_download_complete, _shutdown_requested
        try:
            from mosaic.gradio_app import download_core_models, download_paladin_models

            logger.info("🔄 Starting background download of all models...")

            # Check for shutdown before starting
            if _shutdown_requested:
                logger.info("Shutdown requested, aborting model download")
                return

            download_core_models()
            _core_models_complete = True
            logger.info("✅ Core models downloaded (analysis can proceed)")

            # Check for shutdown before Paladin download
            if _shutdown_requested:
                logger.info("Shutdown requested, skipping Paladin model download")
                return

            download_paladin_models()
            _model_download_complete = True
            logger.info("✅ All model downloads complete")
        except RuntimeError as e:
            if "interpreter shutdown" in str(e) or "cannot schedule new futures" in str(
                e
            ):
                logger.info(
                    "Interpreter shutting down, stopping model download gracefully"
                )
                _core_models_complete = True  # Mark as complete to avoid blocking
            else:
                logger.error(f"❌ Error during background model download: {e}")
                import traceback

                traceback.print_exc()
        except Exception as e:
            logger.error(f"❌ Error during background model download: {e}")
            import traceback

            traceback.print_exc()

    import threading

    _model_download_thread = threading.Thread(target=download_worker, daemon=True)
    _model_download_thread.start()
    logger.info("🚀 Model download started in background thread")


def analyze_slides(
    slides,
    settings_input,
    site_type,
    sex,
    tissue_site,
    cancer_subtype,
    ihc_subtype,
    seg_config,
    user_dir,
    progress=gr.Progress(track_tqdm=True),
    request: gr.Request = None,
):
    global _model_download_thread, _model_download_complete, _core_models_complete

    # Initialize telemetry tracking
    from mosaic.telemetry import TelemetryTracker

    tracker = TelemetryTracker.get_instance()
    analysis_id = str(uuid.uuid4())
    start_time = time.time()
    session_hash = request.session_hash if request else None

    # Wait for core models download to complete (Paladin models can continue in background)
    if _model_download_thread is not None and not _core_models_complete:
        logger.info("Waiting for core models to download...")
        progress(0, desc="Waiting for core models to download...")

        while not _core_models_complete:
            time.sleep(0.5)
        logger.info(
            "Core models ready, proceeding with analysis (Paladin models may still be downloading)"
        )

    if slides is None or len(slides) == 0:
        raise gr.Error("Please upload at least one slide.")
    if user_dir is None:
        if request is not None:
            user_dir = create_user_directory(None, request)
        if user_dir is None:
            # Fallback to temp directory if session hash not available
            import tempfile

            user_dir = Path(tempfile.mkdtemp(prefix="mosaic_"))

    # Handle empty settings_input (e.g., when dataframe is hidden for single slide)
    # Regenerate settings from dropdowns if settings_input is empty
    if settings_input is None or len(settings_input) == 0:
        logger.info("Settings dataframe is empty, regenerating from dropdown values")
        settings = []
        for file in slides:
            filename = file.name if hasattr(file, "name") else file
            slide_name = filename.split("/")[-1]
            settings.append(
                [
                    slide_name,
                    site_type,
                    sex,
                    tissue_site,
                    cancer_subtype,
                    ihc_subtype,
                    seg_config,
                ]
            )
        settings_input = pd.DataFrame(settings, columns=SETTINGS_COLUMNS)

    settings_input = validate_settings(
        settings_input,
        cancer_subtype_name_map,
        cancer_subtypes,
        reversed_cancer_subtype_name_map,
    )
    if len(slides) != len(settings_input):
        raise gr.Error("Missing settings for uploaded slides")

    # Check that all slides have sex specified
    if (
        settings_input["Sex"].isna().any()
        or (settings_input["Sex"] == "").any()
        or (settings_input["Sex"] == None).any()
    ):
        raise gr.Error("Sex is required for all slides. Please select Male or Female.")

    all_slide_masks = []
    all_aeon_results = []
    all_paladin_results = []

    # Log analysis start event
    tracker.log_usage_event(
        event_type="analysis_start",
        analysis_id=analysis_id,
        session_hash=session_hash,
        slide_count=len(slides),
        site_type=site_type,
        cancer_subtype=cancer_subtype,
        seg_config=seg_config,
        gpu_type=GPU_TYPE,
    )

    # Load models once (for batch) or per-slide (for single)
    # On T4: Keep models loaded globally across all requests (concurrency=1 ensures no conflicts)
    # On high-memory GPUs: Load models per-batch, reload for single slides
    global _global_model_cache
    model_cache = None

    if IS_T4_GPU:
        # T4: Use global cache to keep models loaded across requests
        if _global_model_cache is None:
            logger.info("T4: Loading models once (will persist across all requests)")
            progress(0.0, desc="Loading models (one-time initialization)")
            _global_model_cache = load_all_models(
                use_gpu=True, aggressive_memory_mgmt=None
            )
        else:
            logger.info(f"T4: Reusing pre-loaded models from global cache")
        model_cache = _global_model_cache
    elif len(slides) > 1:
        logger.info(f"Batch mode: Loading models once for {len(slides)} slides")
        progress(0.0, desc=f"Loading models for batch processing")
        model_cache = load_all_models(use_gpu=True, aggressive_memory_mgmt=None)
    else:
        logger.info("Single-slide mode: models loaded within analyze_slide")

    try:
        # Process all slides with unified analyze_slide function
        for idx, slide_path in enumerate(slides):
            row = settings_input.iloc[idx]
            slide_name = row["Slide"]

            slides_remaining = len(slides) - idx
            logger.info(
                f"[{idx + 1}/{len(slides)}] Processing: {slide_name} ({slides_remaining} slides remaining)"
            )

            # Create a progress wrapper that prefixes all updates with slide info
            class SlideProgressWrapper:
                def __init__(self, base_progress, slide_idx, total_slides, slide_name):
                    self.base_progress = base_progress
                    self.slide_idx = slide_idx
                    self.total_slides = total_slides
                    self.slide_name = slide_name
                    self.slides_remaining = total_slides - slide_idx

                def __call__(self, current_progress=None, desc=None):
                    prefix = f"Slide {self.slide_idx + 1}/{self.total_slides}: {self.slide_name}"
                    if desc:
                        full_desc = f"{prefix} - {desc}"
                    else:
                        full_desc = f"{prefix} ({self.slides_remaining} remaining)"

                    if current_progress is not None:
                        # Scale progress within this slide's range
                        slide_start = self.slide_idx / self.total_slides
                        slide_end = (self.slide_idx + 1) / self.total_slides
                        scaled_progress = slide_start + (
                            current_progress * (slide_end - slide_start)
                        )
                        self.base_progress(scaled_progress, desc=full_desc)
                    else:
                        self.base_progress(
                            self.slide_idx / self.total_slides, desc=full_desc
                        )

                # Support for tqdm integration
                @property
                def track_tqdm(self):
                    return True

                # Delegate attribute access to base progress
                def __getattr__(self, name):
                    return getattr(self.base_progress, name)

            # Create wrapped progress object
            slide_progress_wrapper = SlideProgressWrapper(
                progress, idx, len(slides), slide_name
            )

            # Initial progress for this slide
            slide_progress_wrapper()

            slide_mask, aeon_results, paladin_results = analyze_slide(
                slide_path=slide_path,
                seg_config=row["Segmentation Config"],
                site_type=row["Site Type"],
                sex=row["Sex"],
                tissue_site=row.get("Tissue Site", "Unknown"),
                cancer_subtype=row["Cancer Subtype"],
                cancer_subtype_name_map=cancer_subtype_name_map,
                ihc_subtype=row.get("IHC Subtype", ""),
                num_workers=4,
                progress=slide_progress_wrapper,  # Pass wrapped progress
                request=request,
                model_cache=model_cache,  # Pre-loaded for batch, None for single
                analysis_id=analysis_id,  # For telemetry tracking
                session_hash=session_hash,  # For telemetry tracking
            )

            if slide_mask is not None:
                mask_filename = f"{Path(slide_name).stem}_mask.png"
                saved = _save_mask_as_png(slide_mask, mask_filename)
                if saved is not None:
                    all_slide_masks.append((saved, slide_name))
                else:
                    all_slide_masks.append((slide_mask, slide_name))
            if aeon_results is not None:
                # Rename "Confidence" column to slide name for proper concatenation
                aeon_results = aeon_results.rename(columns={"Confidence": slide_name})
                all_aeon_results.append(aeon_results)
            if paladin_results is not None:
                paladin_results.insert(
                    0, "Slide", pd.Series([slide_name] * len(paladin_results))
                )
                all_paladin_results.append(paladin_results)

        # Log analysis complete event (successful completion)
        duration_sec = time.time() - start_time
        tracker.log_usage_event(
            event_type="analysis_complete",
            analysis_id=analysis_id,
            session_hash=session_hash,
            slide_count=len(slides),
            duration_sec=duration_sec,
            success=True,
            gpu_type=GPU_TYPE,
        )

        # Upload telemetry immediately after analysis to ensure data is saved
        # before HF Spaces potentially goes to sleep
        tracker.upload_to_hf()

        progress(0.99, desc="Analysis complete, wrapping up results")

    except Exception as e:
        # Log failure event
        tracker.log_failure_event(
            error_type=type(e).__name__,
            error_message=str(e),
            error_stage="analysis_pipeline",
            analysis_id=analysis_id,
            session_hash=session_hash,
            slide_count=len(slides) if slides else 0,
            gpu_type=GPU_TYPE,
        )
        raise

    finally:
        # Clean up model cache if it was loaded for batch processing
        # On T4: Keep global cache loaded, only cleanup Paladin models
        # On high-memory GPUs: Cleanup everything after batch
        if model_cache is not None and not IS_T4_GPU:
            logger.info("Cleaning up model cache after batch")
            model_cache.cleanup()
        elif IS_T4_GPU and model_cache is not None:
            logger.info(
                "T4: Keeping core models loaded, cleaning up Paladin models only"
            )
            model_cache.cleanup_paladin()

    timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
    combined_paladin_results = (
        pd.concat(all_paladin_results, ignore_index=True)
        if all_paladin_results
        else pd.DataFrame()
    )
    combined_aeon_results = gr.DataFrame(visible=False)
    aeon_output = gr.DownloadButton(visible=False)
    if all_aeon_results:
        combined_aeon_results = pd.concat(all_aeon_results, axis=1)
        combined_aeon_results.reset_index(inplace=True)

        combined_aeon_results = combined_aeon_results.round(3)
        cancer_subtype_names = [
            f"{get_oncotree_code_name(code)} ({code})"
            for code in combined_aeon_results["Cancer Subtype"]
        ]
        combined_aeon_results["Cancer Subtype"] = cancer_subtype_names

        aeon_output_path = user_dir / f"aeon_results-{timestamp}.csv"
        combined_aeon_results.to_csv(aeon_output_path)

        combined_aeon_results = gr.DataFrame(
            combined_aeon_results,
            visible=True,
            column_widths=["4px"] + ["2px"] * (combined_aeon_results.shape[1] - 1),
        )
        aeon_output = gr.DownloadButton(value=aeon_output_path, visible=True)

    # Convert Oncotree codes to names for display
    paladin_output = gr.DownloadButton(visible=False)
    if len(combined_paladin_results) > 0:
        cancer_subtype_names = [
            f"{get_oncotree_code_name(code)} ({code})"
            for code in combined_paladin_results["Cancer Subtype"]
        ]
        combined_paladin_results["Cancer Subtype"] = cancer_subtype_names

        # Ensure Score is numeric before rounding
        combined_paladin_results["Score"] = pd.to_numeric(
            combined_paladin_results["Score"], errors="coerce"
        )
        combined_paladin_results["Score"] = combined_paladin_results["Score"].round(3)

        paladin_output_path = user_dir / f"paladin_results-{timestamp}.csv"
        combined_paladin_results.to_csv(paladin_output_path, index=False)
        paladin_output = gr.DownloadButton(value=paladin_output_path, visible=True)

    progress(1.0, desc="All done!")

    # Final yield with complete results
    # Hide settings table if only one slide, keep visible for multiple slides
    settings_visible = len(slides) > 1

    # Store final results before cleanup
    final_slide_masks = all_slide_masks
    final_combined_paladin = (
        combined_paladin_results if len(combined_paladin_results) > 0 else None
    )

    # Memory cleanup: Clear intermediate data structures from RAM
    import gc

    all_slide_masks = None
    all_aeon_results = None
    all_paladin_results = None
    combined_paladin_results = None

    # Force garbage collection to free Python memory
    gc.collect()

    yield (
        gr.Dataframe(
            value=settings_input, visible=settings_visible
        ),  # Hide if single slide
        final_slide_masks,
        combined_aeon_results,
        aeon_output,
        final_combined_paladin,
        paladin_output,
        user_dir,
        gr.Button(interactive=True),  # Re-enable analyze button
        gr.Button(interactive=True),  # Re-enable clear button
    )


def launch_gradio(server_name, server_port, share):
    # Initialize telemetry and log app start
    from mosaic.telemetry import TelemetryTracker
    import os

    tracker = TelemetryTracker.get_instance()
    tracker.log_app_start()

    # Start downloading models in background
    start_background_model_download()

    with gr.Blocks(title="Mosaic") as demo:
        user_dir_state = gr.State(None)
        gr.Markdown(
            "# Mosaic: H&E Whole Slide Image Cancer Subtype and Biomarker Inference"
        )
        gr.Markdown(
            "Upload an H&E whole slide image in SVS or TIFF format, or fetch a slide directly from TCGA. The slide will be processed to infer cancer subtype and relevant biomarkers."
        )

        # Comprehensive documentation section
        with gr.Accordion("📖 Documentation & Help", open=False):
            gr.Markdown(
                """
## Quick Start Guide

**Single Slide Analysis:**
1. Upload an H&E whole slide image (.svs, .tiff, or .tif)
2. Configure settings:
   - **Sex**: Required - Patient biological sex (affects analysis)
   - **Site Type**: Primary tumor or metastatic site
   - **Tissue Site**: Anatomical location (if known)
   - **Cancer Subtype**: Leave as "Unknown" for automatic inference
   - **Segmentation Config**: Choose based on slide type
     - **Biopsy**: For small tissue samples
     - **Resection**: For surgical specimens
     - **TCGA**: For TCGA-compatible slides
3. Click "Analyze" and wait for results

**Fetch from TCGA:**
1. Select "Fetch from TCGA" input mode
2. Enter a TCGA slide ID (UUID or barcode like TCGA-A1-A0SB-01Z-00-DX1)
3. Click "Fetch Slide" - metadata will auto-fill the dropdowns
4. Review/adjust settings, then click "Analyze"

**Batch Processing:**
- Upload multiple slides at once to process them sequentially
- Use "Settings CSV Upload" for batch configuration with a CSV file

---

## Input Requirements

**Supported Formats:**
- SVS (Aperio ScanScope Virtual Slide)
- TIFF/TIF (Multi-resolution pyramid TIFF)

**File Requirements:**
- H&E stained whole slide images
- Standard microscopy scanning resolution
- Compatible with OpenSlide library

**Configuration Fields:**
- **Sex** (Required): Male/Female - Used for sex-specific cancer analysis
- **Site Type**: Primary vs Metastatic - Affects biomarker interpretation
- **Tissue Site**: 60+ anatomical sites supported (see dropdown)
- **Cancer Subtype**: Use "Unknown" for automatic inference via Aeon model

---

## Understanding Your Results

**Slide Masks:**
Visual representation of tissue segmentation showing analyzed regions

**Cancer Subtype Predictions:**
- Uses **Aeon foundation model** for cancer type classification
- Confidence scores range 0-1 (higher = more confident)
- Displays top predicted subtypes with OncoTree codes

**Biomarker Predictions:**
- Uses **Paladin model** for biomarker inference
- Biomarkers are cancer-subtype specific
- Scores indicate predicted expression/mutation status

**OncoTree Codes:**
Standardized cancer classification system (e.g., LUAD = Lung Adenocarcinoma, BRCA = Breast Invasive Carcinoma). Learn more at [oncotree.mskcc.org](https://oncotree.mskcc.org/)

---

## Processing Information

**Expected Processing Time:**
- Single slide: 5-15 minutes (depending on slide size)
- Multiple slides: Processed sequentially
- Results appear progressively as each slide completes

**Hardware:**
- Running on GPU-accelerated infrastructure
- Models: CTransPath, Optimus, Aeon (subtype), Paladin (biomarker)
- Background model loading may occur on first use

---

## CSV Batch Upload Format

Upload a CSV file with these columns:
```
Slide,Site Type,Sex,Tissue Site,Cancer Subtype,Segmentation Config
slide1.svs,Primary,Female,Breast,Unknown,Resection
slide2.tif,Metastatic,Male,Lung,LUAD,Biopsy
```

**Tips:**
- Slide names must match uploaded filenames
- Use "Unknown" for automatic cancer subtype inference
- All rows must have Sex specified (Male or Female)

---

## TCGA Slide Fetching

**Supported ID Formats:**
- **UUID**: 36-character identifier (e.g., `9c960533-ce6d-4a77-9227-eb974710a607`)
- **TCGA Barcode**: Standard TCGA format (e.g., `TCGA-E2-A1LS-01Z-00-DX1`)

**Example Slide IDs (Public Access, with Paladin biomarker models):**
| ID | Cancer Type | Format |
|----|-------------|--------|
| `6a0ea716-a5f2-47f3-880b-537a5cdc2324` | Lung Adenocarcinoma (LUAD) | UUID |
| `d65c5d21-6333-4a9e-9a2a-139a122a3c8a` | Colon Adenocarcinoma (COAD) | UUID |
| `1bc2766c-7064-43c7-97cb-5962d3894671` | Prostate Adenocarcinoma (PRAD) | UUID |
| `0d083e4f-0037-462e-b810-3182506f15ba` | Bladder Cancer (BLCA) | UUID |
| `TCGA-86-8074-01Z-00-DX1` | Lung Adenocarcinoma (LUAD) | Barcode |
| `TCGA-AZ-4313-01Z-00-DX1` | Colon Adenocarcinoma (COAD) | Barcode |

**How It Works:**
1. Slides are downloaded from the GDC (Genomic Data Commons) API
2. Downloaded slides are cached locally to avoid re-downloading
3. Clinical metadata (sex, tissue site, cancer subtype) is extracted and pre-filled

**Authentication:**
- Public TCGA slides can be fetched without authentication
- Controlled-access data requires a GDC token (set via `GDC_TOKEN` environment variable)

**Cache Location:**
- Default: `~/.cache/mosaic/tcga_slides/`
- Override with `MOSAIC_TCGA_CACHE_DIR` environment variable

---

## Important Limitations

**Research Use Only:**
This tool is for research purposes only and not approved for clinical diagnosis.

**Model Limitations:**
- Predictions are probabilistic and may contain errors
- Performance varies by tissue type and slide quality
- Best results with standard H&E staining protocols
- Not validated for all rare cancer subtypes

**Quality Requirements:**
- Slides should have clear H&E staining
- Adequate tissue present (not mostly blank)
- Standard microscopy resolution

---

## Models & Technical Details

**Models Used:**
- **[CTransPath](https://github.com/Xiyue-Wang/TransPath)**: Used for filtering marker tiles (not the main analysis model)
- **[Optimus-0](https://huggingface.co/bioptimus/H-optimus-0)**: Vision-language foundation model (main feature extraction)
- **Aeon**: Cancer subtype classification
- **Paladin**: Biomarker prediction

**Learn More:**
- [Mosaic GitHub Repository](https://github.com/pathology-data-mining/mosaic)

---

## Troubleshooting

**Common Issues:**
- **"Sex is required"**: Select Male or Female before analyzing
- **Slow processing**: Large slides take longer; be patient
- **Upload failed**: Check file format (.svs, .tiff, .tif only)
- **No results**: Ensure slide contains sufficient tissue
            """
            )

        with gr.Row():
            with gr.Column():
                # Input mode selection
                input_mode = gr.Radio(
                    choices=["Upload Files", "Fetch from TCGA"],
                    value="Upload Files",
                    label="Input Mode",
                    info="Choose how to provide the slide",
                )

                # File upload component (visible by default)
                input_slides = gr.File(
                    label="Upload H&E Whole Slide Image (SVS or TIFF format)",
                    file_types=[".svs", ".tiff", ".tif"],
                    file_count="multiple",
                )

                # TCGA fetch components (hidden by default)
                with gr.Group(visible=False) as tcga_input_group:
                    tcga_id_input = gr.Textbox(
                        label="TCGA Slide ID",
                        placeholder="Enter UUID or TCGA barcode",
                        info="Supports UUID format or TCGA barcode format",
                    )
                    gr.Examples(
                        examples=[
                            [
                                "6a0ea716-a5f2-47f3-880b-537a5cdc2324"
                            ],  # TCGA-LUAD (Lung) - has Paladin models
                            [
                                "d65c5d21-6333-4a9e-9a2a-139a122a3c8a"
                            ],  # TCGA-COAD (Colon) - has Paladin models
                            [
                                "1bc2766c-7064-43c7-97cb-5962d3894671"
                            ],  # TCGA-PRAD (Prostate) - has Paladin models
                            [
                                "0d083e4f-0037-462e-b810-3182506f15ba"
                            ],  # TCGA-BLCA (Bladder) - has Paladin models
                            ["TCGA-86-8074-01Z-00-DX1"],  # TCGA-LUAD barcode (Lung)
                            ["TCGA-AZ-4313-01Z-00-DX1"],  # TCGA-COAD barcode (Colon)
                        ],
                        inputs=tcga_id_input,
                        label="Example TCGA Slide IDs (click to use)",
                    )
                    with gr.Row():
                        fetch_button = gr.Button("Fetch Slide", variant="secondary")
                    tcga_status = gr.Textbox(
                        label="Fetch Status",
                        interactive=False,
                        visible=False,
                    )

                # State to track fetched slide path
                tcga_slide_path_state = gr.State(None)

                site_dropdown = gr.Dropdown(
                    choices=["Primary", "Metastatic"],
                    label="Site Type",
                    value="Primary",
                    info="Primary tumor or metastatic site - affects biomarker interpretation",
                )
                sex_dropdown = gr.Dropdown(
                    choices=SEX_OPTIONS,
                    label="Sex",
                    value=None,
                    info="Required - Patient biological sex (used for sex-specific cancer analysis)",
                )
                tissue_site_dropdown = gr.Dropdown(
                    choices=get_tissue_sites(),
                    label="Tissue Site",
                    value="Unknown",
                    info="Anatomical location of the tissue sample (if known)",
                )
                cancer_subtype_dropdown = gr.Dropdown(
                    choices=[name for name in cancer_subtype_name_map.keys()],
                    label="Cancer Subtype",
                    value="Unknown",
                    info="Leave as 'Unknown' for automatic inference using the Aeon model",
                )
                # TODO: Enable IHC subtype dropdown in future work
                # ihc_subtype_dropdown = gr.Dropdown(
                #     choices=IHC_SUBTYPES,
                #     label="IHC Subtype (if applicable)",
                #     value="",
                #     visible=False,
                # )
                ihc_subtype_dropdown = gr.State("")  # Placeholder state for now
                seg_config_dropdown = gr.Dropdown(
                    choices=["Biopsy", "Resection", "TCGA"],
                    label="Segmentation Config",
                    value="Biopsy",
                    info="Biopsy: small samples | Resection: surgical specimens | TCGA: TCGA-compatible slides",
                )
                with gr.Row():
                    settings_input = gr.Dataframe(
                        headers=SETTINGS_COLUMNS,
                        label="Current Settings (editable for multiple slides)",
                        datatype=["str"] * len(SETTINGS_COLUMNS),
                        visible=False,
                        interactive=True,
                        static_columns="Slide",
                    )

                with gr.Row():
                    settings_csv = gr.File(
                        file_types=[".csv"],
                        label="Upload Settings CSV (columns: Slide, Site Type, Sex, Tissue Site, Cancer Subtype, Segmentation Config)",
                        visible=False,
                    )

                with gr.Row():
                    clear_button = gr.Button("Clear")
                    analyze_button = gr.Button("Analyze", variant="primary")
            with gr.Column():
                slide_masks = gr.Gallery(
                    label="Slide Masks (tissue segmentation)",
                    columns=3,
                    object_fit="contain",
                    height="auto",
                )
                aeon_output_table = gr.Dataframe(
                    headers=["Cancer Subtype", "Slide Name"],
                    label="Cancer Subtype Inference Confidence (Aeon model, scores 0-1)",
                    datatype=["str", "number"],
                    visible=False,
                )
                aeon_download_button = gr.DownloadButton(
                    "Download Aeon Results as CSV",
                    label="Download Results",
                    visible=False,
                )
                paladin_output_table = gr.Dataframe(
                    headers=["Slide", "Cancer Subtype", "Biomarker", "Score"],
                    label="Biomarker Inference (Paladin model)",
                    datatype=["str", "str", "str", "number"],
                )
                paladin_download_button = gr.DownloadButton(
                    "Download Paladin Results as CSV",
                    label="Download Results",
                    visible=False,
                )

        def get_settings(
            files, site_type, sex, tissue_site, cancer_subtype, ihc_subtype, seg_config
        ):
            """Generate initial settings DataFrame from uploaded files and dropdown values."""
            if files is None:
                return pd.DataFrame()
            settings = []
            for file in files:
                filename = file.name if hasattr(file, "name") else file
                slide_name = filename.split("/")[-1]
                settings.append(
                    [
                        slide_name,
                        site_type,
                        sex if sex is not None else "",
                        tissue_site,
                        cancer_subtype,
                        ihc_subtype,
                        seg_config,
                    ]
                )
            df = pd.DataFrame(settings, columns=SETTINGS_COLUMNS)
            return df

        def update_settings_column(settings_df, column_name, new_value):
            """Update a specific column in the settings DataFrame."""
            if settings_df is None or len(settings_df) == 0:
                return settings_df
            # Create a copy to avoid modifying the original
            updated_df = settings_df.copy()
            # Convert None to empty string for display (especially for Sex column)
            if new_value is None:
                new_value = ""
            # Convert legacy "Unknown" sex values to empty string
            if column_name == "Sex" and new_value == "Unknown":
                new_value = ""
            updated_df[column_name] = new_value
            return updated_df

        # Handle input mode switching
        @input_mode.change(
            inputs=[input_mode],
            outputs=[input_slides, tcga_input_group, settings_input, settings_csv],
        )
        def switch_input_mode(mode):
            """Switch between file upload and TCGA fetch modes."""
            if mode == "Upload Files":
                return (
                    gr.File(visible=True),
                    gr.Group(visible=False),
                    gr.Dataframe(visible=False),
                    gr.File(visible=False),
                )
            else:
                return (
                    gr.File(visible=False),
                    gr.Group(visible=True),
                    gr.Dataframe(visible=False),
                    gr.File(visible=False),
                )

        # Handle TCGA slide fetch
        def fetch_tcga_slide(tcga_id, seg_config, progress=gr.Progress()):
            """Fetch a slide from TCGA and pre-fill metadata."""
            if not tcga_id or not tcga_id.strip():
                raise gr.Error("Please enter a TCGA slide ID (UUID or barcode).")

            try:
                # Validate ID format first
                tcga_id = validate_slide_id(tcga_id)
                progress(0.1, desc="Validating slide ID...")

                # Get tissue sites for metadata mapping
                tissue_sites = get_tissue_sites()

                # Fetch the slide (will use cache if available)
                progress(
                    0.2,
                    desc="Fetching slide from TCGA (this may take a while for large files)...",
                )

                def update_progress(downloaded, total):
                    if total > 0:
                        pct = 0.2 + (downloaded / total) * 0.7
                        size_mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        progress(
                            pct, desc=f"Downloading: {size_mb:.1f} / {total_mb:.1f} MB"
                        )

                slide_path, file_uuid = fetch_slide(
                    tcga_id,
                    progress_callback=update_progress,
                )

                progress(0.9, desc="Extracting metadata...")

                # Get case metadata and convert to Mosaic values
                metadata = get_case_metadata(file_uuid, tissue_sites)

                progress(1.0, desc="Slide fetched successfully!")

                # Determine cancer subtype display value
                cancer_subtype_value = "Unknown"
                if metadata.cancer_subtype and metadata.cancer_subtype != "Unknown":
                    # Try to find the display name in the map
                    if metadata.cancer_subtype in reversed_cancer_subtype_name_map:
                        cancer_subtype_value = reversed_cancer_subtype_name_map[
                            metadata.cancer_subtype
                        ]
                    else:
                        # Code not in map - keep as Unknown, log warning
                        logger.warning(
                            f"Cancer subtype code '{metadata.cancer_subtype}' not found in "
                            "available subtypes. Using 'Unknown'."
                        )

                status_msg = f"Slide fetched: {metadata.filename or slide_path.name}"
                if metadata.sex or metadata.tissue_site != "Unknown":
                    status_msg += f"\nPre-filled: Sex={metadata.sex or 'N/A'}, Site={metadata.tissue_site}, Subtype={cancer_subtype_value}"
                status_msg += (
                    "\nSegmentation Config: TCGA (auto-selected for TCGA slides)"
                )

                return (
                    str(slide_path),  # tcga_slide_path_state
                    metadata.site_type or "Primary",  # site_dropdown
                    metadata.sex,  # sex_dropdown
                    metadata.tissue_site or "Unknown",  # tissue_site_dropdown
                    cancer_subtype_value,  # cancer_subtype_dropdown
                    "TCGA",  # seg_config_dropdown - Auto-select TCGA config for TCGA slides
                    gr.Textbox(value=status_msg, visible=True),  # tcga_status
                    gr.Dataframe(value=None, visible=False),  # settings_input
                    gr.File(visible=False),  # settings_csv
                )

            except GDCInvalidIDError as e:
                raise gr.Error(str(e))
            except GDCNotFoundError as e:
                raise gr.Error(str(e))
            except GDCAccessDeniedError as e:
                raise gr.Error(str(e))
            except GDCError as e:
                raise gr.Error(f"Error fetching slide: {e}")
            except Exception as e:
                logger.exception("Unexpected error fetching TCGA slide")
                raise gr.Error(f"Unexpected error: {e}")

        fetch_button.click(
            fetch_tcga_slide,
            inputs=[tcga_id_input, seg_config_dropdown],
            outputs=[
                tcga_slide_path_state,
                site_dropdown,
                sex_dropdown,
                tissue_site_dropdown,
                cancer_subtype_dropdown,
                seg_config_dropdown,  # Include seg_config_dropdown to auto-select TCGA
                tcga_status,
                settings_input,
                settings_csv,
            ],
        )

        # Handle file uploads - regenerate entire settings table
        @input_slides.change(
            inputs=[
                input_slides,
                site_dropdown,
                sex_dropdown,
                tissue_site_dropdown,
                cancer_subtype_dropdown,
                ihc_subtype_dropdown,
                seg_config_dropdown,
            ],
            outputs=[settings_input, settings_csv, ihc_subtype_dropdown],
        )
        def update_files(
            files, site_type, sex, tissue_site, cancer_subtype, ihc_subtype, seg_config
        ):
            """Handle file upload - regenerate settings table from scratch."""
            # has_ihc = "Breast" in cancer_subtype  # TODO: Enable when IHC dropdown is active
            if not files:
                return None, None, ""  # Return empty string for State component
            settings_df = get_settings(
                files,
                site_type,
                sex,
                tissue_site,
                cancer_subtype,
                ihc_subtype,
                seg_config,
            )
            # if settings_df is not None:
            #     has_ihc = any("Breast" in cs for cs in settings_df["Cancer Subtype"])
            visible = files and len(files) > 1
            return (
                gr.Dataframe(value=settings_df, visible=visible),
                gr.File(visible=visible),
                "",  # Return empty string for State component
            )

        # Handle individual dropdown changes - only update the relevant column
        @site_dropdown.change(
            inputs=[settings_input, site_dropdown],
            outputs=[settings_input],
        )
        def update_site_type(settings_df, site_type):
            """Update Site Type column when dropdown changes."""
            if settings_df is None or len(settings_df) == 0:
                return settings_df
            updated_df = update_settings_column(settings_df, "Site Type", site_type)
            return gr.Dataframe(value=updated_df)

        @sex_dropdown.change(
            inputs=[settings_input, sex_dropdown],
            outputs=[settings_input],
        )
        def update_sex(settings_df, sex):
            """Update Sex column when dropdown changes."""
            if settings_df is None or len(settings_df) == 0:
                return settings_df
            updated_df = update_settings_column(settings_df, "Sex", sex)
            return gr.Dataframe(value=updated_df)

        @tissue_site_dropdown.change(
            inputs=[settings_input, tissue_site_dropdown],
            outputs=[settings_input],
        )
        def update_tissue_site(settings_df, tissue_site):
            """Update Tissue Site column when dropdown changes."""
            if settings_df is None or len(settings_df) == 0:
                return settings_df
            updated_df = update_settings_column(settings_df, "Tissue Site", tissue_site)
            return gr.Dataframe(value=updated_df)

        @cancer_subtype_dropdown.change(
            inputs=[settings_input, cancer_subtype_dropdown],
            outputs=[settings_input, ihc_subtype_dropdown],
        )
        def update_cancer_subtype(settings_df, cancer_subtype):
            """Update Cancer Subtype column when dropdown changes."""
            # has_ihc = "Breast" in cancer_subtype  # TODO: Enable when IHC dropdown is active
            if settings_df is None or len(settings_df) == 0:
                return settings_df, ""  # Return empty string for State component
            updated_df = update_settings_column(
                settings_df, "Cancer Subtype", cancer_subtype
            )
            return (
                gr.Dataframe(value=updated_df),
                "",
            )  # Return empty string for State component

        # TODO: Enable IHC subtype change handler in future work
        # @ihc_subtype_dropdown.change(
        #     inputs=[settings_input, ihc_subtype_dropdown],
        #     outputs=[settings_input],
        # )
        # def update_ihc_subtype(settings_df, ihc_subtype):
        #     """Update IHC Subtype column when dropdown changes."""
        #     if settings_df is None or len(settings_df) == 0:
        #         return settings_df
        #     updated_df = update_settings_column(settings_df, "IHC Subtype", ihc_subtype)
        #     return gr.Dataframe(value=updated_df)

        @seg_config_dropdown.change(
            inputs=[settings_input, seg_config_dropdown],
            outputs=[settings_input],
        )
        def update_seg_config(settings_df, seg_config):
            """Update Segmentation Config column when dropdown changes."""
            if settings_df is None or len(settings_df) == 0:
                return settings_df
            updated_df = update_settings_column(
                settings_df, "Segmentation Config", seg_config
            )
            return gr.Dataframe(value=updated_df)

        @settings_csv.upload(
            inputs=[settings_csv],
            outputs=[settings_input],
        )
        def read_settings(file):
            if file is None:
                return None
            df = load_settings(file.name if hasattr(file, "name") else file)
            return gr.Dataframe(df, visible=True)

        def analyze_with_tcga_support(
            input_mode_val,
            slides,
            tcga_slide_path,
            settings_input_val,
            site_type,
            sex,
            tissue_site,
            cancer_subtype,
            ihc_subtype,
            seg_config,
            user_dir,
            progress=gr.Progress(track_tqdm=True),
            request: gr.Request = None,
        ):
            """Wrapper to handle both file upload and TCGA slide inputs."""
            # Determine the slides to analyze based on input mode
            if input_mode_val == "Fetch from TCGA":
                if not tcga_slide_path:
                    raise gr.Error(
                        "Please fetch a TCGA slide first using the 'Fetch Slide' button."
                    )
                # Create a mock file-like object for the TCGA slide
                slides = [Path(tcga_slide_path)]

                # Regenerate settings for the single TCGA slide to avoid issues
                # with stale settings from previously uploaded slides
                slide_name = Path(tcga_slide_path).name
                settings_input_val = pd.DataFrame(
                    [
                        [
                            slide_name,
                            site_type,
                            sex,
                            tissue_site,
                            cancer_subtype,
                            ihc_subtype,
                            seg_config,
                        ]
                    ],
                    columns=SETTINGS_COLUMNS,
                )
            elif slides is None or len(slides) == 0:
                raise gr.Error("Please upload at least one slide.")

            # Call the original analyze_slides function
            yield from analyze_slides(
                slides,
                settings_input_val,
                site_type,
                sex,
                tissue_site,
                cancer_subtype,
                ihc_subtype,
                seg_config,
                user_dir,
                progress=progress,
                request=request,
            )

        analyze_button.click(
            analyze_with_tcga_support,
            inputs=[
                input_mode,
                input_slides,
                tcga_slide_path_state,
                settings_input,
                site_dropdown,
                sex_dropdown,
                tissue_site_dropdown,
                cancer_subtype_dropdown,
                ihc_subtype_dropdown,
                seg_config_dropdown,
                user_dir_state,
            ],
            outputs=[
                settings_input,
                slide_masks,
                aeon_output_table,
                aeon_download_button,
                paladin_output_table,
                paladin_download_button,
                user_dir_state,
                analyze_button,  # Add analyze_button as output to control its state
                clear_button,  # Add clear_button as output to control its state
            ],
            queue=True,
            show_progress_on=paladin_output_table,
        )
        settings_input.change(
            lambda df: validate_settings(
                df,
                cancer_subtype_name_map,
                cancer_subtypes,
                reversed_cancer_subtype_name_map,
            ),
            inputs=[settings_input],
            outputs=[settings_input],
        )

        @clear_button.click(
            outputs=[
                input_slides,
                slide_masks,
                paladin_output_table,
                paladin_download_button,
                aeon_output_table,
                aeon_download_button,
                settings_input,
                settings_csv,
                tcga_id_input,
                tcga_status,
                tcga_slide_path_state,
                input_mode,
                tcga_input_group,
            ],
        )
        def clear_fn():
            return (
                None,  # input_slides
                None,  # slide_masks
                None,  # paladin_output_table
                gr.DownloadButton(visible=False),  # paladin_download_button
                gr.Dataframe(visible=False),  # aeon_output_table
                gr.DownloadButton(visible=False),  # aeon_download_button
                gr.Dataframe(visible=False),  # settings_input
                gr.File(visible=False),  # settings_csv
                "",  # tcga_id_input
                gr.Textbox(visible=False),  # tcga_status
                None,  # tcga_slide_path_state
                "Upload Files",  # input_mode - reset to default
                gr.Group(visible=False),  # tcga_input_group - hide TCGA inputs
            )

        demo.load(
            create_user_directory,
            inputs=[user_dir_state],
            outputs=[user_dir_state],
        )

    # Use hardware-specific concurrency limit
    # T4 GPUs (16GB) can only handle one analysis at a time to prevent OOM
    # Higher-memory GPUs and ZeroGPU can handle multiple concurrent analyses
    demo.queue(max_size=10, default_concurrency_limit=DEFAULT_CONCURRENCY_LIMIT)

    # Register cleanup handler for graceful shutdown
    import atexit

    def cleanup_on_exit():
        """Signal background thread to stop and log telemetry shutdown."""
        global _shutdown_requested
        _shutdown_requested = True
        logger.info("Shutdown signal sent to background model download thread")

        # Log app shutdown for telemetry
        tracker.log_app_shutdown()

        # Upload telemetry to HF Dataset if configured
        if os.environ.get("MOSAIC_TELEMETRY_HF_REPO"):
            tracker.upload_to_hf()

    atexit.register(cleanup_on_exit)

    # Start heartbeat thread for telemetry (every 5 minutes)
    # Also uploads telemetry on each heartbeat to ensure data is saved before HF Spaces sleep
    def heartbeat_worker():
        while not _shutdown_requested:
            time.sleep(tracker.config.heartbeat_interval_sec)
            tracker.log_heartbeat()
            # Upload telemetry on each heartbeat to avoid data loss when HF Spaces goes to sleep
            # (atexit handlers may not run when the container is frozen/killed)
            tracker.upload_to_hf()

    heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
    heartbeat_thread.start()

    demo.launch(
        server_name=server_name,
        share=share,
        server_port=server_port,
        show_error=True,
        favicon_path=current_dir / "favicon.svg",
    )
