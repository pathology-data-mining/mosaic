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
from typing import Optional

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
from mosaic.data_directory import get_tcga_cache_directory
from mosaic.model_manager import load_all_models
from mosaic.hardware import DEFAULT_CONCURRENCY_LIMIT, IS_HF_SPACES, IS_T4_GPU, GPU_TYPE
from mosaic.telemetry import extract_user_info
from mosaic.tcga import (
    compute_settings_hash,
    download_results_from_hf,
    fetch_slide,
    get_case_metadata,
    load_analysis_results,
    save_analysis_results,
    upload_results_to_hf,
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
    slide_ids=None,  # Mapping from filename to slide_id
    progress=gr.Progress(track_tqdm=True),
    request: gr.Request = None,
    profile: Optional[gr.OAuthProfile] = None,
):
    global _model_download_thread, _model_download_complete, _core_models_complete

    def _truncate_name(name: str, max_length: int = 40) -> str:
        """Truncate long filenames for progress display.

        Args:
            name: Filename to truncate
            max_length: Maximum length before truncation

        Returns:
            Truncated filename with ellipsis if needed
        """
        if len(name) <= max_length:
            return name
        # Keep the extension visible
        stem = Path(name).stem
        ext = Path(name).suffix
        # Reserve space for extension (including dot) and ellipsis (3 chars)
        available = max_length - len(ext) - 3
        if available > 0:
            return f"{stem[:available]}...{ext}"
        # Edge case: extension itself is very long
        return f"{name[:max_length-3]}..."

    # Initialize telemetry tracking
    from mosaic.telemetry import TelemetryTracker

    tracker = TelemetryTracker.get_instance()
    analysis_id = str(uuid.uuid4())
    start_time = time.time()
    session_hash = request.session_hash if request else None

    # Extract user info for telemetry (HF Spaces only)
    # gr.LoginButton() OAuth provides user info via OAuthProfile
    user_info = extract_user_info(request, IS_HF_SPACES, profile)

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
    cached_slides = 0
    fresh_slides = 0

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
        is_logged_in=user_info.is_logged_in,
        hf_username=user_info.username,
    )

    # Disable analyze and clear buttons during analysis
    yield (
        gr.Dataframe(),  # settings_input (unchanged)
        [],  # slide_masks (empty initially)
        None,  # aeon_output_table (empty initially)
        gr.DownloadButton(visible=False),  # aeon_download_button (hidden)
        None,  # paladin_output_table (empty initially)
        gr.DownloadButton(visible=False),  # paladin_download_button (hidden)
        user_dir,  # user_dir_state (unchanged)
        gr.Button(interactive=False),  # Disable analyze button
        gr.Button(interactive=False),  # Disable clear button
    )

    # Defer model loading until needed - skip GPU loading if all slides have cached results
    # On T4: Reuse global cache if already loaded, otherwise load on first cache miss
    # On high-memory GPUs: Load on first cache miss for batch, per-slide for single
    global _global_model_cache
    model_cache = None

    if IS_T4_GPU and _global_model_cache is not None:
        logger.info("T4: Reusing pre-loaded models from global cache")
        model_cache = _global_model_cache

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
                    # Truncate long slide names for better progress display
                    self.slide_name = _truncate_name(slide_name)
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

            # Check if this is a TCGA slide and try to load cached results
            slide_mask, aeon_results, paladin_results = None, None, None
            is_tcga_slide = False
            file_uuid = None

            try:
                tcga_cache_dir = get_tcga_cache_directory()
                slide_path_obj = Path(slide_path)

                # Check if slide is under TCGA cache directory
                if tcga_cache_dir in slide_path_obj.parents:
                    # Extract UUID from parent directory name
                    file_uuid = slide_path_obj.parent.name
                    is_tcga_slide = True

                    # Compute settings hash for this slide
                    settings_hash = compute_settings_hash(
                        seg_config=row["Segmentation Config"],
                        site_type=row["Site Type"],
                        sex=row["Sex"],
                        tissue_site=row.get("Tissue Site", "Unknown"),
                        cancer_subtype=row["Cancer Subtype"],
                        ihc_subtype=row.get("IHC Subtype", ""),
                    )

                    # Try to load cached results
                    cached = load_analysis_results(file_uuid, settings_hash)
                    if cached is not None:
                        slide_mask, aeon_results, paladin_results = cached
                        cached_slides += 1
                        logger.info(
                            f"Using cached results for {file_uuid}/{settings_hash}"
                        )
                        # Skip analysis, use cached results
            except Exception as e:
                logger.warning(
                    f"Failed to check/load cached results for slide: {e}. "
                    f"Proceeding with fresh analysis."
                )
                # Continue with normal analysis on any error

            # If no cached results, run analysis
            if slide_mask is None or aeon_results is None or paladin_results is None:
                fresh_slides += 1
                # Lazy model loading: load on first cache miss
                if model_cache is None:
                    if IS_T4_GPU:
                        logger.info(
                            "T4: Loading models once (will persist across all requests)"
                        )
                        progress(0.0, desc="Loading models (one-time initialization)")
                        _global_model_cache = load_all_models(
                            use_gpu=True, aggressive_memory_mgmt=None
                        )
                        model_cache = _global_model_cache
                    elif len(slides) > 1:
                        logger.info(
                            f"Batch mode: Loading models once for {len(slides)} slides"
                        )
                        progress(0.0, desc="Loading models for batch processing")
                        model_cache = load_all_models(
                            use_gpu=True, aggressive_memory_mgmt=None
                        )
                    # else: single-slide mode - analyze_slide handles its own model loading

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
                    profile=profile,  # For OAuth user info
                )

                # Save results to cache if this is a TCGA slide
                if is_tcga_slide and file_uuid is not None:
                    try:
                        settings_hash = compute_settings_hash(
                            seg_config=row["Segmentation Config"],
                            site_type=row["Site Type"],
                            sex=row["Sex"],
                            tissue_site=row.get("Tissue Site", "Unknown"),
                            cancer_subtype=row["Cancer Subtype"],
                            ihc_subtype=row.get("IHC Subtype", ""),
                        )
                        save_analysis_results(
                            file_uuid=file_uuid,
                            settings_hash=settings_hash,
                            aeon_results=aeon_results,
                            paladin_results=paladin_results,
                            slide_mask=slide_mask,
                        )

                        # Upload results to HF Dataset for persistence
                        try:
                            upload_results_to_hf(file_uuid)
                        except Exception as upload_error:
                            logger.warning(
                                f"Failed to upload results to HF Dataset for {file_uuid}: {upload_error}. "
                                f"Results are cached locally but not synced to HF."
                            )

                    except Exception as e:
                        logger.warning(
                            f"Failed to save analysis results to cache for {file_uuid}: {e}. "
                            f"Results are still valid but won't be cached for future use."
                        )

            # Save results for logged-in users (best-effort, non-blocking)
            # On HF Spaces: only for logged-in users
            # Locally: always save for 'local_user' for debugging
            should_save_results = False
            save_username = None

            if IS_HF_SPACES:
                if user_info.is_logged_in and user_info.username:
                    should_save_results = True
                    save_username = user_info.username
            else:
                # Local mode - always save for debugging
                from mosaic.ui.user_tabs import LOCAL_DEBUG_USERNAME

                should_save_results = True
                save_username = LOCAL_DEBUG_USERNAME

            if should_save_results and save_username:
                try:
                    from mosaic.user_results import (
                        save_analysis_results as save_user_results,
                    )

                    # Prepare settings dict for metadata
                    settings_dict = {
                        "seg_config": row["Segmentation Config"],
                        "site_type": row["Site Type"],
                        "sex": row["Sex"],
                        "tissue_site": row.get("Tissue Site", "Unknown"),
                        "cancer_subtype": row["Cancer Subtype"],
                        # "ihc_subtype": row.get("IHC Subtype", ""),  # Not yet in use
                    }

                    # Use analysis_id + slide index for unique ID per slide
                    slide_analysis_id = f"{analysis_id}_{idx}"

                    # Get slide_id from tracking (if available)
                    file_slide_id = ""
                    if slide_ids and slide_name in slide_ids:
                        file_slide_id = slide_ids[slide_name]

                    save_user_results(
                        username=save_username,
                        analysis_id=slide_analysis_id,
                        slide_id=file_slide_id,
                        slide_name=slide_name,
                        settings=settings_dict,
                        aeon_results=aeon_results,
                        paladin_results=paladin_results,
                        slide_mask=slide_mask,
                    )
                    logger.debug(
                        f"Saved results for user {save_username}, slide {slide_name}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to save user results (non-fatal): {e}")

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
            cached_slide_count=cached_slides if cached_slides > 0 else None,
            gpu_type=GPU_TYPE,
            is_logged_in=user_info.is_logged_in,
            hf_username=user_info.username,
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
            is_logged_in=user_info.is_logged_in,
            hf_username=user_info.username,
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
    # Hide Aeon results when all slides had a user-supplied cancer subtype
    # (Aeon inference was skipped, so showing a synthetic 100% row is not useful)
    all_subtypes_known = all(
        row["Cancer Subtype"] not in ["Unknown", None, ""]
        for _, row in settings_input.iterrows()
    )

    combined_aeon_results = gr.DataFrame(visible=False)
    aeon_output = gr.DownloadButton(visible=False)
    if all_aeon_results and not all_subtypes_known:
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
        slide_ids_state = gr.State(
            {}
        )  # Maps slide filename to slide_id for user storage

        # Add login button for OAuth.
        # On HF Spaces the LoginButton is always hidden because sdk:docker
        # injects the Space owner's identity into request.username, causing
        # the button to show "Logged in as <owner>" for all visitors.
        # Instead we use a custom Markdown with direct OAuth route links.
        # The LoginButton must still exist in the layout so Gradio registers
        # the /login/huggingface and /logout OAuth routes (via expects_oauth).
        login_button = gr.LoginButton(visible=not IS_HF_SPACES)

        if IS_HF_SPACES:
            # LoginButton.activate() registers a load event (_check_login_status)
            # that returns a NEW LoginButton with visible=True, overriding our
            # visible=False. It reads session oauth_info (populated by the HF
            # reverse proxy with the Space owner's identity) and shows
            # "Logout (raylim)" to everyone. Clear the load event so the button
            # stays permanently hidden.
            login_button.load_event_to_attach = None

            # Custom login display — sign-in via markdown link,
            # sign-out via a Gradio button (stays in Gradio event system,
            # avoiding navigation away from the SPA).
            with gr.Row():
                login_status_md = gr.Markdown(
                    value="[Sign in with HuggingFace](/api/auth/login) to save slides and results",
                    visible=True,
                )
                logout_button = gr.Button(
                    "Sign out", visible=False, size="sm", scale=0,
                    min_width=80,
                )

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
| ID | Cancer Type |
|----|-------------|
| `TCGA-86-8074-01Z-00-DX1` | Lung Adenocarcinoma (LUAD) |
| `TCGA-AZ-4313-01Z-00-DX1` | Colon Adenocarcinoma (COAD) |
| `TCGA-EJ-7123-01Z-00-DX1` | Prostate Adenocarcinoma (PRAD) |
| `TCGA-G2-A2ES-01Z-00-DX1` | Bladder Cancer (BLCA) |
| `TCGA-HU-A4GJ-01Z-00-DX1` | Stomach Adenocarcinoma (STAD) |
| `TCGA-EY-A1GK-01Z-00-DX1` | Endometrial Carcinoma (UCEC) |

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

## Login & User Storage
"""
            )

            if IS_HF_SPACES:
                gr.Markdown(
                    """
**Logging In:**
- Click **Sign in with HuggingFace** at the top of the page to sign in with your HuggingFace account
- Login uses HuggingFace OAuth — you will be redirected to HuggingFace to authorize access
- After logging in, the page reloads and unlocks additional features

**Features Available After Login:**
- **My Files** tab: View, download, and delete your previously uploaded slides
- **My Results** tab: View, download, and delete your past analysis results
- **Storage Usage**: See how much of your storage quota is used
- **Re-analyze Existing Slides**: Select a previously uploaded slide from the dropdown without re-uploading
"""
                )
            else:
                gr.Markdown(
                    """
**Local Mode:**
- Login is not required — all user storage features are available automatically
- Files are stored under a debug username (`local_user`) in a temporary directory

**User Storage Features:**
- **My Files** tab: View, download, and delete your previously uploaded slides
- **My Results** tab: View, download, and delete your past analysis results
- **Storage Usage**: See how much of your storage quota is used
- **Re-analyze Existing Slides**: Select a previously uploaded slide from the dropdown without re-uploading
"""
                )

            gr.Markdown(
                """
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

                # Storage usage indicator (shown for logged-in users or local mode)
                # Initially visible in local mode, hidden on HF Spaces until login
                storage_usage_warning = gr.Markdown(visible=not IS_HF_SPACES)

                # Analyze Existing File option (shown for logged-in users or local mode)
                # Initially visible in local mode, hidden on HF Spaces until login
                with gr.Row(visible=not IS_HF_SPACES) as existing_file_row:
                    gr.Markdown("**Or select an existing slide:**")
                    existing_slides_dropdown = gr.Dropdown(
                        label="Previously Uploaded Slides",
                        choices=[],
                        value=None,
                        interactive=True,
                        multiselect=True,
                        info="Select one or more slides you've already uploaded to re-analyze",
                    )
                    refresh_existing_btn = gr.Button(
                        "🔄 Refresh List", size="sm", variant="secondary"
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
                            ["TCGA-86-8074-01Z-00-DX1"],  # LUAD (Lung)
                            ["TCGA-AZ-4313-01Z-00-DX1"],  # COAD (Colon)
                            ["TCGA-EJ-7123-01Z-00-DX1"],  # PRAD (Prostate)
                            ["TCGA-G2-A2ES-01Z-00-DX1"],  # BLCA (Bladder)
                            ["TCGA-HU-A4GJ-01Z-00-DX1"],  # STAD (Stomach)
                            ["TCGA-EY-A1GK-01Z-00-DX1"],  # UCEC (Endometrial)
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
                gr.Markdown(
                    """
                    **Tissue Segmentation Masks**: Cyan outlines show detected tissue regions. Red fill indicates tissue area.
                    """
                )
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
            files, site_type, sex, tissue_site, cancer_subtype, seg_config
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
            outputs=[
                input_slides,
                tcga_input_group,
                settings_input,
                settings_csv,
                existing_file_row,
            ],
        )
        def switch_input_mode(mode):
            """Switch between file upload and TCGA fetch modes."""
            if mode == "Upload Files":
                return (
                    gr.File(visible=True),
                    gr.Group(visible=False),
                    gr.Dataframe(visible=False),
                    gr.File(visible=False),
                    gr.Row(
                        visible=not IS_HF_SPACES
                    ),  # Show existing files row in upload mode
                )
            else:
                return (
                    gr.File(visible=False),
                    gr.Group(visible=True),
                    gr.Dataframe(visible=False),
                    gr.File(visible=False),
                    gr.Row(visible=False),  # Hide existing files row in TCGA mode
                )

        # Load storage usage and show warnings
        def load_storage_usage(
            request: gr.Request = None,
            profile: Optional[gr.OAuthProfile] = None,
        ):
            """Load and display storage usage with warnings."""
            from mosaic.ui.user_tabs import _get_username

            username, is_local = _get_username(request, profile)

            if not username:
                # Not logged in on HF Spaces - hide storage info
                return gr.Markdown(visible=False), gr.Row(visible=False)

            try:
                from mosaic.user_storage import get_storage_usage

                usage = get_storage_usage(username)
                quota_gb = usage.quota_bytes / (1024**3)
                used_gb = usage.total_bytes / (1024**3)
                percent = (
                    (usage.total_bytes / usage.quota_bytes * 100)
                    if usage.quota_bytes > 0
                    else 0
                )

                mode_indicator = " 🔧 [Local Debug]" if is_local else ""

                # Build warning message
                if percent >= 100:
                    warning_md = (
                        f"⛔ **Storage quota exceeded!** {used_gb:.2f} GB / {quota_gb:.0f} GB "
                        f"({percent:.0f}%){mode_indicator}\n\n"
                        f"Delete old files from the **My Files** tab before uploading."
                    )
                    warning_color = "red"
                elif percent >= 80:
                    warning_md = (
                        f"⚠️ **Storage nearly full:** {used_gb:.2f} GB / {quota_gb:.0f} GB "
                        f"({percent:.0f}%){mode_indicator}\n\n"
                        f"Consider deleting old files from the **My Files** tab."
                    )
                    warning_color = "orange"
                else:
                    warning_md = (
                        f"💾 **Storage:** {used_gb:.2f} GB / {quota_gb:.0f} GB "
                        f"({percent:.0f}%) | {usage.file_count} files{mode_indicator}"
                    )
                    warning_color = "gray"

                return gr.Markdown(value=warning_md, visible=True), gr.Row(visible=True)

            except Exception as e:
                logger.error(f"Failed to load storage usage: {e}")
                return gr.Markdown(visible=False), gr.Row(visible=False)

        # Load list of existing slides for re-analysis
        def load_existing_slides(
            request: gr.Request = None,
            profile: Optional[gr.OAuthProfile] = None,
        ):
            """Load dropdown with user's existing slides."""
            from mosaic.ui.user_tabs import _get_username

            username, _ = _get_username(request, profile)

            if not username:
                return gr.Dropdown(choices=[], value=None)

            try:
                from mosaic.user_storage import list_user_slides

                slides = list_user_slides(username)

                if not slides:
                    return gr.Dropdown(
                        choices=[], value=None, info="No slides uploaded yet"
                    )

                # Create dropdown choices: display name → slide_id
                choices = [
                    (
                        f"{slide.original_filename} ({slide.size_bytes / (1024**2):.1f} MB)",
                        slide.slide_id,
                    )
                    for slide in slides
                ]

                return gr.Dropdown(
                    choices=choices,
                    value=None,
                    info=f"{len(slides)} slide(s) available for re-analysis",
                )

            except Exception as e:
                logger.error(f"Failed to load existing slides: {e}")
                return gr.Dropdown(choices=[], value=None)

        # Handle slide upload and save to user storage
        def handle_slide_upload(
            files,
            slide_ids,
            request: gr.Request = None,
            profile: Optional[gr.OAuthProfile] = None,
        ):
            """Save uploaded slides to user storage and track slide IDs."""
            if not files:
                return files, slide_ids or {}

            # Determine username via OAuthProfile (primary) or request (fallback)
            from mosaic.ui.user_tabs import _get_username

            username, _ = _get_username(request, profile)
            if not username:
                return files, slide_ids or {}

            try:
                from mosaic.user_storage import save_uploaded_slide

                new_slide_ids = slide_ids.copy() if slide_ids else {}

                for file in files:
                    try:
                        # Save slide to user storage
                        slide_id, _ = save_uploaded_slide(username, file)

                        # Track mapping from filename to slide_id
                        filename = (
                            Path(file.name).name if hasattr(file, "name") else str(file)
                        )
                        new_slide_ids[filename] = slide_id

                        logger.info(
                            f"Saved slide {filename} with ID {slide_id} for user {username}"
                        )
                    except Exception as e:
                        # Best effort - don't fail upload if storage fails
                        logger.warning(
                            f"Failed to save slide {file} to user storage: {e}"
                        )

                return files, new_slide_ids

            except Exception as e:
                logger.error(f"Error in slide upload handler: {e}")
                # Don't fail the upload, just log the error
                return files, slide_ids or {}

        # Handle TCGA slide fetch
        def fetch_tcga_slide(tcga_id, seg_config, progress=gr.Progress()):
            """Fetch a slide from TCGA and pre-fill metadata."""
            if not tcga_id or not tcga_id.strip():
                raise gr.Error("Please enter a TCGA slide ID (UUID or barcode).")

            try:
                # Validate ID format first
                tcga_id = validate_slide_id(tcga_id)
                progress(0.1, desc="Validating slide ID...")

                # Determine file UUID for HF download
                from mosaic.tcga import is_valid_uuid, query_slide_uuid

                if is_valid_uuid(tcga_id):
                    file_uuid = tcga_id
                else:
                    # For barcodes, we need to query first to get UUID
                    file_uuid = query_slide_uuid(tcga_id)

                # Try to download cached results from HF Dataset (on demand)
                try:
                    download_results_from_hf(file_uuid)
                except Exception as e:
                    logger.warning(
                        f"Failed to download HF cache for {file_uuid}: {e}. "
                        f"Will analyze from scratch if no local cache exists."
                    )

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

                # Check if metadata is cached
                from mosaic.tcga import load_metadata_cache

                cached_metadata = load_metadata_cache(file_uuid)
                if cached_metadata is not None:
                    progress(0.9, desc="Loading cached metadata...")
                else:
                    progress(0.9, desc="Extracting metadata from TCGA...")

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
        # Save uploaded slides to user storage (HF Spaces only)
        input_slides.upload(
            handle_slide_upload,
            inputs=[input_slides, slide_ids_state],
            outputs=[input_slides, slide_ids_state],
        ).then(
            # Refresh storage usage after upload
            load_storage_usage,
            inputs=None,
            outputs=[storage_usage_warning, existing_file_row],
        ).then(
            # Refresh existing slides list after upload
            load_existing_slides,
            inputs=None,
            outputs=[existing_slides_dropdown],
        )

        # Refresh existing slides dropdown
        refresh_existing_btn.click(
            load_existing_slides,
            inputs=None,
            outputs=[existing_slides_dropdown],
        )

        # Handle selection of existing slide(s) for re-analysis
        def select_existing_slide(
            slide_ids,
            request: gr.Request = None,
            profile: Optional[gr.OAuthProfile] = None,
        ):
            """When user selects existing slide(s), load them for analysis.

            Args:
                slide_ids: Single slide_id (str) or list of slide_ids (multiselect)
                request: Gradio request object
                profile: OAuthProfile for HF Spaces login

            Returns:
                gr.File update with selected slide(s)
            """
            if not slide_ids:
                return None

            # Handle both single selection and multiselect
            if isinstance(slide_ids, str):
                slide_ids = [slide_ids]

            from mosaic.ui.user_tabs import _get_username

            username, _ = _get_username(request, profile)
            if not username:
                return None

            try:
                from mosaic.user_storage import get_slide_path, _load_slide_metadata

                metadata = _load_slide_metadata(username)
                slide_paths = []

                for slide_id in slide_ids:
                    slide_path = get_slide_path(username, slide_id)
                    if slide_path is None:
                        logger.warning(
                            f"Slide {slide_id} not found for user {username}"
                        )
                        continue

                    # Create a symlink with the original filename so the File
                    # component displays the real name instead of the UUID storage name
                    if slide_id in metadata:
                        original_name = metadata[slide_id].original_filename
                        if slide_path.name != original_name:
                            tmp_dir = Path(tempfile.mkdtemp(prefix="mosaic_slide_"))
                            symlink_path = tmp_dir / original_name
                            symlink_path.symlink_to(slide_path)
                            slide_paths.append(str(symlink_path))
                        else:
                            slide_paths.append(str(slide_path))
                    else:
                        slide_paths.append(str(slide_path))

                if not slide_paths:
                    return None

                return gr.File(value=slide_paths)

            except Exception as e:
                logger.error(f"Failed to load existing slides {slide_ids}: {e}")
                return None

        existing_slides_dropdown.change(
            select_existing_slide,
            inputs=[existing_slides_dropdown],
            outputs=[input_slides],
        )

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
            slide_ids,
            progress=gr.Progress(track_tqdm=True),
            request: gr.Request = None,
            profile: Optional[gr.OAuthProfile] = None,
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
                slide_ids=slide_ids,
                progress=progress,
                request=request,
                profile=profile,
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
                slide_ids_state,
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

        # Load storage usage on page load
        demo.load(
            load_storage_usage,
            inputs=None,
            outputs=[storage_usage_warning, existing_file_row],
        )

        # Load existing slides dropdown on page load
        demo.load(
            load_existing_slides,
            inputs=None,
            outputs=[existing_slides_dropdown],
        )

        # Hidden textbox that carries the localStorage session ID into
        # every Gradio event handler that needs the visitor's identity.
        # On HF Spaces the js= parameter on demo.load() fills this from
        # localStorage; locally it stays empty and _get_username() falls
        # back to LOCAL_DEBUG_USERNAME.
        session_id_input = gr.Textbox(visible=False, elem_id="session_id_input")

        # Add My Files and My Results tabs for user storage
        # On HF Spaces: visible for logged-in users
        # Locally: always visible with "local_user" for debugging
        gr.Markdown("---")
        if IS_HF_SPACES:
            user_storage_header = gr.Markdown(
                "## User Storage (Logged-In Users Only)", visible=False
            )
            user_storage_info = gr.Markdown(
                "The tabs below are only accessible to logged-in users. "
                "Upload and view your slides and analysis results.",
                visible=False,
            )
        else:
            gr.Markdown("## User Storage (Local Debug Mode)")
            gr.Markdown(
                "🔧 **Debug Mode**: Files stored as user 'local_user' in `/tmp/mosaic_user_data/local_user/`"
            )

        _LS_JS = "() => [localStorage.getItem('mosaic_session_id') || '']"

        # User storage tabs - initially hidden on HF Spaces, visible locally
        with gr.Tabs(visible=not IS_HF_SPACES) as user_storage_tabs:
            with gr.Tab("My Files"):
                from mosaic.ui.user_tabs import create_my_files_tab

                my_files_components = create_my_files_tab(session_id_input)

            with gr.Tab("My Results"):
                from mosaic.ui.user_tabs import create_my_results_tab

                my_results_components = create_my_results_tab(session_id_input)

        # On HF Spaces: a single demo.load() reads the session ID from
        # localStorage via JS and does ALL page-load work in one Python
        # call — check login, load files, load results.  This avoids
        # .then() chaining issues where intermediate component updates
        # may not propagate to the frontend before the next handler reads
        # them as inputs.
        if IS_HF_SPACES:
            _load_files = my_files_components["load_files"]
            _load_results = my_results_components["load_results"]

            # Number of output components from on_page_load:
            # login(5) + session_id(1) + files(4) + results(3) = 13
            _N_PAGE_LOAD_OUTPUTS = 13

            def on_page_load(session_id_from_browser):
                """Handle page load: check login, load files, load results.

                Gradio fires demo.load() both on initial page load AND on
                WebSocket reconnections.  The ``js=`` parameter only runs
                on the initial page load; reconnection events pass the
                *current component value* (empty string for a Textbox with
                no value set).  If we blindly apply the logged-out state
                on reconnection, it overwrites a successful login.

                Fix: when session_id is empty, return ``gr.update()`` for
                every output (no-op).  The default UI is already set to
                the logged-out state, so this is safe on the very first
                load too.
                """
                from mosaic.ui.oauth import get_user_by_session_id, _sessions

                sid_repr = (
                    session_id_from_browser[:8] + "..."
                    if session_id_from_browser
                    else "(empty)"
                )
                n_sessions = len(_sessions)
                logger.info(
                    f"on_page_load: sid={sid_repr}, "
                    f"active_sessions={n_sessions}"
                )

                # Empty session_id means either:
                #  a) user is genuinely not logged in (initial load), OR
                #  b) Gradio reconnection where js= didn't re-execute.
                # In both cases, return no-ops so we don't overwrite a
                # potentially logged-in state.  The default UI already
                # shows the sign-in prompt.
                if not session_id_from_browser:
                    logger.debug("on_page_load: empty sid, returning no-ops")
                    return tuple(gr.update() for _ in range(_N_PAGE_LOAD_OUTPUTS))

                userinfo = get_user_by_session_id(session_id_from_browser)

                if userinfo is not None:
                    username = userinfo.get("preferred_username", "User")
                    login_out = (
                        gr.update(
                            value=f"Signed in as **{username}**",
                            visible=True,
                        ),
                        gr.update(visible=True),   # logout_button
                        gr.update(visible=True),   # user_storage_tabs
                        gr.update(visible=True),   # user_storage_header
                        gr.update(visible=True),   # user_storage_info
                    )
                else:
                    # Session_id present but not found server-side
                    # (expired or server restarted). Show sign-in prompt.
                    login_out = (
                        gr.update(
                            value="[Sign in with HuggingFace](/api/auth/login) to save slides and results",
                            visible=True,
                        ),
                        gr.update(visible=False),  # logout_button
                        gr.update(visible=False),  # user_storage_tabs
                        gr.update(visible=False),  # user_storage_header
                        gr.update(visible=False),  # user_storage_info
                    )

                # Call load functions directly with session_id kwarg.
                # On HF Spaces _get_username only uses session_id (ignores
                # request/profile which are None when called outside Gradio
                # event dispatch).
                try:
                    files_out = _load_files(session_id=session_id_from_browser)
                except Exception as e:
                    logger.error(f"on_page_load: _load_files error: {e}")
                    files_out = (
                        f"**Storage:** Error: {e}",
                        [],
                        gr.Textbox(visible=False),
                        gr.File(visible=False),
                    )
                try:
                    results_out = _load_results(
                        session_id=session_id_from_browser
                    )
                except Exception as e:
                    logger.error(f"on_page_load: _load_results error: {e}")
                    results_out = (
                        [],
                        gr.Textbox(visible=False),
                        gr.File(visible=False),
                    )

                # Return: login(5) + session_id(1) + files(4) + results(3)
                return (
                    *login_out,
                    session_id_from_browser,
                    *files_out,
                    *results_out,
                )

            demo.load(
                on_page_load,
                inputs=[session_id_input],
                outputs=[
                    # login state (5)
                    login_status_md,
                    logout_button,
                    user_storage_tabs,
                    user_storage_header,
                    user_storage_info,
                    # persist session_id for subsequent button clicks (1)
                    session_id_input,
                    # files tab (4)
                    my_files_components["storage_usage"],
                    my_files_components["files_table"],
                    my_files_components["file_action_status"],
                    my_files_components["slide_download_file"],
                    # results tab (3)
                    my_results_components["results_table"],
                    my_results_components["result_action_status"],
                    my_results_components["result_download_file"],
                ],
                js=_LS_JS,
            )

            # Sign-out: pure JS — clear localStorage and reload.
            # On reload demo.load() fires, reads empty localStorage,
            # and on_page_load renders the logged-out state.
            logout_button.click(
                fn=None,
                inputs=None,
                js="() => { localStorage.removeItem('mosaic_session_id'); location.reload(); }",
            )

    # Use hardware-specific concurrency limit
    # T4 GPUs (16GB) can only handle one analysis at a time to prevent OOM
    # Higher-memory GPUs and ZeroGPU can handle multiple concurrent analyses
    demo.queue(max_size=10, default_concurrency_limit=DEFAULT_CONCURRENCY_LIMIT)

    # OAuth routes for HF Spaces (sdk:docker):
    # Custom OAuth routes (/api/auth/login, /api/auth/callback, /api/auth/logout)
    # are mounted via the lifespan protocol in create_oauth_lifespan(). This
    # ensures routes survive launch() recreating the app. Identity is read from
    # our server-side session (mosaic_auth cookie), not gr.OAuthProfile, which
    # returns the Space owner's identity on Docker SDK Spaces.

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

    # On HF Spaces, mount custom OAuth routes via the lifespan protocol.
    # Gradio's launch() recreates the app, but the lifespan handler receives
    # the final App instance, so routes added there survive.
    launch_kwargs = {}
    if IS_HF_SPACES:
        from mosaic.ui.oauth import create_oauth_lifespan

        launch_kwargs["app_kwargs"] = {"lifespan": create_oauth_lifespan}

    demo.launch(
        server_name=server_name,
        share=share,
        server_port=server_port,
        show_error=True,
        favicon_path=current_dir / "favicon.svg",
        **launch_kwargs,
    )
