"""Analysis pipeline for Mosaic slide processing.

This module owns the global model/subtype state and the ``analyze_slides``
generator function that drives the Gradio event loop.  It is intentionally
separate from ``app.py`` so that the business-logic can be imported and tested
without loading the entire Gradio UI.

Public surface (re-exported by ``mosaic.ui.app`` for backward compatibility):
- ``analyze_slides`` — main generator used by the Gradio ``analyze_button``
- ``set_cancer_subtype_maps`` — called by ``gradio_app.py`` after model download
- ``cancer_subtype_name_map`` — module-level dict (read by ``launch_gradio``)
"""

import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd
from loguru import logger
from PIL import Image

from mosaic.ui.utils import (
    get_oncotree_code_name,
    create_user_directory,
    validate_settings,
    SETTINGS_COLUMNS,
)
from mosaic.analysis import analyze_slide
from mosaic.data_directory import get_tcga_cache_directory
from mosaic.model_manager import load_all_models
from mosaic.hardware import IS_HF_SPACES, IS_T4_GPU, GPU_TYPE, DEFAULT_NUM_WORKERS
from mosaic.telemetry import extract_user_info
from mosaic.tcga import (
    compute_settings_hash,
    load_analysis_results,
    save_analysis_results,
    upload_results_to_hf,
)


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
_core_models_complete = False  # Track core models separately
_shutdown_requested = False  # Flag to signal graceful shutdown


def set_cancer_subtype_maps(csn_map, rcsn_map, cs):
    """Set the global cancer subtype maps."""
    global cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes
    cancer_subtype_name_map = csn_map
    reversed_cancer_subtype_name_map = rcsn_map
    cancer_subtypes = cs


def request_shutdown():
    """Signal the background model download thread to stop gracefully."""
    global _shutdown_requested
    _shutdown_requested = True


def is_shutdown_requested() -> bool:
    """Return True if a shutdown has been requested."""
    return _shutdown_requested


def start_background_model_download():
    """Start downloading all models in a background thread.

    Downloads core models (CTransPath, Optimus, Aeon) and Paladin models asynchronously.
    This allows Gradio to start immediately and satisfy HF Spaces health checks.
    """
    global _model_download_thread, _core_models_complete

    if _model_download_thread is not None:
        logger.info("Background model download already started, skipping")
        return

    def download_worker():
        global _core_models_complete, _shutdown_requested
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
                logger.exception(f"❌ Error during background model download: {e}")
        except Exception as e:
            logger.exception(f"❌ Error during background model download: {e}")

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
    session_id: str = "",
    progress=gr.Progress(track_tqdm=True),
    request: gr.Request = None,
    profile: Optional[gr.OAuthProfile] = None,
):
    global _model_download_thread, _core_models_complete

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
    user_info = extract_user_info(request, IS_HF_SPACES, profile, session_id)

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
            slide_name = Path(filename).name
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
                    num_workers=DEFAULT_NUM_WORKERS,
                    progress=slide_progress_wrapper,  # Pass wrapped progress
                    request=request,
                    model_cache=model_cache,  # Pre-loaded for batch, None for single
                    analysis_id=analysis_id,  # For telemetry tracking
                    session_hash=session_hash,  # For telemetry tracking
                    profile=profile,  # For OAuth user info
                    session_id=session_id,  # For OAuth session lookup
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
