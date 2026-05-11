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
from mussel.utils.feature_extract import aggregate_sample_features
from PIL import Image

from mosaic.ui.utils import (
    expand_cancer_subtype_column,
    create_user_directory,
    validate_settings,
    SETTINGS_COLUMNS,
)
from mosaic.settings import row_to_slide_kwargs
from mosaic.analysis import (
    analyze_slide,
    extract_slide_features,
    run_inference_from_features,
)
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
            else:
                logger.exception(f"❌ Error during background model download: {e}")
            _core_models_complete = True  # Unblock pipeline even on error
        except Exception as e:
            logger.exception(f"❌ Error during background model download: {e}")
            _core_models_complete = True  # Unblock pipeline even on error

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
                    # ihc_subtype,  # Not yet in use
                    seg_config,
                    Path(slide_name).stem,  # Sample ID defaults to slide stem
                ]
            )
        settings_input = pd.DataFrame(settings, columns=SETTINGS_COLUMNS)

    validated = validate_settings(
        settings_input,
        cancer_subtype_name_map,
        cancer_subtypes,
        reversed_cancer_subtype_name_map,
    )
    # validate_settings returns gr.skip() when the DataFrame is unchanged
    # (to prevent the .change feedback loop in the UI).  In the pipeline we
    # want the original DataFrame in that case, not the sentinel object.
    if isinstance(validated, pd.DataFrame):
        settings_input = validated
    if len(slides) != len(settings_input):
        raise gr.Error("Missing settings for uploaded slides")

    # Check that all slides have sex specified
    if (
        settings_input["Sex"].isna().any()
        or (settings_input["Sex"] == "").any()
        or (settings_input["Sex"] == None).any()
    ):
        raise gr.Error("Sex is required for all slides. Please select Male or Female.")

    # Build sample groups: sample_id → list of (idx, slide_path, row)
    # When all sample IDs are unique, this is identical to the original per-slide loop.
    sample_groups: dict = {}
    for idx in range(len(slides)):
        row = settings_input.iloc[idx]
        sid = str(row.get("Sample ID", "")) or Path(str(row["Slide"])).stem
        sample_groups.setdefault(sid, []).append((idx, slides[idx], row))

    is_multi_slide = any(len(v) > 1 for v in sample_groups.values())
    if is_multi_slide:
        logger.info(
            f"Multi-slide sample mode: {len(slides)} slides → {len(sample_groups)} samples"
        )

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

    # Disable analyze and clear buttons during analysis.
    # Use gr.skip() for Dataframe outputs to avoid sending empty DataframeData
    # to the browser — even an empty DataframeData update can trigger TanStack
    # Table type-inference reactivity before the component is visible.
    yield (
        gr.skip(),  # settings_input (unchanged)
        [],  # slide_masks (empty initially)
        gr.skip(),  # aeon_output_table (unchanged)
        gr.skip(),  # aeon_note (unchanged)
        gr.DownloadButton(visible=False),  # aeon_download_button (hidden)
        gr.skip(),  # paladin_output_table (unchanged)
        gr.skip(),  # paladin_note (unchanged)
        gr.DownloadButton(visible=False),  # paladin_download_button (hidden)
        user_dir,  # user_dir_state (unchanged)
        gr.update(interactive=False),  # Disable analyze button
        gr.update(interactive=False),  # Disable clear button
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
        # ---------------------------------------------------------------
        # Process samples. For single-slide samples (the common case) the
        # path is identical to the original per-slide loop.  For multi-slide
        # samples we run a 2-phase pipeline:
        #   Phase 1 – extract features from every slide in the sample.
        #   Phase 2 – aggregate with aggregate_sample_features() then infer.
        # ---------------------------------------------------------------

        total_samples = len(sample_groups)
        sample_idx = 0

        for sample_id, slide_list in sample_groups.items():
            # ---- progress label helper ----
            class SlideProgressWrapper:
                def __init__(self, base_progress, s_idx, total, label):
                    self.base_progress = base_progress
                    self.s_idx = s_idx
                    self.total = total
                    self.label = _truncate_name(label)

                def __call__(self, current_progress=None, desc=None):
                    prefix = f"Sample {self.s_idx + 1}/{self.total}: {self.label}"
                    full_desc = f"{prefix} - {desc}" if desc else prefix
                    if current_progress is not None:
                        start = self.s_idx / self.total
                        end = (self.s_idx + 1) / self.total
                        self.base_progress(
                            start + current_progress * (end - start), desc=full_desc
                        )
                    else:
                        self.base_progress(self.s_idx / self.total, desc=full_desc)

                @property
                def track_tqdm(self):
                    return True

                def __getattr__(self, name):
                    return getattr(self.base_progress, name)

            sample_progress = SlideProgressWrapper(
                progress, sample_idx, total_samples, sample_id
            )
            sample_progress()

            first_row = slide_list[0][2]

            if len(slide_list) == 1:
                # ---- Single-slide sample: original path (with TCGA cache) ----
                idx, slide_path, row = slide_list[0]
                slide_name = row["Slide"]
                logger.info(
                    f"[sample {sample_idx + 1}/{total_samples}] {sample_id} — 1 slide"
                )

                slide_mask, aeon_results, paladin_results = None, None, None
                is_tcga_slide = False
                file_uuid = None

                try:
                    tcga_cache_dir = get_tcga_cache_directory()
                    slide_path_obj = Path(slide_path)
                    if tcga_cache_dir in slide_path_obj.parents:
                        file_uuid = slide_path_obj.parent.name
                        is_tcga_slide = True
                        settings_hash = compute_settings_hash(
                            **row_to_slide_kwargs(row)
                        )
                        cached = load_analysis_results(file_uuid, settings_hash)
                        if cached is not None:
                            slide_mask, aeon_results, paladin_results = cached
                            cached_slides += 1
                            logger.info(
                                f"Using cached results for {file_uuid}/{settings_hash}"
                            )
                except Exception as e:
                    logger.warning(
                        f"Failed to check/load cached results for slide: {e}. "
                        "Proceeding with fresh analysis."
                    )

                if (
                    slide_mask is None
                    or aeon_results is None
                    or paladin_results is None
                ):
                    if model_cache is None:
                        if IS_T4_GPU:
                            progress(
                                0.0, desc="Loading models (one-time initialization)"
                            )
                            _global_model_cache = load_all_models(
                                use_gpu=True, aggressive_memory_mgmt=None
                            )
                            model_cache = _global_model_cache
                        elif len(slides) > 1:
                            progress(0.0, desc="Loading models for batch processing")
                            model_cache = load_all_models(
                                use_gpu=True, aggressive_memory_mgmt=None
                            )

                    slide_mask, aeon_results, paladin_results = analyze_slide(
                        slide_path=slide_path,
                        **row_to_slide_kwargs(row),
                        cancer_subtype_name_map=cancer_subtype_name_map,
                        num_workers=DEFAULT_NUM_WORKERS,
                        progress=sample_progress,
                        request=request,
                        model_cache=model_cache,
                        analysis_id=analysis_id,
                        session_hash=session_hash,
                        profile=profile,
                        session_id=session_id,
                    )

                    if is_tcga_slide and file_uuid is not None:
                        try:
                            settings_hash = compute_settings_hash(
                                **row_to_slide_kwargs(row)
                            )
                            save_analysis_results(
                                file_uuid=file_uuid,
                                settings_hash=settings_hash,
                                aeon_results=aeon_results,
                                paladin_results=paladin_results,
                                slide_mask=slide_mask,
                            )
                            try:
                                upload_results_to_hf(file_uuid)
                            except Exception as upload_error:
                                logger.warning(
                                    f"Failed to upload to HF for {file_uuid}: {upload_error}"
                                )
                        except Exception as e:
                            logger.warning(
                                f"Failed to save cached results for {file_uuid}: {e}"
                            )

                # Save per-user results
                should_save_results = False
                save_username = None
                if IS_HF_SPACES:
                    if user_info.is_logged_in and user_info.username:
                        should_save_results = True
                        save_username = user_info.username
                else:
                    from mosaic.ui.user_tabs import LOCAL_DEBUG_USERNAME

                    should_save_results = True
                    save_username = LOCAL_DEBUG_USERNAME

                if should_save_results and save_username:
                    try:
                        from mosaic.user_results import (
                            save_analysis_results as save_user_results,
                        )

                        settings_dict = {
                            "seg_config": row["Segmentation Config"],
                            "site_type": row["Site Type"],
                            "sex": row["Sex"],
                            "tissue_site": row.get("Tissue Site", "Unknown"),
                            "cancer_subtype": row["Cancer Subtype"],
                        }
                        slide_analysis_id = f"{analysis_id}_{idx}"
                        file_slide_id = (
                            slide_ids.get(slide_name, "") if slide_ids else ""
                        )
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
                    except Exception as e:
                        logger.warning(f"Failed to save user results (non-fatal): {e}")

                # Collect results
                if slide_mask is not None:
                    mask_filename = f"{Path(slide_name).stem}_mask.png"
                    saved = _save_mask_as_png(slide_mask, mask_filename)
                    all_slide_masks.append(
                        (saved if saved is not None else slide_mask, slide_name)
                    )
                if aeon_results is not None:
                    aeon_results = aeon_results.rename(
                        columns={"Confidence": sample_id}
                    )
                    all_aeon_results.append(aeon_results)
                if paladin_results is not None:
                    paladin_results.insert(
                        0, "Slide", pd.Series([sample_id] * len(paladin_results))
                    )
                    all_paladin_results.append(paladin_results)

            else:
                # ---- Multi-slide sample: 2-phase extraction + aggregation ----
                logger.info(
                    f"[sample {sample_idx + 1}/{total_samples}] {sample_id} — "
                    f"{len(slide_list)} slides"
                )

                # Ensure models are loaded
                if model_cache is None:
                    progress(0.0, desc="Loading models for multi-slide sample")
                    model_cache = load_all_models(
                        use_gpu=True, aggressive_memory_mgmt=None
                    )
                    if IS_T4_GPU:
                        _global_model_cache = model_cache

                # Phase 1: extract features from each slide
                per_slide_features = []
                per_slide_coords = []
                per_slide_masks = []
                per_sample_ids = []

                for slide_sub_idx, (idx, slide_path, row) in enumerate(slide_list):
                    slide_name = row["Slide"]
                    logger.info(
                        f"  [{slide_sub_idx + 1}/{len(slide_list)}] Extracting features: {slide_name}"
                    )

                    def _sub_progress(
                        frac=None,
                        desc=None,
                        *,
                        _base=sample_progress,
                        _sub=slide_sub_idx,
                        _total=len(slide_list),
                    ):
                        # Scale within the extraction half of this sample's slot
                        sub_frac = (_sub + (frac or 0)) / _total * 0.5
                        _base(sub_frac, desc=desc)

                    slide_mask, features, coords = extract_slide_features(
                        slide_path=slide_path,
                        seg_config=row["Segmentation Config"],
                        num_workers=DEFAULT_NUM_WORKERS,
                        progress=_sub_progress,
                        model_cache=model_cache,
                    )

                    if features is None:
                        logger.warning(
                            f"No tissue in {slide_name}, skipping for sample {sample_id}"
                        )
                        continue

                    per_slide_features.append(features)
                    per_slide_coords.append(coords)
                    per_sample_ids.append(sample_id)
                    if slide_mask is not None:
                        mask_filename = f"{Path(slide_name).stem}_mask.png"
                        saved = _save_mask_as_png(slide_mask, mask_filename)
                        per_slide_masks.append(
                            (saved if saved is not None else slide_mask, slide_name)
                        )

                if not per_slide_features:
                    logger.warning(
                        f"No usable slides for sample {sample_id}, skipping inference"
                    )
                    sample_idx += 1
                    continue

                # Aggregate per-slide features into one array per sample
                sample_progress(
                    0.5, desc=f"Aggregating {len(per_slide_features)} slides"
                )
                agg_dict = aggregate_sample_features(
                    per_slide_features, per_slide_coords, per_sample_ids
                )
                agg_features, _ = agg_dict[sample_id]
                logger.info(
                    f"Aggregated {sample_id}: {agg_features.shape[0]} tiles from "
                    f"{len(per_slide_features)} slides"
                )

                # Phase 2: run Aeon + Paladin on aggregated features
                from mosaic.inference.data import encode_sex, encode_tissue_site

                sex_idx_val = (
                    encode_sex(first_row["Sex"]) if first_row.get("Sex") else None
                )
                ts_idx_val = encode_tissue_site(first_row.get("Tissue Site", "Unknown"))

                aeon_results, paladin_results = run_inference_from_features(
                    features=agg_features,
                    site_type=first_row["Site Type"],
                    sex_idx=sex_idx_val,
                    tissue_site_idx=ts_idx_val,
                    cancer_subtype=first_row["Cancer Subtype"],
                    cancer_subtype_name_map=cancer_subtype_name_map,
                    model_cache=model_cache,
                    num_workers=DEFAULT_NUM_WORKERS,
                    progress=lambda frac=None, desc=None: sample_progress(
                        0.5 + (frac or 0) * 0.5, desc=desc
                    ),
                )

                # Collect results for this sample
                all_slide_masks.extend(per_slide_masks)
                if aeon_results is not None:
                    aeon_results = aeon_results.rename(
                        columns={"Confidence": sample_id}
                    )
                    all_aeon_results.append(aeon_results)
                if paladin_results is not None:
                    paladin_results.insert(
                        0, "Slide", pd.Series([sample_id] * len(paladin_results))
                    )
                    all_paladin_results.append(paladin_results)

            sample_idx += 1

            # Yield incremental UI update after each sample.
            # Use gr.skip() for Dataframe outputs to avoid triggering TanStack
            # type-inference on empty/stale data while analysis is still running.
            yield (
                gr.skip(),
                all_slide_masks.copy(),
                gr.skip(),
                gr.skip(),  # aeon_note
                gr.DownloadButton(visible=False),
                gr.skip(),
                gr.skip(),  # paladin_note
                gr.DownloadButton(visible=False),
                user_dir,
                gr.update(interactive=False),
                gr.update(interactive=False),
            )

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

        tracker.upload_to_hf()
        progress(0.99, desc="Analysis complete, wrapping up results")

    except Exception as e:
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
        # Re-enable buttons before re-raising so the UI doesn't stay locked
        yield (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            gr.skip(),  # aeon_note
            gr.skip(),
            gr.skip(),
            gr.skip(),  # paladin_note
            gr.skip(),
            gr.skip(),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
        raise

    finally:
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

    # Use gr.update() (not gr.DataFrame()) so only `value` and `visible` are sent
    # to the browser in the final yield.  Sending `datatype` simultaneously with
    # `data` triggers a circular Svelte 5 reactive dependency in TanStack Table
    # that freezes the browser renderer.  The component's `datatype` is fixed at
    # initialisation time in app.py and must never be changed mid-session.
    combined_aeon_results = gr.update(visible=False)
    aeon_note_update = gr.update(visible=False)
    aeon_output = gr.DownloadButton(visible=False)
    if all_aeon_results and not all_subtypes_known:
        combined_aeon_results = pd.concat(all_aeon_results, axis=1)
        combined_aeon_results.reset_index(inplace=True)

        combined_aeon_results = combined_aeon_results.round(3)
        expand_cancer_subtype_column(combined_aeon_results)

        aeon_output_path = user_dir / f"aeon_results-{timestamp}.csv"
        combined_aeon_results.to_csv(aeon_output_path)

        combined_aeon_results = gr.update(value=combined_aeon_results, visible=True)
        aeon_note_update = gr.update(visible=True)
        aeon_output = gr.DownloadButton(value=aeon_output_path, visible=True)

    # Convert Oncotree codes to names for display
    paladin_output = gr.DownloadButton(visible=False)
    paladin_note_update = gr.update(visible=False)
    if len(combined_paladin_results) > 0:
        expand_cancer_subtype_column(combined_paladin_results)

        # Ensure Score is numeric before rounding
        combined_paladin_results["Prob. Present"] = pd.to_numeric(
            combined_paladin_results["Prob. Present"], errors="coerce"
        )
        combined_paladin_results["Prob. Present"] = combined_paladin_results[
            "Prob. Present"
        ].round(3)

        paladin_output_path = user_dir / f"paladin_results-{timestamp}.csv"
        combined_paladin_results.to_csv(paladin_output_path, index=False)
        paladin_output = gr.DownloadButton(value=paladin_output_path, visible=True)
        paladin_note_update = gr.update(visible=True)

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
        gr.update(
            value=settings_input, visible=settings_visible
        ),  # Hide if single slide
        final_slide_masks,
        combined_aeon_results,
        aeon_note_update,
        aeon_output,
        gr.update(
            value=final_combined_paladin,
            visible=final_combined_paladin is not None,
        ),
        paladin_note_update,
        paladin_output,
        user_dir,
        gr.update(interactive=True),  # Re-enable analyze button
        gr.update(interactive=True),  # Re-enable clear button
    )
