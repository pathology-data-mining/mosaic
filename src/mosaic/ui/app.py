"""Gradio web interface for Mosaic.

This module provides the web-based user interface for analyzing whole slide images.
It includes functionality for:
- Multi-slide upload and analysis
- Settings configuration (site type, cancer subtype, IHC subtype, segmentation)
- Results visualization and export
- CSV-based batch processing
"""

import gradio as gr
import pandas as pd
from pathlib import Path
from loguru import logger

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
from mosaic.hardware import DEFAULT_CONCURRENCY_LIMIT, IS_T4_GPU

current_dir = Path(__file__).parent.parent

# Global variables for cancer subtypes (set by download_and_process_models)
cancer_subtype_name_map = {}
reversed_cancer_subtype_name_map = {}
cancer_subtypes = []

# Global model cache for T4 (to persist models across sequential requests)
_global_model_cache = None

# Background download thread for model initialization
_model_download_thread = None
_model_download_complete = False


def set_cancer_subtype_maps(csn_map, rcsn_map, cs):
    """Set the global cancer subtype maps."""
    global cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes
    cancer_subtype_name_map = csn_map
    reversed_cancer_subtype_name_map = rcsn_map
    cancer_subtypes = cs


def start_background_model_download():
    """Start downloading Paladin models in a background thread.

    Core models (CTransPath, Optimus, Aeon) are already downloaded synchronously.
    This downloads the remaining Paladin models asynchronously.
    """
    global _model_download_thread, _model_download_complete

    if _model_download_thread is not None:
        logger.info("Background model download already started, skipping")
        return

    def download_worker():
        try:
            from mosaic.gradio_app import download_paladin_models
            logger.info("🔄 Starting background download of Paladin models...")
            download_paladin_models()
            global _model_download_complete
            _model_download_complete = True
            logger.info("✅ Background Paladin model download complete")
        except Exception as e:
            logger.error(f"❌ Error during background Paladin model download: {e}")
            import traceback
            traceback.print_exc()

    import threading
    _model_download_thread = threading.Thread(target=download_worker, daemon=True)
    _model_download_thread.start()
    logger.info("🚀 Paladin model download started in background thread")


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
    global _model_download_thread, _model_download_complete

    # Wait for background model download to complete
    if _model_download_thread is not None and not _model_download_complete:
        logger.info("Waiting for model download to complete...")
        progress(0, desc="Waiting for models to download...")
        _model_download_thread.join()
        _model_download_complete = True
        logger.info("Model download complete, proceeding with analysis")

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
    if settings_input["Sex"].isna().any() or (settings_input["Sex"] == "").any() or (settings_input["Sex"] == None).any():
        raise gr.Error("Sex is required for all slides. Please select Male or Female.")

    all_slide_masks = []
    all_aeon_results = []
    all_paladin_results = []

    # Yield initial state to make settings table visible immediately
    yield (
        gr.Dataframe(value=settings_input, visible=True),  # Make settings visible
        [],  # Empty slide masks
        gr.DataFrame(visible=False),  # Hidden AEON table
        gr.DownloadButton(visible=False),  # Hidden AEON download
        None,  # No PALADIN results yet
        gr.DownloadButton(visible=False),  # Hidden PALADIN download
        user_dir,  # user_dir_state
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
            _global_model_cache = load_all_models(use_gpu=True, aggressive_memory_mgmt=None)
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

            logger.info(f"[{idx + 1}/{len(slides)}] Processing: {slide_name}")
            slide_progress = idx / len(slides)
            progress(slide_progress, desc=f"Analyzing slide {idx + 1}/{len(slides)}")

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
                progress=progress,
                request=request,
                model_cache=model_cache,  # Pre-loaded for batch, None for single
            )

            if slide_mask is not None:
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

            # Build partial AEON results for display
            partial_aeon_df = gr.DataFrame(visible=False)
            if all_aeon_results:
                partial_aeon = pd.concat(all_aeon_results, axis=1)
                partial_aeon.reset_index(inplace=True)
                partial_aeon = partial_aeon.round(3)

                # Convert OncoTree codes to names for display
                cancer_subtype_names = [
                    f"{get_oncotree_code_name(code)} ({code})"
                    for code in partial_aeon["Cancer Subtype"]
                ]
                partial_aeon["Cancer Subtype"] = cancer_subtype_names

                partial_aeon_df = gr.DataFrame(
                    partial_aeon,
                    visible=True,
                    column_widths=["4px"] + ["2px"] * (partial_aeon.shape[1] - 1),
                )

            # Build partial PALADIN results for display
            partial_paladin_df = None
            if all_paladin_results:
                partial_paladin = pd.concat(all_paladin_results, ignore_index=True)

                # Convert OncoTree codes to names for display
                cancer_subtype_names = [
                    f"{get_oncotree_code_name(code)} ({code})"
                    for code in partial_paladin["Cancer Subtype"]
                ]
                partial_paladin["Cancer Subtype"] = cancer_subtype_names

                # Ensure Score is numeric before rounding
                partial_paladin["Score"] = pd.to_numeric(partial_paladin["Score"], errors='coerce')
                partial_paladin["Score"] = partial_paladin["Score"].round(3)

                partial_paladin_df = partial_paladin

            # Yield intermediate update to show progressive results
            # Download buttons stay hidden until all slides are processed
            # Make settings visible during processing (for progress bar display)
            yield (
                gr.Dataframe(value=settings_input, visible=True),  # Settings visible for progress
                all_slide_masks.copy(),  # Current slide masks
                partial_aeon_df,  # Partial AEON results (growing)
                gr.DownloadButton(visible=False),  # Download button hidden until complete
                partial_paladin_df,  # Partial PALADIN results (growing)
                gr.DownloadButton(visible=False),  # Download button hidden until complete
                user_dir,  # user_dir_state
            )

    finally:
        # Clean up model cache if it was loaded for batch processing
        # On T4: Keep global cache loaded, only cleanup Paladin models
        # On high-memory GPUs: Cleanup everything after batch
        if model_cache is not None and not IS_T4_GPU:
            logger.info("Cleaning up model cache after batch")
            model_cache.cleanup()
        elif IS_T4_GPU and model_cache is not None:
            logger.info("T4: Keeping core models loaded, cleaning up Paladin models only")
            model_cache.cleanup_paladin()

    progress(0.99, desc="Analysis complete, wrapping up results")

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
        combined_paladin_results["Score"] = pd.to_numeric(combined_paladin_results["Score"], errors='coerce')
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
    final_combined_paladin = combined_paladin_results if len(combined_paladin_results) > 0 else None

    # Memory cleanup: Clear intermediate data structures from RAM
    import gc

    all_slide_masks = None
    all_aeon_results = None
    all_paladin_results = None
    combined_paladin_results = None

    # Force garbage collection to free Python memory
    gc.collect()
    
    yield (
        gr.Dataframe(value=settings_input, visible=settings_visible),  # Hide if single slide
        final_slide_masks,
        combined_aeon_results,
        aeon_output,
        final_combined_paladin,
        paladin_output,
        user_dir,
    )


def launch_gradio(server_name, server_port, share):
    # Start downloading models in background
    start_background_model_download()

    with gr.Blocks(title="Mosaic") as demo:
        user_dir_state = gr.State(None)
        gr.Markdown(
            "# Mosaic: H&E Whole Slide Image Cancer Subtype and Biomarker Inference"
        )
        gr.Markdown(
            "Upload an H&E whole slide image in SVS or TIFF format. The slide will be processed to infer cancer subtype and relevant biomarkers."
        )
        with gr.Row():
            with gr.Column():
                input_slides = gr.File(
                    label="Upload H&E Whole Slide Image",
                    file_types=[".svs", ".tiff", ".tif"],
                    file_count="multiple",
                )
                site_dropdown = gr.Dropdown(
                    choices=["Primary", "Metastatic"],
                    label="Site Type",
                    value="Primary",
                )
                sex_dropdown = gr.Dropdown(
                    choices=SEX_OPTIONS,
                    label="Sex",
                    value=None,
                )
                tissue_site_dropdown = gr.Dropdown(
                    choices=get_tissue_sites(),
                    label="Tissue Site",
                    value="Unknown",
                )
                cancer_subtype_dropdown = gr.Dropdown(
                    choices=[name for name in cancer_subtype_name_map.keys()],
                    label="Cancer Subtype",
                    value="Unknown",
                )
                ihc_subtype_dropdown = gr.Dropdown(
                    choices=IHC_SUBTYPES,
                    label="IHC Subtype (if applicable)",
                    value="",
                    visible=False,
                )
                seg_config_dropdown = gr.Dropdown(
                    choices=["Biopsy", "Resection", "TCGA"],
                    label="Segmentation Config",
                    value="Biopsy",
                )
                with gr.Row():
                    settings_input = gr.Dataframe(
                        headers=SETTINGS_COLUMNS,
                        label="Current Settings",
                        datatype=["str"] * len(SETTINGS_COLUMNS),
                        visible=False,
                        interactive=True,
                        static_columns="Slide",
                    )

                with gr.Row():
                    settings_csv = gr.File(
                        file_types=[".csv"], label="Upload Settings CSV", visible=False
                    )

                with gr.Row():
                    clear_button = gr.Button("Clear")
                    analyze_button = gr.Button("Analyze", variant="primary")
            with gr.Column():
                slide_masks = gr.Gallery(
                    label="Slide Masks",
                    columns=3,
                    object_fit="contain",
                    height="auto",
                )
                aeon_output_table = gr.Dataframe(
                    headers=["Cancer Subtype", "Slide Name"],
                    label="Cancer Subtype Inference Confidence",
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
                    label="Biomarker Inference",
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
            has_ihc = "Breast" in cancer_subtype
            if not files:
                return None, None, gr.Dropdown(visible=has_ihc)
            settings_df = get_settings(
                files,
                site_type,
                sex,
                tissue_site,
                cancer_subtype,
                ihc_subtype,
                seg_config,
            )
            if settings_df is not None:
                has_ihc = any("Breast" in cs for cs in settings_df["Cancer Subtype"])
            visible = files and len(files) > 1
            return (
                gr.Dataframe(value=settings_df, visible=visible),
                gr.File(visible=visible),
                gr.Dropdown(visible=has_ihc),
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
            has_ihc = "Breast" in cancer_subtype
            if settings_df is None or len(settings_df) == 0:
                return settings_df, gr.Dropdown(visible=has_ihc)
            updated_df = update_settings_column(
                settings_df, "Cancer Subtype", cancer_subtype
            )
            return gr.Dataframe(value=updated_df), gr.Dropdown(visible=has_ihc)

        @ihc_subtype_dropdown.change(
            inputs=[settings_input, ihc_subtype_dropdown],
            outputs=[settings_input],
        )
        def update_ihc_subtype(settings_df, ihc_subtype):
            """Update IHC Subtype column when dropdown changes."""
            if settings_df is None or len(settings_df) == 0:
                return settings_df
            updated_df = update_settings_column(settings_df, "IHC Subtype", ihc_subtype)
            return gr.Dataframe(value=updated_df)

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

        analyze_button.click(
            analyze_slides,
            inputs=[
                input_slides,
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
            ],
            queue=True,
            show_progress_on=settings_input,
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
    demo.launch(
        server_name=server_name,
        share=share,
        server_port=server_port,
        show_error=True,
        favicon_path=current_dir / "favicon.svg",
    )
