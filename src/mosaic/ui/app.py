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
    IHC_SUBTYPES,
    SETTINGS_COLUMNS,
)
from mosaic.analysis import analyze_slide

current_dir = Path(__file__).parent.parent

# Global variables for cancer subtypes (set by download_and_process_models)
cancer_subtype_name_map = {}
reversed_cancer_subtype_name_map = {}
cancer_subtypes = []


def set_cancer_subtype_maps(csn_map, rcsn_map, cs):
    """Set the global cancer subtype maps."""
    global cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes
    cancer_subtype_name_map = csn_map
    reversed_cancer_subtype_name_map = rcsn_map
    cancer_subtypes = cs


def analyze_slides(
    slides,
    settings_input,
    user_dir,
    progress=gr.Progress(track_tqdm=True),
):
    if slides is None or len(slides) == 0:
        raise gr.Error("Please upload at least one slide.")
    if user_dir is None:
        user_dir = create_user_directory(None, gr.Request())
    settings_input = validate_settings(
        settings_input, cancer_subtype_name_map, cancer_subtypes, reversed_cancer_subtype_name_map
    )
    if len(slides) != len(settings_input):
        raise gr.Error("Missing settings for uploaded slides")

    all_slide_masks = []
    all_aeon_results = []
    all_paladin_results = []

    progress(0.0, desc="Starting analysis")
    for idx, row in settings_input.iterrows():
        slide_name = row["Slide"]
        progress(
            idx / len(settings_input),
            desc=f"Analyzing {slide_name}, slide {idx + 1} of {len(settings_input)}",
        )
        for x in slides:
            s = x.split("/")[-1]
            if s == slide_name:
                slide_mask = x

        (
            slide_mask,
            aeon_results,
            paladin_results,
        ) = analyze_slide(
            slides[idx],
            row["Segmentation Config"],
            row["Site Type"],
            row["Cancer Subtype"],
            cancer_subtype_name_map,
            row["IHC Subtype"],
            progress=progress,
            request=request,
        )
        if aeon_results is not None:
            if len(slides) > 1:
                aeon_results.columns = [f"{slide_name}"]
            if row["Cancer Subtype"] == "Unknown":
                all_aeon_results.append(aeon_results)
        if paladin_results is not None:
            paladin_results.insert(
                0, "Slide", pd.Series([slide_name] * len(paladin_results))
            )
            all_paladin_results.append(paladin_results)
        if slide_mask is not None:
            all_slide_masks.append((slide_mask, slide_name))
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
    cancer_subtype_names = [
        f"{get_oncotree_code_name(code)} ({code})"
        for code in combined_paladin_results["Cancer Subtype"]
    ]
    combined_paladin_results["Cancer Subtype"] = cancer_subtype_names
    if len(combined_paladin_results) > 0:
        combined_paladin_results["Score"] = combined_paladin_results["Score"].round(3)

    paladin_output = gr.DownloadButton(visible=False)
    if len(combined_paladin_results) > 0:
        paladin_output_path = user_dir / f"paladin_results-{timestamp}.csv"
        combined_paladin_results.to_csv(paladin_output_path, index=False)
        paladin_output = gr.DownloadButton(value=paladin_output_path, visible=True)

    progress(1.0, desc="All done!")

    return (
        all_slide_masks,
        combined_aeon_results,
        aeon_output,
        combined_paladin_results if len(combined_paladin_results) > 0 else None,
        paladin_output,
        user_dir,
    )


def launch_gradio(server_name, server_port, share):
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
                        headers=[
                            "Slide",
                            "Site Type",
                            "Cancer Subtype",
                            "IHC Subtype",
                            "Segmentation Config",
                        ],
                        label="Current Settings",
                        datatype=["str", "str", "str", "str", "str"],
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
                None,
                None,
                None,
                None,
                gr.Dataframe(visible=False),
                gr.DownloadButton(visible=False),
                gr.Dataframe(visible=False),
                gr.File(visible=False),
            )

        def get_settings(files, site_type, cancer_subtype, ihc_subtype, seg_config):
            if files is None:
                return pd.DataFrame()
            settings = []
            for file in files:
                filename = file.name if hasattr(file, "name") else file
                slide_name = filename.split("/")[-1]
                settings.append(
                    [slide_name, site_type, cancer_subtype, ihc_subtype, seg_config]
                )
            df = pd.DataFrame(settings, columns=SETTINGS_COLUMNS)
            return df

        # Only display settings table and upload button if multiple slides are uploaded
        @gr.on(
            [
                input_slides.change,
                site_dropdown.change,
                cancer_subtype_dropdown.change,
                ihc_subtype_dropdown.change,
                seg_config_dropdown.change,
            ],
            inputs=[
                input_slides,
                site_dropdown,
                cancer_subtype_dropdown,
                ihc_subtype_dropdown,
                seg_config_dropdown,
            ],
            outputs=[settings_input, settings_csv, ihc_subtype_dropdown],
        )
        def update_settings(files, site_type, cancer_subtype, ihc_subtype, seg_config):
            has_ihc = "Breast" in cancer_subtype
            if not files:
                return None, None, gr.Dropdown(visible=has_ihc)
            settings_df = get_settings(
                files, site_type, cancer_subtype, ihc_subtype, seg_config
            )
            if settings_df is not None:
                has_ihc = any("Breast" in cs for cs in settings_df["Cancer Subtype"])
            visible = files and len(files) > 1
            return (
                gr.Dataframe(settings_df, visible=visible),
                gr.File(visible=visible),
                gr.Dropdown(visible=has_ihc),
            )

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
                user_dir_state,
            ],
            outputs=[
                slide_masks,
                aeon_output_table,
                aeon_download_button,
                paladin_output_table,
                paladin_download_button,
                user_dir_state,
            ],
            queue=True,
            show_progress_on=paladin_output_table,
        )
        settings_input.change(
            lambda df: validate_settings(df, cancer_subtype_name_map, cancer_subtypes, reversed_cancer_subtype_name_map),
            inputs=[settings_input],
            outputs=[settings_input]
        )
        demo.load(
            create_user_directory,
            inputs=[user_dir_state],
            outputs=[user_dir_state],
        )

    demo.queue(max_size=10, default_concurrency_limit=8)
    demo.launch(
        server_name=server_name,
        share=share,
        server_port=server_port,
        show_error=True,
        favicon_path=current_dir / "favicon.svg",
    )
