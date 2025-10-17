from argparse import ArgumentParser
import gradio as gr
import pandas as pd
import pickle
from mussel.models import ModelType
from mussel.utils import get_features, segment_tissue, filter_features
from mussel.utils.segment import draw_slide_mask
from mussel.cli.tessellate import BiopsySegConfig, ResectionSegConfig, TcgaSegConfig
import torch
from pathlib import Path
from huggingface_hub import snapshot_download
import tempfile
import requests

from mosaic.inference import run_aeon, run_paladin
from loguru import logger

current_dir = Path(__file__).parent

# This path should be outside your project directory if running locally
TEMP_USER_DATA_DIR = Path(tempfile.gettempdir()) / "mosaic_user_data"

IHC_SUBTYPES = ["", "HR+/HER2+", "HR+/HER2-", "HR-/HER2+", "HR-/HER2-"]

SETTINGS_COLUMNS = [
    "Slide",
    "Site Type",
    "Cancer Subtype",
    "IHC Subtype",
    "Segmentation Config",
]

oncotree_code_map = {}


def get_oncotree_code_name(code):
    global oncotree_code_map
    if code in oncotree_code_map.keys():
        return oncotree_code_map[code]

    url = f"https://oncotree.mskcc.org/api/tumorTypes/search/code/{code}?exactMatch=true&version=oncotree_2025_04_08"
    response = requests.get(url)
    code_name = "Unknown"
    if response.status_code == 200:
        data = response.json()
        if data:
            code_name = data[0]["name"]

    oncotree_code_map[code] = code_name
    return code_name


def download_and_process_models():
    global cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes
    snapshot_download(repo_id="PDM-Group/paladin-aeon-models", local_dir="data")

    model_map = pd.read_csv(
        "data/paladin_model_map.csv",
    )
    cancer_subtypes = model_map["cancer_subtype"].unique().tolist()
    cancer_subtype_name_map = {
        f"{get_oncotree_code_name(code)} ({code})": code for code in cancer_subtypes
    }
    cancer_subtype_name_map["Unknown"] = "UNK"
    reversed_cancer_subtype_name_map = {
        value: key for key, value in cancer_subtype_name_map.items()
    }


def create_user_directory(state, request: gr.Request):
    """Create a unique directory for each user session."""
    session_hash = request.session_hash
    if session_hash is None:
        return None, None

    user_dir = TEMP_USER_DATA_DIR / session_hash
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def load_settings(slide_csv_path):
    """Load settings from CSV file and validate columns."""
    settings_df = pd.read_csv(slide_csv_path, na_filter=False)
    if "Segmentation Config" not in settings_df.columns:
        settings_df["Segmentation Config"] = "Biopsy"
    if "Cancer Subtype" not in settings_df.columns:
        settings_df["Cancer Subtype"] = "Unknown"
    if "IHC Subtype" not in settings_df.columns:
        settings_df["IHC Subtype"] = ""
    if not set(SETTINGS_COLUMNS).issubset(settings_df.columns):
        raise ValueError("Missing required column in CSV file")
    settings_df = settings_df[SETTINGS_COLUMNS]
    return settings_df


def validate_settings(settings_df):
    """Validate settings DataFrame and provide warnings for invalid entries."""
    settings_df.columns = SETTINGS_COLUMNS
    warnings = []
    for idx, row in settings_df.iterrows():
        slide_name = row["Slide"]
        subtype = row["Cancer Subtype"]
        if subtype in cancer_subtypes:
            settings_df.at[idx, "Cancer Subtype"] = reversed_cancer_subtype_name_map[
                subtype
            ]
        if settings_df.at[idx, "Cancer Subtype"] not in cancer_subtype_name_map.keys():
            warnings.append(
                f"Slide {slide_name}: Unknown cancer subtype. Valid subtypes are: {', '.join(cancer_subtype_name_map.keys())}. "
            )
            settings_df.at[idx, "Cancer Subtype"] = "Unknown"
        if row["Site Type"] not in ["Metastatic", "Primary"]:
            warnings.append(
                f"Slide {slide_name}: Unknown site type. Valid types are: Metastatic, Primary. "
            )
            settings_df.at[idx, "Site Type"] = "Primary"
        if (
            "Breast" not in settings_df.at[idx, "Cancer Subtype"]
            and row["IHC Subtype"] != ""
        ):
            warnings.append(
                f"Slide {slide_name}: IHC subtype should be empty for non-breast cancer subtypes. "
            )
            settings_df.at[idx, "IHC Subtype"] = ""
        if row["IHC Subtype"] not in IHC_SUBTYPES:
            warnings.append(
                f"Slide {slide_name}: Unknown IHC subtype. Valid subtypes are: {', '.join(IHC_SUBTYPES)}. "
            )
            settings_df.at[idx, "IHC Subtype"] = ""
        if row["Segmentation Config"] not in ["Biopsy", "Resection", "TCGA"]:
            warnings.append(
                f"Slide {slide_name}: Unknown segmentation config. Valid configs are: Biopsy, Resection, TCGA. "
            )
            settings_df.at[idx, "Segmentation Config"] = "Biopsy"

    if warnings:
        gr.Warning("\n".join(warnings))

    return settings_df


def export_to_csv(df):
    if df is None or df.empty:
        raise gr.Error("No data to export.")
    csv_path = "paladin_results.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


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
    settings_input = validate_settings(settings_input)
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
            row["IHC Subtype"],
            progress=progress,
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
        # yield slide_mask, None, None, None  # Yield intermediate results
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


def analyze_slide(
    slide_path,
    seg_config,
    site_type,
    cancer_subtype,
    ihc_subtype="",
    num_workers=4,
    progress=gr.Progress(track_tqdm=True),
):
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
            validate_settings, inputs=[settings_input], outputs=[settings_input]
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


def main():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--server-name", type=str, default="0.0.0.0", help="Server name for Gradio app"
    )
    parser.add_argument(
        "--server-port", type=int, default=None, help="Server port for Gradio app"
    )
    parser.add_argument(
        "--share", action="store_true", help="Share Gradio app publicly"
    )
    parser.add_argument(
        "--slide-csv",
        type=str,
        help="CSV file with slide settings (for batch processing), see README for format",
    )
    parser.add_argument(
        "--slide-path",
        type=str,
        help="Path to a single slide (for single slide processing), not used if --slide-csv is provided",
    )
    parser.add_argument(
        "--site-type",
        type=str,
        choices=["Primary", "Metastatic"],
        default="Primary",
        help="Site type of the slide (for single slide processing)",
    )
    parser.add_argument(
        "--cancer-subtype",
        type=str,
        default="Unknown",
        help="Cancer subtype of the slide (for single slide processing), use 'Unknown' to infer with Aeon",
    )
    parser.add_argument(
        "--ihc-subtype",
        type=str,
        choices=IHC_SUBTYPES,
        default="",
        help="IHC subtype if cancer subtype is breast (for single slide processing)",
    )
    parser.add_argument(
        "--segmentation-config",
        type=str,
        choices=["Biopsy", "Resection", "TCGA"],
        default="Biopsy",
        help="Segmentation configuration (for single slide processing)",
    )
    parser.add_argument(
        "--output-dir", type=str, help="Directory to save output results"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for feature extraction",
    )
    args = parser.parse_args()
    if args.debug:
        logger.add("debug.log", level="DEBUG")
        logger.debug("Debug logging enabled")

    download_and_process_models()

    if args.slide_path and not args.slide_csv:
        # Single slide processing mode
        if not args.output_dir:
            raise ValueError("Please provide --output-dir to save results")
        settings_df = pd.DataFrame(
            [
                [
                    args.slide_path,
                    args.site_type,
                    args.cancer_subtype,
                    args.ihc_subtype,
                    args.segmentation_config,
                ]
            ],
            columns=SETTINGS_COLUMNS,
        )
        settings_df = validate_settings(settings_df)
        slide_mask, aeon_results, paladin_results = analyze_slide(
            args.slide_path,
            args.segmentation_config,
            args.site_type,
            args.cancer_subtype,
            args.ihc_subtype,
            num_workers=args.num_workers,
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        slide_name = Path(args.slide_path).stem
        if slide_mask is not None:
            mask_path = output_dir / f"{slide_name}_mask.png"
            slide_mask.save(mask_path)
            logger.info(f"Saved slide mask to {mask_path}")
        if aeon_results is not None:
            aeon_output_path = output_dir / f"{slide_name}_aeon_results.csv"
            aeon_results.reset_index().to_csv(aeon_output_path, index=False)
            logger.info(f"Saved Aeon results to {aeon_output_path}")
        if paladin_results is not None and len(paladin_results) > 0:
            paladin_output_path = output_dir / f"{slide_name}_paladin_results.csv"
            paladin_results.to_csv(paladin_output_path, index=False)
            logger.info(f"Saved Paladin results to {paladin_output_path}")
    elif args.slide_csv:
        if not args.output_dir:
            raise ValueError("Please provide --output-dir to save results")
        # Batch processing mode

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        all_paladin_results = []
        all_aeon_results = []
        settings_df = load_settings(args.slide_csv)
        settings_df = validate_settings(settings_df)
        for idx, row in settings_df.iterrows():
            slide_path = row["Slide"]
            seg_config = row["Segmentation Config"]
            site_type = row["Site Type"]
            cancer_subtype = row["Cancer Subtype"]
            ihc_subtype = row.get("IHC Subtype", "")
            logger.info(
                f"Processing slide {slide_path} ({idx + 1} of {len(settings_df)})"
            )
            slide_mask, aeon_results, paladin_results = analyze_slide(
                slide_path,
                seg_config,
                site_type,
                cancer_subtype,
                ihc_subtype,
                num_workers=args.num_workers,
            )
            slide_name = Path(slide_path).stem
            mask_path = output_dir / f"{slide_name}_mask.png"
            slide_mask.save(mask_path)
            logger.info(f"Saved slide mask to {mask_path}")
            if aeon_results is not None:
                aeon_output_path = output_dir / f"{slide_name}_aeon_results.csv"
                aeon_results.reset_index().to_csv(aeon_output_path, index=False)
                logger.info(f"Saved Aeon results to {aeon_output_path}")
            if paladin_results is not None and len(paladin_results) > 0:
                paladin_output_path = output_dir / f"{slide_name}_paladin_results.csv"
                paladin_results.to_csv(paladin_output_path, index=False)
                logger.info(f"Saved Paladin results to {paladin_output_path}")
            if aeon_results is not None:
                aeon_results.columns = [f"{slide_name}"]
                all_aeon_results.append(aeon_results)
            if paladin_results is not None and len(paladin_results) > 0:
                paladin_results.insert(
                    0, "Slide", pd.Series([slide_name] * len(paladin_results))
                )
                all_paladin_results.append(paladin_results)
        if all_aeon_results:
            combined_aeon_results = pd.concat(all_aeon_results, axis=1)
            combined_aeon_results.reset_index(inplace=True)
            cancer_subtype_names = [
                f"{get_oncotree_code_name(code)} ({code})"
                for code in combined_aeon_results["Cancer Subtype"]
            ]
            combined_aeon_results["Cancer Subtype"] = cancer_subtype_names
            combined_aeon_output_path = output_dir / "combined_aeon_results.csv"
            combined_aeon_results.to_csv(combined_aeon_output_path, index=False)
            logger.info(f"Saved combined Aeon results to {combined_aeon_output_path}")
        if all_paladin_results:
            combined_paladin_results = pd.concat(all_paladin_results, ignore_index=True)
            cancer_subtype_names = [
                f"{get_oncotree_code_name(code)} ({code})"
                for code in combined_paladin_results["Cancer Subtype"]
            ]
            combined_paladin_results["Cancer Subtype"] = cancer_subtype_names
            combined_paladin_output_path = output_dir / "combined_paladin_results.csv"
            combined_paladin_results.to_csv(combined_paladin_output_path, index=False)
            logger.info(
                f"Saved combined Paladin results to {combined_paladin_output_path}"
            )
    else:
        launch_gradio(
            server_name=args.server_name,
            server_port=args.server_port,
            share=args.share,
        )


if __name__ == "__main__":
    main()
