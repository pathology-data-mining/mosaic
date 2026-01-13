"""Mosaic command-line interface and entry point.

This module provides the main CLI for the Mosaic application, handling:
- Model downloading and initialization
- Single slide processing
- Batch slide processing from CSV
- Launching the Gradio web interface
"""

from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
from huggingface_hub import snapshot_download
from loguru import logger

from mosaic.data_directory import set_data_directory
from mosaic.ui import launch_gradio
from mosaic.ui.app import set_cancer_subtype_maps
from mosaic.ui.utils import (
    get_oncotree_code_name,
    load_settings,
    validate_settings,
    IHC_SUBTYPES,
    SETTINGS_COLUMNS,
    SEX_OPTIONS,
)
from mosaic.analysis import analyze_slide
from mosaic.model_manager import load_all_models


def download_and_process_models():
    """Download essential models from HuggingFace and initialize cancer subtype mappings.

    Downloads only the core models (CTransPath, Optimus, Aeon, marker classifier) and
    metadata files from the PDM-Group HuggingFace repository. Paladin models are
    downloaded on-demand when needed for inference.

    Returns:
        tuple: (cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes)
            - cancer_subtype_name_map: Dict mapping display names to OncoTree codes
            - reversed_cancer_subtype_name_map: Dict mapping OncoTree codes to display names
            - cancer_subtypes: List of all supported cancer subtype codes
    """
    # Download only essential files to HF cache directory
    # Paladin models will be downloaded on-demand
    logger.info(
        "Downloading essential models from HuggingFace Hub (Paladin models loaded on-demand)..."
    )
    cache_dir = snapshot_download(
        repo_id="PDM-Group/paladin-aeon-models",
        allow_patterns=[
            "*.csv",  # Model maps and metadata
            "ctranspath.pth",  # CTransPath model
            "aeon_model.pkl",  # Aeon model
            "marker_classifier.pkl",  # Marker classifier
            "tissue_site_*",  # Tissue site mappings
            "metadata/*",  # Metadata files (including target_dict.tsv)
        ],
        # No local_dir - use HF cache
    )
    logger.info(f"Essential models downloaded to: {cache_dir}")

    # Set the data directory for other modules to use
    set_data_directory(cache_dir)

    # Pre-download Optimus model from bioptimus/H-optimus-0
    # This ensures it's cached at startup since it's needed for every slide
    logger.info("Pre-downloading Optimus model from bioptimus/H-optimus-0...")
    logger.info("This may take several minutes on first run - downloading ~1GB model...")
    from mussel.models import ModelType, get_model_factory

    optimus_factory = get_model_factory(ModelType.OPTIMUS)
    # This will trigger the download and cache the model
    _ = optimus_factory.get_model(
        model_path="hf-hub:bioptimus/H-optimus-0",
        use_gpu=False,  # Just download, don't load to GPU yet
        gpu_device_id=None,
    )
    logger.info("✓ Optimus model cached successfully")

    model_map = pd.read_csv(
        Path(cache_dir) / "paladin_model_map.csv",
    )
    cancer_subtypes = model_map["cancer_subtype"].unique().tolist()
    cancer_subtype_name_map = {"Unknown": "UNK"}
    cancer_subtype_name_map.update(
        {f"{get_oncotree_code_name(code)} ({code})": code for code in cancer_subtypes}
    )
    reversed_cancer_subtype_name_map = {
        value: key for key, value in cancer_subtype_name_map.items()
    }

    # Set the global maps in the UI module
    set_cancer_subtype_maps(
        cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes
    )

    return cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes


def main():
    """Main entry point for the Mosaic application.

    Parses command-line arguments and routes to the appropriate mode:
    - Single slide processing (--slide-path)
    - Batch processing (--slide-csv)
    - Web interface (default, no slide arguments)

    Command-line arguments control analysis parameters like site type,
    cancer subtype, segmentation configuration, and output directory.
    """
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
        "--sex",
        type=str,
        choices=SEX_OPTIONS,
        default=None,
        help="Sex of the patient (required for single slide processing)",
    )
    parser.add_argument(
        "--tissue-site",
        type=str,
        default="Unknown",
        help="Tissue site of the slide (for single slide processing)",
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
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Skip downloading models from HuggingFace (assumes models are already cached)",
    )
    parser.add_argument(
        "--download-models-only",
        action="store_true",
        help="Download models from HuggingFace and exit without running analysis",
    )
    args = parser.parse_args()
    if args.debug:
        logger.add("debug.log", level="DEBUG")
        logger.debug("Debug logging enabled")

    # Handle model download options
    if getattr(args, "download_models_only", False):
        logger.info("Downloading models from HuggingFace...")
        download_and_process_models()
        logger.info("✓ Model download complete. Exiting.")
        return

    if not getattr(args, "skip_model_download", False):
        cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes = (
            download_and_process_models()
        )
    else:
        logger.info("Skipping model download, using cached models...")
        # Load cancer subtype mappings from cached data
        from mosaic.data_directory import get_data_directory

        cache_dir = get_data_directory()
        model_map = pd.read_csv(Path(cache_dir) / "paladin_model_map.csv")
        cancer_subtypes = model_map["cancer_subtype"].unique().tolist()
        cancer_subtype_name_map = {"Unknown": "UNK"}
        cancer_subtype_name_map.update(
            {
                f"{get_oncotree_code_name(code)} ({code})": code
                for code in cancer_subtypes
            }
        )
        reversed_cancer_subtype_name_map = {
            value: key for key, value in cancer_subtype_name_map.items()
        }
        # Set the global maps in the UI module
        set_cancer_subtype_maps(
            cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes
        )

    if args.slide_path and not args.slide_csv:
        # Single slide processing mode
        if not args.output_dir:
            raise ValueError("Please provide --output-dir to save results")
        if not args.sex:
            raise ValueError("Please provide --sex (Male or Female) for single slide processing")
        settings_df = pd.DataFrame(
            [
                [
                    args.slide_path,
                    args.site_type,
                    args.sex,
                    args.tissue_site,
                    args.cancer_subtype,
                    args.ihc_subtype,
                    args.segmentation_config,
                ]
            ],
            columns=SETTINGS_COLUMNS,
        )
        settings_df = validate_settings(
            settings_df,
            cancer_subtype_name_map,
            cancer_subtypes,
            reversed_cancer_subtype_name_map,
        )
        slide_mask, aeon_results, paladin_results = analyze_slide(
            args.slide_path,
            args.segmentation_config,
            args.site_type,
            args.sex,
            args.tissue_site,
            args.cancer_subtype,
            cancer_subtype_name_map,
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
        # Batch processing mode with optimized model loading

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load and validate settings
        settings_df = load_settings(args.slide_csv)
        settings_df = validate_settings(
            settings_df,
            cancer_subtype_name_map,
            cancer_subtypes,
            reversed_cancer_subtype_name_map,
        )

        # Extract slide paths
        slides = settings_df["Slide"].tolist()

        logger.info(
            f"Processing {len(slides)} slides in batch mode with models loaded once"
        )

        # Load models once for batch processing
        model_cache = load_all_models(use_gpu=True, aggressive_memory_mgmt=None)

        all_slide_masks = []
        all_aeon_results = []
        all_paladin_results = []

        try:
            # Process each slide with pre-loaded models
            for idx, slide_path in enumerate(slides):
                row = settings_df.iloc[idx]
                slide_name = row["Slide"]

                logger.info(f"[{idx + 1}/{len(slides)}] Processing: {slide_name}")

                slide_mask, aeon_results, paladin_results = analyze_slide(
                    slide_path=slide_path,
                    seg_config=row["Segmentation Config"],
                    site_type=row["Site Type"],
                    sex=row["Sex"],
                    tissue_site=row.get("Tissue Site", "Unknown"),
                    cancer_subtype=row["Cancer Subtype"],
                    cancer_subtype_name_map=cancer_subtype_name_map,
                    ihc_subtype=row.get("IHC Subtype", ""),
                    num_workers=args.num_workers,
                    progress=lambda frac, desc: None,  # No-op progress for CLI
                    request=None,
                    model_cache=model_cache,
                )

                if slide_mask is not None:
                    all_slide_masks.append((slide_mask, slide_name))
                if aeon_results is not None:
                    all_aeon_results.append(aeon_results)
                if paladin_results is not None:
                    paladin_results.insert(
                        0, "Slide", pd.Series([slide_name] * len(paladin_results))
                    )
                    all_paladin_results.append(paladin_results)
        finally:
            logger.info("Cleaning up model cache")
            model_cache.cleanup()

        # Save individual slide results
        for idx, (slide_mask, slide_name) in enumerate(all_slide_masks):
            mask_path = output_dir / f"{slide_name}_mask.png"
            slide_mask.save(mask_path)
            logger.info(f"Saved slide mask to {mask_path}")

        for idx, aeon_results in enumerate(all_aeon_results):
            slide_name = aeon_results.columns[0]  # Slide name is in column name
            aeon_output_path = output_dir / f"{slide_name}_aeon_results.csv"
            aeon_results.reset_index().to_csv(aeon_output_path, index=False)
            logger.info(f"Saved Aeon results to {aeon_output_path}")

        # Group Paladin results by slide
        if all_paladin_results:
            combined_paladin = pd.concat(all_paladin_results, ignore_index=True)
            for slide_name in combined_paladin["Slide"].unique():
                slide_paladin = combined_paladin[
                    combined_paladin["Slide"] == slide_name
                ]
                paladin_output_path = output_dir / f"{slide_name}_paladin_results.csv"
                slide_paladin.to_csv(paladin_output_path, index=False)
                logger.info(f"Saved Paladin results to {paladin_output_path}")
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
