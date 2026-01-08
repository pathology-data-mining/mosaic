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
import os

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

# Global variable to store the model data directory path
_MODEL_DATA_DIR = None


def get_data_directory():
    """Get the directory containing model data files.

    Returns the HuggingFace cache directory path for the model repository,
    or falls back to local 'data/' directory if not yet downloaded.

    Returns:
        Path: Path to the model data directory
    """
    global _MODEL_DATA_DIR
    if _MODEL_DATA_DIR is not None:
        return _MODEL_DATA_DIR

    # Check if environment variable is set
    if "MOSAIC_DATA_DIR" in os.environ:
        _MODEL_DATA_DIR = Path(os.environ["MOSAIC_DATA_DIR"])
        return _MODEL_DATA_DIR

    # Check if local data/ directory exists (for development/backward compat)
    local_data = Path("data")
    if local_data.exists() and (local_data / "paladin_model_map.csv").exists():
        _MODEL_DATA_DIR = local_data
        return _MODEL_DATA_DIR

    # Fall back to repo root data/ directory
    _MODEL_DATA_DIR = local_data
    return _MODEL_DATA_DIR


def download_and_process_models():
    """Download models from HuggingFace and initialize cancer subtype mappings.

    Downloads the Paladin and Aeon models from the PDM-Group HuggingFace repository
    to the HuggingFace cache directory and creates mappings between cancer subtype
    names and OncoTree codes.

    Returns:
        tuple: (cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes)
            - cancer_subtype_name_map: Dict mapping display names to OncoTree codes
            - reversed_cancer_subtype_name_map: Dict mapping OncoTree codes to display names
            - cancer_subtypes: List of all supported cancer subtype codes
    """
    global _MODEL_DATA_DIR

    # Download to HF cache directory (not local_dir)
    # This returns the path to the cached snapshot
    logger.info("Downloading models from HuggingFace Hub to cache directory...")
    cache_dir = snapshot_download(
        repo_id="PDM-Group/paladin-aeon-models",
        # No local_dir - use HF cache
    )
    _MODEL_DATA_DIR = Path(cache_dir)
    logger.info(f"Models downloaded to: {_MODEL_DATA_DIR}")

    # Also set environment variable for other modules to use
    os.environ["MOSAIC_DATA_DIR"] = str(_MODEL_DATA_DIR)

    model_map = pd.read_csv(
        _MODEL_DATA_DIR / "paladin_model_map.csv",
    )
    cancer_subtypes = model_map["cancer_subtype"].unique().tolist()
    cancer_subtype_name_map = {"Unknown": "UNK"}
    cancer_subtype_name_map.update({
        f"{get_oncotree_code_name(code)} ({code})": code for code in cancer_subtypes
    })
    reversed_cancer_subtype_name_map = {
        value: key for key, value in cancer_subtype_name_map.items()
    }
    
    # Set the global maps in the UI module
    set_cancer_subtype_maps(cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes)
    
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
        default="Unknown",
        help="Sex of the patient (for single slide processing)",
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
    args = parser.parse_args()
    if args.debug:
        logger.add("debug.log", level="DEBUG")
        logger.debug("Debug logging enabled")

    cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes = download_and_process_models()

    if args.slide_path and not args.slide_csv:
        # Single slide processing mode
        if not args.output_dir:
            raise ValueError("Please provide --output-dir to save results")
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
        settings_df = validate_settings(settings_df, cancer_subtype_name_map, cancer_subtypes, reversed_cancer_subtype_name_map)
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
        # Batch processing mode

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        all_paladin_results = []
        all_aeon_results = []
        settings_df = load_settings(args.slide_csv)
        settings_df = validate_settings(settings_df, cancer_subtype_name_map, cancer_subtypes, reversed_cancer_subtype_name_map)
        for idx, row in settings_df.iterrows():
            slide_path = row["Slide"]
            seg_config = row["Segmentation Config"]
            site_type = row["Site Type"]
            sex = row.get("Sex", "Unknown")
            tissue_site = row.get("Tissue Site", "Unknown")
            cancer_subtype = row["Cancer Subtype"]
            ihc_subtype = row.get("IHC Subtype", "")
            logger.info(
                f"Processing slide {slide_path} ({idx + 1} of {len(settings_df)})"
            )
            slide_mask, aeon_results, paladin_results = analyze_slide(
                slide_path,
                seg_config,
                site_type,
                sex,
                tissue_site,
                cancer_subtype,
                cancer_subtype_name_map,
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
