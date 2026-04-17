"""Mosaic command-line interface entry point.

Routes command-line invocations to the appropriate mode:
- Single slide processing  (--slide-path)
- Batch processing         (--slide-csv)
- Web interface            (default, no slide arguments)

Heavy logic lives in dedicated sub-modules:
  mosaic.model_download   — HuggingFace model/metadata downloading
  mosaic.cli.args         — argument parser construction
  mosaic.cli.runners      — single-slide and batch execution
"""

import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from loguru import logger

from mosaic.data_directory import get_data_directory
from mosaic.cli.args import build_parser
from mosaic.cli.runners import run_batch, run_single_slide
from mosaic.model_download import download_and_process_models, download_metadata
from mosaic.ui import launch_gradio
from mosaic.ui.pipeline import set_cancer_subtype_maps
from mosaic.ui.utils import get_oncotree_code_name


def main():
    """Entry point: parse arguments and dispatch to the correct mode."""
    parser = build_parser()
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

    is_launching_ui = not args.slide_path and not args.slide_csv

    if not getattr(args, "skip_model_download", False):
        if is_launching_ui:
            # UI mode: metadata only (fast); models download in background
            (
                cancer_subtype_name_map,
                reversed_cancer_subtype_name_map,
                cancer_subtypes,
            ) = download_metadata()
        else:
            # CLI mode: download everything synchronously
            (
                cancer_subtype_name_map,
                reversed_cancer_subtype_name_map,
                cancer_subtypes,
            ) = download_and_process_models()
    else:
        logger.info("Skipping model download, using cached models...")
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
        set_cancer_subtype_maps(
            cancer_subtype_name_map, reversed_cancer_subtype_name_map, cancer_subtypes
        )

    if args.slide_path and not args.slide_csv:
        run_single_slide(
            args,
            cancer_subtype_name_map,
            cancer_subtypes,
            reversed_cancer_subtype_name_map,
        )
    elif args.slide_csv:
        run_batch(
            args,
            cancer_subtype_name_map,
            cancer_subtypes,
            reversed_cancer_subtype_name_map,
        )
    else:
        launch_gradio(
            server_name=args.server_name,
            server_port=args.server_port,
            share=args.share,
        )


if __name__ == "__main__":
    main()
